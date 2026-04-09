"""Helpers for plotting AlphaPulldown diagnostic figures."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt

from af2plots.plotter import plotter
from colabfold.plot import plot_msa_v2

from alphapulldown.utils.lightweight_pickles import extract_feature_dict, load_lightweight_pickle


_RESULT_PICKLE_PATTERNS = ("result*.pkl", "result*.pkl.gz", "result*.pkl.xz")
_RESULT_PICKLE_PATTERN = re.compile(
    r"^result(?P<jobid>_[\w\d]+)?_model_(?P<idx>\d+)(?:_\w+)?\.pkl(?:\.(?:gz|xz))?$"
)


def _normalise_stem(path: str | Path) -> str:
    input_path = Path(path)
    name = input_path.name
    for suffix in (".pkl.xz", ".pkl.gz", ".pkl", ".json", ".xz", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name or input_path.stem


def _ensure_output_dir(output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _infer_asym_id_from_result_pickle(result_pickle: str | Path) -> tuple[list[int], int] | None:
    payload = load_lightweight_pickle(result_pickle)
    if not isinstance(payload, dict):
        return None
    seqs = payload.get("seqs")
    if not isinstance(seqs, list) or not all(isinstance(sequence, str) for sequence in seqs):
        return None

    asym_id: list[int] = []
    for index, sequence in enumerate(seqs, start=1):
        asym_id.extend([index] * len(sequence))
    return asym_id, len(seqs)


def _ensure_chain_metadata(parsed_models: dict[str, dict]) -> None:
    for model_data in parsed_models.values():
        if "asym_id" in model_data and "assembly_num_chains" in model_data:
            continue
        inferred = _infer_asym_id_from_result_pickle(model_data["fn"])
        if inferred is None:
            continue
        asym_id, assembly_num_chains = inferred
        model_data["asym_id"] = np.asarray(asym_id, dtype=np.int32)
        model_data["assembly_num_chains"] = assembly_num_chains


def _find_result_pickles(prediction_dir: str | Path) -> list[Path]:
    prediction_root = Path(prediction_dir)
    suffix_priority = {".pkl": 0, ".gz": 1, ".xz": 2}
    selected_paths: dict[str, Path] = {}

    for pattern in _RESULT_PICKLE_PATTERNS:
        for path in prediction_root.glob(pattern):
            if _RESULT_PICKLE_PATTERN.fullmatch(path.name) is None:
                continue
            key = _normalise_stem(path)
            current = selected_paths.get(key)
            if current is None or suffix_priority[path.suffix] < suffix_priority[current.suffix]:
                selected_paths[key] = path

    return [selected_paths[key] for key in sorted(selected_paths)]


def _parse_prediction_pickles(prediction_dir: str | Path) -> dict[str, dict]:
    parsed_models: dict[str, dict] = {}

    for result_pickle in _find_result_pickles(prediction_dir):
        match = _RESULT_PICKLE_PATTERN.fullmatch(result_pickle.name)
        if match is None:
            continue

        data = load_lightweight_pickle(result_pickle)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict payload in {result_pickle}, got {type(data)!r}")

        if "ptm" in data:
            ptm = float(data["ptm"])
        elif "ranking_confidence" in data:
            ptm = float(data["ranking_confidence"])
        else:
            ptm = float(np.mean(data["plddt"], dtype=float))

        parsed_models[str(result_pickle)] = {
            "datadir": str(prediction_dir),
            "fn": str(result_pickle),
            "idx": int(match.group("idx")),
            "ptm": ptm,
            "iptm": data.get("iptm"),
            "distogram": data.get("distogram"),
            "sm_contacts": data.get("sm_contacts"),
            "pae": data.get("predicted_aligned_error"),
            "plddt": data["plddt"],
        }

    if not parsed_models:
        raise FileNotFoundError(
            f"{prediction_dir} does not contain result*.pkl, result*.pkl.gz, or result*.pkl.xz files"
        )

    for rank, path in enumerate(
        sorted(parsed_models, key=lambda item: parsed_models[item]["ptm"], reverse=True)
    ):
        parsed_models[path]["rank"] = rank + 1
        parsed_models[path]["description"] = f"ranked_{rank}.pdb pTM={parsed_models[path]['ptm']:.2f}"
        if parsed_models[path]["iptm"] is not None:
            parsed_models[path]["description"] += f" iPTM={parsed_models[path]['iptm']:.2f}"

    return parsed_models


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=axis, keepdims=True)


def _plot_distogram_fallback(
    parsed_models: dict[str, dict],
    *,
    dpi: int = 100,
    distance: float = 8.0,
) -> tuple[plt.Figure, list[str]] | None:
    top_model = next(
        (model for model in parsed_models.values() if model.get("rank") == 1),
        max(parsed_models.values(), key=lambda model: float(model["ptm"]), default=None),
    )
    if top_model is None:
        return None

    predicted_distogram = top_model.get("distogram")
    if not isinstance(predicted_distogram, dict):
        return None

    logits = predicted_distogram.get("logits")
    bin_edges = predicted_distogram.get("bin_edges")
    if logits is None or bin_edges is None:
        return None

    logits_array = np.asarray(logits)
    bin_edges_array = np.asarray(bin_edges)
    if logits_array.ndim != 3 or bin_edges_array.ndim != 1:
        return None

    upper_bounds = np.concatenate([bin_edges_array, [np.inf]])
    threshold_mask = upper_bounds < float(np.clip(distance, 3.0, 20.0))
    if not np.any(threshold_mask):
        threshold_mask[0] = True

    contact_probabilities = _softmax(logits_array, axis=-1)[..., threshold_mask].sum(axis=-1)

    figure, axis = plt.subplots(figsize=(8, 8), dpi=dpi)
    image = axis.imshow(
        contact_probabilities,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        extent=(0, contact_probabilities.shape[0], contact_probabilities.shape[0], 0),
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel(f"Probability(distance<{int(distance)}A)")
    axis.set_title("Predicted contacts")
    axis.set_xlabel("Residue number")
    axis.set_ylabel("Residue number")

    asym_id = top_model.get("asym_id")
    assembly_num_chains = top_model.get("assembly_num_chains")
    if asym_id is not None and assembly_num_chains is not None:
        asym_id_array = np.asarray(asym_id)
        for chain_index in range(int(assembly_num_chains) - 1):
            chain_positions = np.where(asym_id_array == (chain_index + 1))[0]
            if chain_positions.size == 0:
                continue
            chain_cut = int(chain_positions.max())
            axis.axvline(x=chain_cut, ls="--", c="k", lw=1)
            axis.axhline(y=chain_cut, ls="--", c="k", lw=1)

    return figure, []


def _plot_distogram_compat(
    af2_plotter: object,
    parsed_models: dict[str, dict],
    *,
    dpi: int = 100,
) -> tuple[plt.Figure, list[str]] | None:
    plot_distogram = getattr(af2_plotter, "plot_distogram", None)
    if callable(plot_distogram):
        return plot_distogram(parsed_models, dpi=dpi)
    return _plot_distogram_fallback(parsed_models, dpi=dpi)


def save_msa_coverage_plot(
    feature_pickle: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 100,
    output_stem: str | None = None,
) -> Path:
    """Save a ColabFold-style MSA coverage plot from a feature pickle."""

    payload = load_lightweight_pickle(feature_pickle)
    feature_dict = extract_feature_dict(payload)
    destination = _ensure_output_dir(output_dir)
    plot_module = plot_msa_v2(feature_dict, dpi=dpi)
    output_path = destination / f"{output_stem or _normalise_stem(feature_pickle)}_msa_coverage.png"
    plot_module.savefig(output_path, bbox_inches="tight")
    plot_module.close()
    return output_path


def save_prediction_plots(
    prediction_dir: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 100,
) -> list[Path]:
    """Save pLDDT, PAE, and distogram plots from a prediction directory."""

    prediction_root = Path(prediction_dir)
    destination = _ensure_output_dir(output_dir)
    af2_plotter = plotter()
    parsed_models = _parse_prediction_pickles(prediction_root)
    _ensure_chain_metadata(parsed_models)
    output_prefix = destination / prediction_root.name

    written_paths: list[Path] = []

    pae_figure = af2_plotter.plot_predicted_alignment_error(parsed_models, dpi=dpi)
    pae_path = output_prefix.with_name(f"{output_prefix.name}_pae.png")
    pae_figure.savefig(pae_path, bbox_inches="tight")
    plt.close(pae_figure)
    written_paths.append(pae_path)

    plddt_figure = af2_plotter.plot_plddts(parsed_models, dpi=dpi)
    plddt_path = output_prefix.with_name(f"{output_prefix.name}_plddt.png")
    plddt_figure.savefig(plddt_path, bbox_inches="tight")
    plt.close(plddt_figure)
    written_paths.append(plddt_path)

    distogram_result = _plot_distogram_compat(af2_plotter, parsed_models, dpi=dpi)
    if distogram_result is not None:
        distogram_figure, _ = distogram_result
        distogram_path = output_prefix.with_name(f"{output_prefix.name}_distogram.png")
        distogram_figure.savefig(distogram_path, bbox_inches="tight")
        plt.close(distogram_figure)
        written_paths.append(distogram_path)

    return written_paths


def plot_inputs(
    inputs: Iterable[str | Path],
    *,
    output_dir: str | Path | None = None,
    dpi: int = 100,
) -> list[Path]:
    """Dispatch plotting based on the provided input paths."""

    written_paths: list[Path] = []
    for raw_input in inputs:
        input_path = Path(raw_input)
        destination = Path(output_dir) if output_dir is not None else input_path.parent

        if input_path.is_dir():
            if _find_result_pickles(input_path):
                written_paths.extend(save_prediction_plots(input_path, destination, dpi=dpi))
                continue
            feature_pickle = input_path / "features.pkl"
            if feature_pickle.exists():
                written_paths.append(
                    save_msa_coverage_plot(
                        feature_pickle,
                        destination,
                        dpi=dpi,
                        output_stem=input_path.name,
                    )
                )
                continue
            raise FileNotFoundError(
                f"{input_path} does not contain result*.pkl(.gz/.xz) files or a features.pkl file"
            )

        written_paths.append(save_msa_coverage_plot(input_path, destination, dpi=dpi))

    return written_paths
