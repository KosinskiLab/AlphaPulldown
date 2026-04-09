"""Helpers for plotting AlphaPulldown diagnostic figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt

from af2plots.plotter import plotter
from colabfold.plot import plot_msa_v2

from alphapulldown.utils.lightweight_pickles import extract_feature_dict, load_lightweight_pickle


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
    parsed_models = af2_plotter.parse_model_pickles(str(prediction_root))
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

    distogram_result = af2_plotter.plot_distogram(parsed_models, dpi=dpi)
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
            if list(input_path.glob("result*.pkl")):
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
                f"{input_path} does not contain result*.pkl files or a features.pkl file"
            )

        written_paths.append(save_msa_coverage_plot(input_path, destination, dpi=dpi))

    return written_paths
