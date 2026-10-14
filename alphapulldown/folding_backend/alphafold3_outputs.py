"""What an AlphaFold 3 fold leaves in its output directory, and what a rerun reads back.

Each seed's samples, embeddings and distogram are written as the seed finishes; the best
sample is copied under the job name at the end. Resume decides from these files and the fold
record which seeds, or whether the whole fold, are already done.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import json
import logging
import os
import pathlib
import random
import shutil
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import alphafold3.cpp
import numpy as np
from alphafold3.common import folding_input
from alphafold3.model import model, post_processing

from alphapulldown.prediction.resume import (
    changed_settings,
    finished_fold,
    read_fold_record,
    record_fold_complete,
    start_fold_record,
)
from alphapulldown.folding_backend.alphafold3_modelcif import augment_af3_modelcif_file
from alphapulldown.folding_backend.alphafold3_storage import (
    AF3_RANKING_SCORES,
    af3_best_sample_dir,
    af3_ranking_scores,
)
from alphapulldown.utils.feature_metadata import (
    extract_metadata_from_fold_input,
    load_feature_metadata_sidecars,
)
from alphapulldown.utils.file_handling import open_atomic

if TYPE_CHECKING:
    from alphapulldown.folding_backend.alphafold3_runner import ResultsForSeed


# What AlphaFold 3 writes per sample, and again under the job name for the best one.
_SAMPLE_OUTPUTS = ('model.cif', 'confidences.json', 'summary_confidences.json')


@contextlib.contextmanager
def _directory_written_whole(path: os.PathLike[str] | str):
    """Fill ``<path>.partial``, then put it in place of ``path`` once complete."""
    partial = f'{os.fspath(path)}.partial'
    shutil.rmtree(partial, ignore_errors=True)
    os.makedirs(partial)
    yield partial
    shutil.rmtree(path, ignore_errors=True)
    os.replace(partial, path)


def _write_ranking_rows(
    output_dir: os.PathLike[str] | str, seed: int, rows: Sequence[tuple[int, int, float]]
) -> None:
    """Put a seed's samples into ranking_scores.csv, replacing any rows it had."""
    kept = [row for row in af3_ranking_scores(output_dir) if row[0] != seed]
    # As AlphaFold 3 writes it: text mode with csv's own line endings.
    with open_atomic(os.path.join(output_dir, AF3_RANKING_SCORES)) as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'sample', 'ranking_score'])
        writer.writerows(kept + list(rows))


def _augment_modelcif(cif_path: str, metadata_records: Sequence[dict]) -> None:
    try:
        augment_af3_modelcif_file(cif_path, metadata_records)
    except (OSError, ValueError) as exc:
        logging.warning(
            "Could not add AlphaPulldown metadata to %s: %s",
            cif_path,
            exc,
        )


def write_seed_outputs(
    results_for_seed: ResultsForSeed,
    output_dir: os.PathLike[str] | str,
    job_name: str,
    modelcif_metadata: Sequence[dict] | None = None,
) -> None:
    """Write a seed's samples, embeddings and distogram as soon as it is predicted.

    Each directory appears whole and the ranking comes last, so a ranked seed is on
    disk. ModelCIF metadata goes into each sample once; the best model is a copy.
    """
    seed = results_for_seed.seed
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for sample_idx, result in enumerate(results_for_seed.inference_results):
        result = _make_viewer_compatible_inference_result(result)
        sample_path = os.path.join(output_dir, f'seed-{seed}_sample-{sample_idx}')
        with _directory_written_whole(sample_path) as sample_dir:
            post_processing.write_output(
                inference_result=result, output_dir=sample_dir
            )
            _augment_confidence_json_with_author_numbering(
                os.path.join(sample_dir, 'confidences.json'),
                result,
            )
            if modelcif_metadata:
                _augment_modelcif(os.path.join(sample_dir, 'model.cif'), modelcif_metadata)
        rows.append((seed, sample_idx, float(result.metadata['ranking_score'])))

    # Optionally write embeddings/distogram per-seed if present
    if results_for_seed.embeddings:
        embeddings_path = os.path.join(output_dir, f'seed-{seed}_embeddings')
        with _directory_written_whole(embeddings_path) as embeddings_dir:
            post_processing.write_embeddings(
                embeddings=results_for_seed.embeddings,
                output_dir=embeddings_dir,
                name=f'{job_name}_seed-{seed}',
            )
    if results_for_seed.distogram is not None:
        with _directory_written_whole(
            os.path.join(output_dir, f'seed-{seed}_distogram')
        ) as dist_dir:
            dist_path = os.path.join(dist_dir, f'{job_name}_seed-{seed}_distogram.npz')
            with open(dist_path, 'wb') as f:
                np.savez_compressed(f, distogram=results_for_seed.distogram.astype(np.float16))
    _write_ranking_rows(output_dir, seed, rows)


def write_fold_outputs(output_dir: os.PathLike[str] | str, job_name: str) -> None:
    """Copy the best ranked sample under the job name, with the terms of use: the same
    bytes AlphaFold 3 writes there, also for samples of an interrupted earlier run."""
    best = af3_best_sample_dir(output_dir)
    if best is None:
        logging.warning(f'No ranked samples in {output_dir}; writing no best model.')
        return
    for name in _SAMPLE_OUTPUTS:
        with open(os.path.join(output_dir, best, name), 'rb') as source, open_atomic(
            os.path.join(output_dir, f'{job_name}_{name}'), 'wb'
        ) as target:
            shutil.copyfileobj(source, target)
    output_terms = (
        pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
    ).read_text()
    with open_atomic(os.path.join(output_dir, 'TERMS_OF_USE.md')) as f:
        f.write(output_terms)


def _duplicate_occurrence_to_insertion_code(
    occurrence_index: int,
    *,
    strict: bool = True,
) -> str:
    """Maps the Nth occurrence of a residue ID to a mmCIF insertion code."""
    if occurrence_index <= 1:
        return '.'
    offset = occurrence_index - 2
    if offset >= 26:
        if strict:
            raise ValueError(
                'More than 27 repeated residue occurrences in one chain are not '
                'supported for mmCIF insertion-code output.'
            )
        return '.'
    return chr(ord('A') + offset)


def _residue_author_ids(struc) -> list[str]:
    """Returns author-facing residue IDs, falling back to residue IDs if unset."""
    author_residue_ids = [str(residue_id) for residue_id in struc.residues_table.auth_seq_id]
    if all(residue_id in {".", "?"} for residue_id in author_residue_ids):
        return [str(int(residue_id)) for residue_id in struc.residues_table.id]
    return author_residue_ids


def _existing_insertion_codes(struc) -> list[str]:
    """Returns normalised residue insertion codes from a structure."""
    return [
        "."
        if insertion_code in {".", "?", ""}
        else str(insertion_code)
        for insertion_code in struc.residues_table.insertion_code
    ]


def _author_ids_with_insertion_codes(
    chain_ids: Sequence[str],
    author_residue_ids: Sequence[str],
    existing_insertion_codes: Sequence[str] | None = None,
    *,
    strict: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Returns author IDs, insertion codes, and combined author labels."""
    occurrence_count_by_residue: dict[tuple[str, str], int] = {}
    insertion_codes: list[str] = []
    combined_labels: list[str] = []

    for index, (chain_id, residue_id) in enumerate(
        zip(chain_ids, author_residue_ids, strict=True)
    ):
        explicit_insertion_code = "."
        overflow_occurrence = 0
        if existing_insertion_codes is not None:
            explicit_insertion_code = existing_insertion_codes[index]

        if explicit_insertion_code not in {".", "?", ""}:
            insertion_code = explicit_insertion_code
        else:
            key = (chain_id, residue_id)
            occurrence = occurrence_count_by_residue.get(key, 0) + 1
            occurrence_count_by_residue[key] = occurrence
            insertion_code = _duplicate_occurrence_to_insertion_code(
                occurrence,
                strict=strict,
            )
            if insertion_code == "." and occurrence > 1:
                overflow_occurrence = occurrence

        insertion_codes.append(insertion_code)
        if insertion_code == ".":
            if overflow_occurrence and not strict:
                combined_labels.append(f"{residue_id}[{overflow_occurrence}]")
            else:
                combined_labels.append(residue_id)
        else:
            combined_labels.append(f"{residue_id}{insertion_code}")

    return list(author_residue_ids), insertion_codes, combined_labels


def _coerce_json_scalar(value: str) -> int | str:
    """Converts a stringified integer back to int where possible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _augment_confidence_json_with_author_numbering(
    confidences_path: os.PathLike[str] | str,
    inference_result: model.InferenceResult,
) -> None:
    """Adds preserved author numbering to the confidence sidecar JSON."""
    token_auth_res_ids = inference_result.metadata.get("token_auth_res_ids")
    token_pdb_ins_codes = inference_result.metadata.get("token_pdb_ins_codes")
    token_auth_res_labels = inference_result.metadata.get("token_auth_res_labels")
    if (
        token_auth_res_ids is None
        or token_pdb_ins_codes is None
        or token_auth_res_labels is None
    ):
        return

    with open(confidences_path, "rt", encoding="utf-8") as handle:
        confidence_data = json.load(handle)

    confidence_data["token_label_seq_ids"] = [
        int(token_id) for token_id in confidence_data.get("token_res_ids", [])
    ]
    confidence_data["token_auth_res_ids"] = [
        _coerce_json_scalar(str(token_id)) for token_id in token_auth_res_ids
    ]
    confidence_data["token_pdb_ins_codes"] = [str(code) for code in token_pdb_ins_codes]
    confidence_data["token_auth_res_labels"] = [
        str(label) for label in token_auth_res_labels
    ]

    with open(confidences_path, "wt", encoding="utf-8") as handle:
        json.dump(confidence_data, handle, indent=1)
        handle.write("\n")


def _make_viewer_compatible_inference_result(
    inference_result: model.InferenceResult,
) -> model.InferenceResult:
    """Creates a viewer-safe copy with sequential label IDs and preserved auth IDs."""
    struc = inference_result.predicted_structure
    residue_chain_ids = [
        str(chain_id)
        for chain_id in struc.chains_table.apply_array_to_column(
            column_name='id',
            arr=struc.residues_table.chain_key,
        )
    ]
    author_residue_ids = _residue_author_ids(struc)
    existing_insertion_codes = _existing_insertion_codes(struc)

    sequential_label_ids = np.asarray(
        _sequential_residue_ids_per_chain(residue_chain_ids),
        dtype=np.int32,
    )
    (
        author_residue_ids,
        insertion_codes,
        _,
    ) = _author_ids_with_insertion_codes(
        residue_chain_ids,
        author_residue_ids,
        existing_insertion_codes,
    )

    viewer_structure = struc.copy_and_update_residues(
        res_id=sequential_label_ids,
        res_auth_seq_id=np.asarray(author_residue_ids, dtype=object),
        res_insertion_code=np.asarray(insertion_codes, dtype=object),
    )

    metadata = dict(inference_result.metadata)
    token_chain_ids = [
        str(chain_id)
        for chain_id in metadata.get("token_chain_ids", [])
    ]
    if token_chain_ids and "token_res_ids" in metadata:
        token_author_ids = [str(token_id) for token_id in metadata["token_res_ids"]]
        (
            token_author_ids,
            token_insertion_codes,
            token_author_labels,
        ) = _author_ids_with_insertion_codes(
            token_chain_ids,
            token_author_ids,
            strict=False,
        )
        metadata["token_res_ids"] = _sequential_residue_ids_per_chain(token_chain_ids)
        metadata["token_auth_res_ids"] = token_author_ids
        metadata["token_pdb_ins_codes"] = token_insertion_codes
        metadata["token_auth_res_labels"] = token_author_labels
    return dataclasses.replace(
        inference_result,
        predicted_structure=viewer_structure,
        metadata=metadata,
    )


def _sequential_residue_ids_per_chain(chain_ids: Sequence[str]) -> list[int]:
    """Returns sequential residue IDs that are unique within each chain."""
    next_residue_id_by_chain: dict[str, int] = {}
    residue_ids = []
    for chain_id in chain_ids:
        next_residue_id = next_residue_id_by_chain.get(chain_id, 0) + 1
        next_residue_id_by_chain[chain_id] = next_residue_id
        residue_ids.append(next_residue_id)
    return residue_ids


def _earlier_fold_seeds(
    output_dir: os.PathLike[str] | str, job_name: str, record: dict
) -> list[int]:
    """The seeds of an earlier run of fold ``job_name`` in ``output_dir``, if any."""
    settings = record.get('settings') or {}
    if settings.get('job_name') == job_name and settings.get('seeds'):
        return [int(seed) for seed in settings['seeds']]
    # Output of a version without fold records: a finished fold lists its seeds here.
    if not record and os.path.isfile(os.path.join(output_dir, f'{job_name}_model.cif')):
        return sorted({seed for seed, _, _ in af3_ranking_scores(output_dir)})
    return []


def _fold_seed(
    requested_seed: int | None,
    earlier_seeds: Sequence[int],
    output_dir: os.PathLike[str] | str,
) -> int:
    """The first rng seed of a fold: the earlier run's when resuming, else the requested
    one, else a new one. A requested seed must not mix a second run into the first."""
    if earlier_seeds:
        if requested_seed is not None and requested_seed != earlier_seeds[0]:
            raise ValueError(
                f'{output_dir} holds a prediction of this fold with seed(s) '
                f'{list(earlier_seeds)}, which --random_seed={requested_seed} would mix with '
                f'new ones. Pass --random_seed={earlier_seeds[0]} to resume it, write to '
                'another output directory, or pass --allow_resume=false to predict it again.'
            )
        return earlier_seeds[0]
    if requested_seed is not None:
        return requested_seed
    return random.randrange(2**32 - 1)


def _fold_settings(
    fold_input: folding_input.Input,
    model_flags: dict[str, Any],
    postprocess_flags: dict[str, Any],
) -> dict[str, Any]:
    """What a fold's record holds: the settings that shape its outputs."""
    return {
        'backend': 'alphafold3',
        'job_name': fold_input.sanitised_name(),
        'seeds': list(fold_input.rng_seeds),
        'num_diffusion_samples': model_flags.get('num_diffusion_samples'),
        'num_recycles': model_flags.get('num_recycles'),
        'return_embeddings': bool(model_flags.get('return_embeddings')),
        'return_distogram': bool(model_flags.get('return_distogram')),
        'storage_mode': postprocess_flags.get('storage_mode', 'vanilla'),
        'convert_to_modelcif': postprocess_flags.get('convert_to_modelcif', True),
    }


def _complete_without_record(output_dir: os.PathLike[str] | str, settings: dict) -> bool:
    """Whether output of a version without fold records holds this whole fold: its best
    model, and every sample of these seeds ranked, with model and summary."""
    num_samples = settings['num_diffusion_samples']
    if not num_samples or not os.path.isfile(
        os.path.join(output_dir, f"{settings['job_name']}_model.cif")
    ):
        return False
    expected = {(seed, sample) for seed in settings['seeds'] for sample in range(num_samples)}
    ranked = {(seed, sample) for seed, sample, _ in af3_ranking_scores(output_dir)}
    if ranked != expected:
        return False
    return all(
        os.path.isfile(os.path.join(output_dir, f'seed-{seed}_sample-{sample}', name))
        for seed, sample in expected
        for name in ('model.cif', 'summary_confidences.json')
    )


def _finished_seeds(output_dir: os.PathLike[str] | str, settings: dict) -> list[int]:
    """The seeds an interrupted run wrote out whole. write_seed_outputs ranks a seed
    only once all of it is on disk; its directories are checked as well."""
    num_samples = settings['num_diffusion_samples'] or 0
    ranked = {(seed, sample) for seed, sample, _ in af3_ranking_scores(output_dir)}
    extras = [kind for kind in ('embeddings', 'distogram') if settings[f'return_{kind}']]

    def finished(seed: int) -> bool:
        names = [f'seed-{seed}_sample-{sample}' for sample in range(num_samples)]
        names += [f'seed-{seed}_{kind}' for kind in extras]
        return num_samples > 0 and all(
            (seed, sample) in ranked for sample in range(num_samples)
        ) and all(os.path.isdir(os.path.join(output_dir, name)) for name in names)

    return [seed for seed in settings['seeds'] if finished(seed)]


def _modelcif_metadata(
    fold_input: folding_input.Input, output_dir: os.PathLike[str] | str
) -> list[dict]:
    """The AlphaPulldown metadata for the fold's ModelCIF files, each record once."""
    metadata_records = extract_metadata_from_fold_input(fold_input)
    try:
        metadata_records.extend(load_feature_metadata_sidecars(output_dir))
    except (OSError, ValueError) as exc:
        logging.warning(
            "Could not read copied feature metadata from %s: %s",
            output_dir,
            exc,
        )
    # AF2 metadata may be present both in the transported description and as a copied
    # sidecar. Deduplicate exact records before adding ModelCIF categories.
    unique_metadata = []
    seen_metadata = set()
    for metadata in metadata_records:
        fingerprint = json.dumps(metadata, sort_keys=True, default=str)
        if fingerprint not in seen_metadata:
            seen_metadata.add(fingerprint)
            unique_metadata.append(metadata)
    return unique_metadata


def _start_or_resume_fold(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
    requested_seed: int | None,
    allow_resume: bool,
    model_flags: dict[str, Any],
    postprocess_flags: dict[str, Any],
) -> tuple[folding_input.Input, list[int]] | None:
    """The fold input with its seeds, and the seeds an interrupted run already wrote
    out, recorded as being predicted; None if the fold is already complete."""
    job_name = fold_input.sanitised_name()
    record = read_fold_record(output_dir) if allow_resume else {}
    earlier_seeds = _earlier_fold_seeds(output_dir, job_name, record) if allow_resume else []
    fold_input = dataclasses.replace(
        fold_input, rng_seeds=[_fold_seed(requested_seed, earlier_seeds, output_dir)]
    )
    num_seeds = model_flags.get('num_seeds')
    if num_seeds is not None:
        logging.info(f'Expanding fold job {fold_input.name} to {num_seeds} seeds')
        fold_input = fold_input.with_multiple_seeds(num_seeds)

    settings = _fold_settings(fold_input, model_flags, postprocess_flags)
    seeds_done = []
    if allow_resume:
        if not record and _complete_without_record(output_dir, settings):
            record_fold_complete(output_dir, settings)
            logging.info(f'{job_name} is already complete in {output_dir} (older version).')
            return None
        if finished_fold(record, settings, job_name, output_dir):
            return None
        if record and not record.get('complete') and not changed_settings(record, settings):
            seeds_done = _finished_seeds(output_dir, settings)
    if not seeds_done:
        # A new prediction: the ranking of an earlier one must not mix into it.
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(output_dir, AF3_RANKING_SCORES))
    start_fold_record(output_dir, settings)
    return fold_input, seeds_done
