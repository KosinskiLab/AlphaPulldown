"""
Implements structure prediction backend using AlphaFold 3.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""

import csv
import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import os
import pathlib
import re
import time
import typing
from collections.abc import Sequence
from typing import List, Dict, Union, overload

import alphafold3.cpp
import haiku as hk
import jax
import numpy as np
from alphafold3.common import base_config
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.data import parsers as af3_parsers
from alphafold3.jax.attention import attention
from alphafold3.model import features, params, post_processing
from alphafold3.model import model
from alphafold3.model.components import utils
from jax import numpy as jnp

from alphafold.common import residue_constants
from alphafold.common.protein import Protein, to_mmcif
from alphapulldown.folding_backend.folding_backend import FoldingBackend
from alphapulldown.objects import MultimericObject, MonomericObject, ChoppedObject
from alphapulldown.utils.af2_to_af3_msa import (
    Af2ToAf3TranslationResult,
    msa_rows_and_deletions_to_a3m,
    translate_af2_individual_chain_features_to_af3_msas_with_stats,
    translate_af2_complex_msa_to_af3_unpaired_chain_msas_with_stats,
)
from alphapulldown.utils.msa_encoding import ids_to_a3m_af3



# -----------------------------------------------------------------------------
# Data Structures and Utility Classes/Functions
# -----------------------------------------------------------------------------

class ConfigurableModel(typing.Protocol):
    """A model with a nested config class."""

    class Config(base_config.BaseConfig):
        ...

    def __call__(self, config: 'Config') -> 'ConfigurableModel':
        ...

    @classmethod
    def get_inference_result(
        cls,
        batch: features.BatchDict,
        # OLD: result: base_model.ModelResult
        result: model.ModelResult,
        target_name: str = '',
    ) -> typing.Iterable[model.InferenceResult]:
        ...


ModelT = typing.TypeVar('ModelT', bound=ConfigurableModel)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
    """Stores the inference results (samples) for a single seed.

    Attributes:
        seed: The seed used to generate the samples.
        inference_results: The inference results, one per sample.
        full_fold_input: The fold input that must also include the results of
                         running the data pipeline - MSA and templates.
    """
    seed: int
    # OLD: Sequence[base_model.InferenceResult]
    inference_results: Sequence[model.InferenceResult]
    full_fold_input: folding_input.Input
    # Optional extras when enabled via config
    embeddings: dict[str, np.ndarray] | None = None
    distogram: np.ndarray | None = None


# -----------------------------------------------------------------------------
# Model Configuration and Runner
# -----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ModelRunner:
    """Helper class to run structure prediction stages."""
    model_class: type[ConfigurableModel]
    config: base_config.BaseConfig
    device: jax.Device
    model_dir: pathlib.Path

    @functools.cached_property
    def model_params(self) -> hk.Params:
        """Loads model parameters from the model directory."""
        return params.get_model_haiku_params(model_dir=self.model_dir)

    @functools.cached_property
    def _model(
        self,
    ) -> typing.Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
        """Creates and JITs a forward pass function for the model."""
        assert isinstance(self.config, self.model_class.Config)

        @hk.transform
        def forward_fn(batch):
            result = self.model_class(self.config)(batch)
            # Attach identifier from the params
            result['__identifier__'] = self.model_params['__meta__']['__identifier__']
            return result

        return functools.partial(
            jax.jit(forward_fn.apply, device=self.device),
            self.model_params
        )

    def run_inference(
        self, featurised_example: features.BatchDict, rng_key: jnp.ndarray
    ) -> model.ModelResult:
        """Runs inference on a featurised example."""
        featurised_example = jax.device_put(
            jax.tree_util.tree_map(
                jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
            ),
            self.device,
        )
        result = self._model(rng_key, featurised_example)
        result = jax.tree_map(np.asarray, result)
        result = jax.tree_map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
            result,
        )
        # Convert identifier from jnp to bytes
        result['__identifier__'] = result['__identifier__'].tobytes()
        return result

    def extract_structures(
        self,
        batch: features.BatchDict,
        # OLD: result: base_model.ModelResult
        result: model.ModelResult,
        target_name: str,
    ) -> List[model.InferenceResult]:
        """Extracts predicted structures from model output."""
        return list(
            self.model_class.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )



def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
    """Writes outputs (CIF, confidences, embeddings, distograms, ranking CSV)."""
    ranking_scores = []
    max_ranking_score = None
    max_ranking_result = None

    output_terms = (
        pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
    ).read_text()

    os.makedirs(output_dir, exist_ok=True)
    for results_for_seed in all_inference_results:
        seed = results_for_seed.seed
        for sample_idx, result in enumerate(results_for_seed.inference_results):
            sample_dir = os.path.join(output_dir, f'seed-{seed}_sample-{sample_idx}')
            os.makedirs(sample_dir, exist_ok=True)
            result = _make_viewer_compatible_inference_result(result)
            post_processing.write_output(
                inference_result=result, output_dir=sample_dir
            )
            _augment_confidence_json_with_author_numbering(
                os.path.join(sample_dir, 'confidences.json'),
                result,
            )
            ranking_score = float(result.metadata['ranking_score'])
            ranking_scores.append((seed, sample_idx, ranking_score))
            if max_ranking_score is None or ranking_score > max_ranking_score:
                max_ranking_score = ranking_score
                max_ranking_result = result

        # Optionally write embeddings/distogram per-seed if present
        if results_for_seed.embeddings:
            embeddings_dir = os.path.join(output_dir, f'seed-{seed}_embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)
            post_processing.write_embeddings(
                embeddings=results_for_seed.embeddings,
                output_dir=embeddings_dir,
                name=f'{job_name}_seed-{seed}',
            )
        if results_for_seed.distogram is not None:
            dist_dir = os.path.join(output_dir, f'seed-{seed}_distogram')
            os.makedirs(dist_dir, exist_ok=True)
            dist_path = os.path.join(dist_dir, f'{job_name}_seed-{seed}_distogram.npz')
            with open(dist_path, 'wb') as f:
                np.savez_compressed(f, distogram=results_for_seed.distogram.astype(np.float16))

    if max_ranking_result is not None:  # True iff ranking_scores non-empty.
        post_processing.write_output(
            inference_result=max_ranking_result,
            output_dir=output_dir,
            terms_of_use=output_terms,
            name=job_name,
        )
        _augment_confidence_json_with_author_numbering(
            os.path.join(output_dir, f'{job_name}_confidences.json'),
            max_ranking_result,
        )
        with open(os.path.join(output_dir, 'ranking_scores.csv'), 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(['seed', 'sample', 'ranking_score'])
            writer.writerows(ranking_scores)


def _duplicate_occurrence_to_insertion_code(occurrence_index: int) -> str:
    """Maps the Nth occurrence of a residue ID to a mmCIF insertion code."""
    if occurrence_index <= 1:
        return '.'
    offset = occurrence_index - 2
    if offset >= 26:
        raise ValueError(
            'More than 27 repeated residue occurrences in one chain are not '
            'supported for mmCIF insertion-code output.'
        )
    return chr(ord('A') + offset)


def _normalise_output_name_fragment(raw_name: str) -> str:
    """Normalises one output-name fragment while preserving readable IDs."""
    cleaned = re.sub(r"\s+", "_", raw_name.strip())
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", cleaned)
    cleaned = cleaned.strip("_.-")
    return cleaned or "ranked_0"


def _collapse_repeated_name_fragments(
    fragments: Sequence[str],
) -> list[str]:
    """Collapses consecutive identical fragments into a readable count suffix."""
    if not fragments:
        return []

    collapsed: list[str] = []
    current_fragment = fragments[0]
    current_count = 1

    for fragment in fragments[1:]:
        if fragment == current_fragment:
            current_count += 1
            continue

        collapsed.append(
            current_fragment
            if current_count == 1
            else f"{current_fragment}__x{current_count}"
        )
        current_fragment = fragment
        current_count = 1

    collapsed.append(
        current_fragment
        if current_count == 1
        else f"{current_fragment}__x{current_count}"
    )
    return collapsed


def _compact_output_job_name(job_name: str, *, max_chars: int = 200) -> str:
    """Keeps job names readable while staying below common filename limits."""
    if len(job_name) <= max_chars:
        return job_name

    digest = hashlib.sha1(job_name.encode("utf-8")).hexdigest()[:12]
    suffix = f"__{digest}"
    prefix = job_name[: max_chars - len(suffix)].rstrip("_.-")
    if not prefix:
        return f"job{suffix}"
    return f"{prefix}{suffix}"


def _compact_existing_compound_name(raw_name: str) -> str:
    """Compacts already-joined `_and_` names such as multimer descriptions."""
    parts = [
        _normalise_output_name_fragment(part)
        for part in raw_name.split("_and_")
        if part.strip("_.-")
    ]
    if not parts:
        return "ranked_0"
    return _compact_output_job_name(
        "_and_".join(_collapse_repeated_name_fragments(parts))
    )


def _regions_to_name_fragment(regions: Sequence[tuple[int, int]]) -> str:
    """Returns a readable name fragment for a set of closed residue intervals."""
    return "_".join(f"{start}-{end}" for start, end in regions)


def _json_input_basename(json_path: str) -> str:
    """Returns a readable basename for an AF3 JSON input path."""
    stem = pathlib.Path(json_path).stem
    for suffix in ("_af3_input", "_input"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or pathlib.Path(json_path).stem


def _object_name_fragment(obj: typing.Any) -> str:
    """Builds a deterministic output-name fragment for one modelling object."""
    if isinstance(obj, dict) and "json_input" in obj:
        fragment = _json_input_basename(obj["json_input"])
        regions = obj.get("regions")
        if isinstance(regions, Sequence) and regions:
            fragment = f"{fragment}__{_regions_to_name_fragment(regions)}"
        return _normalise_output_name_fragment(fragment)

    if isinstance(obj, MultimericObject):
        return _compact_existing_compound_name(obj.description or "multimer")

    if isinstance(obj, (MonomericObject, ChoppedObject)):
        return _normalise_output_name_fragment(obj.description or "monomer")

    if isinstance(obj, folding_input.Input):
        return _compact_existing_compound_name(obj.name)

    return _normalise_output_name_fragment(type(obj).__name__)


def _build_output_job_name(objects_to_model: Sequence[dict]) -> str:
    """Builds a readable AF3 job name from the requested modelling objects."""
    fragments: list[str] = []
    for entry in objects_to_model:
        object_to_model = entry["object"]
        if isinstance(object_to_model, list):
            fragments.extend(_object_name_fragment(obj) for obj in object_to_model)
        else:
            fragments.append(_object_name_fragment(object_to_model))
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        return "ranked_0"
    readable_name = "_and_".join(_collapse_repeated_name_fragments(fragments))
    return _compact_output_job_name(readable_name)


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
) -> tuple[list[str], list[str], list[str]]:
    """Returns author IDs, insertion codes, and combined author labels."""
    occurrence_count_by_residue: dict[tuple[str, str], int] = {}
    insertion_codes: list[str] = []
    combined_labels: list[str] = []

    for index, (chain_id, residue_id) in enumerate(
        zip(chain_ids, author_residue_ids, strict=True)
    ):
        explicit_insertion_code = "."
        if existing_insertion_codes is not None:
            explicit_insertion_code = existing_insertion_codes[index]

        if explicit_insertion_code not in {".", "?", ""}:
            insertion_code = explicit_insertion_code
        else:
            key = (chain_id, residue_id)
            occurrence = occurrence_count_by_residue.get(key, 0) + 1
            occurrence_count_by_residue[key] = occurrence
            insertion_code = _duplicate_occurrence_to_insertion_code(occurrence)

        insertion_codes.append(insertion_code)
        if insertion_code == ".":
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


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
    output_dir: os.PathLike[str] | str | None = None,
    resolve_msa_overlaps: bool = True,
    debug_msas: bool = False,
) -> Sequence[ResultsForSeed]:
    """Run inference (featurisation + model) to predict structures for each seed."""
    logging.info(f'Featurising data for seeds {fold_input.rng_seeds}...')
    featurisation_start_time = time.time()
    ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input,
        buckets=buckets,
        ccd=ccd,
        resolve_msa_overlaps=resolve_msa_overlaps,
        verbose=True,
    )
    logging.info(
        f'Featurising took {time.time() - featurisation_start_time:.2f} seconds.'
    )

    all_inference_start_time = time.time()
    all_inference_results = []
    # Utility: write final complex MSA (after pairing/dedup/merge) to A3M
    def _write_final_msa_a3m(example_batch, seed_value: int):
        try:
            if output_dir is None:
                return
            os.makedirs(output_dir, exist_ok=True)

            def write_from_array(rows: np.ndarray, suffix: str):
                # AF3 uses a different integer alphabet for MSA arrays.
                a3m_text = ids_to_a3m_af3(rows)
                a3m_path = os.path.join(output_dir, f"{fold_input.sanitised_name()}_seed-{seed_value}_{suffix}.a3m")
                with open(a3m_path, 'wt') as f:
                    f.write(a3m_text)
                logging.info(f"Wrote {suffix} A3M to {a3m_path}")

            # Write merged msa
            msa_rows = example_batch.get('msa', None)
            if msa_rows is not None:
                num_alignments = int(example_batch.get('num_alignments', 0)) if 'num_alignments' in example_batch else msa_rows.shape[0]
                if num_alignments > 0:
                    write_from_array(np.asarray(msa_rows)[:num_alignments], 'final_complex_msa')

        except Exception as e:
            logging.error(f"Failed to write final complex MSA A3M: {e}")

    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        logging.info(f'Running model inference for seed {seed}...')
        inference_start_time = time.time()
        # If requested, dump the post-featurisation merged complex MSA for inspection.
        if debug_msas:
            _write_final_msa_a3m(example, seed)

        rng_key = jax.random.PRNGKey(seed)
        result = model_runner.run_inference(example, rng_key)
        logging.info(
            f'Model inference for seed {seed} took {time.time() - inference_start_time:.2f} seconds.'
        )

        logging.info(f'Extracting structures for seed {seed}...')
        extract_start = time.time()
        inference_results = model_runner.extract_structures(
            batch=example, result=result, target_name=fold_input.name
        )
        logging.info(
            f'Extracting structures for seed {seed} took {time.time() - extract_start:.2f} seconds.'
        )

        # Optional: gather embeddings and distogram
        embeddings_out = None
        try:
            # Use token_chain_ids length as number of tokens
            num_tokens = len(inference_results[0].metadata['token_chain_ids']) if inference_results else 0
        except Exception:
            num_tokens = 0
        try:
            if 'single_embeddings' in result and 'pair_embeddings' in result and num_tokens > 0:
                single = np.asarray(result['single_embeddings'])[:num_tokens]
                pair = np.asarray(result['pair_embeddings'])[:num_tokens, :num_tokens]
                embeddings_out = {'single_embeddings': single, 'pair_embeddings': pair}
        except Exception:
            embeddings_out = None
        dist_out = None
        try:
            dist = result.get('distogram', {}).get('distogram', None)  # type: ignore[union-attr]
            if dist is not None and num_tokens > 0:
                dist_out = np.asarray(dist)[:num_tokens, :num_tokens]
        except Exception:
            dist_out = None

        all_inference_results.append(ResultsForSeed(
            seed=seed,
            inference_results=inference_results,
            full_fold_input=fold_input,
            embeddings=embeddings_out,
            distogram=dist_out,
        ))
    logging.info(
        f'Inference + extraction for seeds {fold_input.rng_seeds} took {time.time() - all_inference_start_time:.2f} seconds.'
    )
    return all_inference_results


def process_fold_input(
    fold_input: folding_input.Input,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    resolve_msa_overlaps: bool = True,
    debug_msas: bool = False,
) -> folding_input.Input | Sequence[ResultsForSeed]:
    """Runs data pipeline and/or inference on a single fold input, then writes outputs."""
    logging.info(f'Processing fold input {fold_input.name}')

    # Validation
    if not fold_input.chains:
        logging.error('Fold input has no chains.')
        raise ValueError('Fold input has no chains.')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check model parameters can be loaded
    if model_runner is not None:
        logging.info('Checking model parameters availability...')
        try:
            _ = model_runner.model_params
        except Exception as e:
            logging.error(f'Failed to load model parameters: {e}')
            raise

    # There's no data pipeline here, so skip it:
    logging.info('Skipping data pipeline...')

    # Write input JSON (ensure directory exists)
    os.makedirs(output_dir, exist_ok=True)
    prepared_path = os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json')
    logging.info(f'Writing model input JSON to {prepared_path}')
    with open(prepared_path, 'wt') as f:
        f.write(fold_input.to_json())

    if model_runner is None:
        logging.info('Skipping inference...')
        return fold_input

    # Run inference (writing of outputs is deferred to postprocess)
    logging.info(
        f'Predicting 3D structure for {fold_input.name} with seed(s) {fold_input.rng_seeds}...'
    )
    all_inference_results = predict_structure(
        fold_input=fold_input,
        model_runner=model_runner,
        buckets=buckets,
        output_dir=output_dir,
        resolve_msa_overlaps=resolve_msa_overlaps,
        debug_msas=debug_msas,
    )
    logging.info(f'Done processing fold input {fold_input.name}.')
    return all_inference_results


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AlphaFold3Backend(FoldingBackend):
    """A backend to perform structure prediction using AlphaFold 3."""

    @staticmethod
    def setup(
        num_diffusion_samples: int,
        flash_attention_implementation: str,
        buckets: list,
        jax_compilation_cache_dir: str,
        model_dir: str,
        num_recycles: int = 10,
        return_embeddings: bool = False,
        return_distogram: bool = False,
        **kwargs,
    ) -> Dict:
        """Sets up the ModelRunner with the given configurations."""

        # Suppose we rely on your new code's "model.Model" or a custom class
        from alphafold3.model.model import Model as MyNewModel

        def make_model_config(
            *,
            model_class: type[ModelT] = MyNewModel,
            flash_attention_implementation: attention.Implementation,
            num_diffusion_samples: int = 5,
            num_recycles: int = 10,
            return_embeddings: bool = False,
            return_distogram: bool = False,
        ):
            # The new code approach:
            config = model_class.Config()
            if hasattr(config, 'global_config'):
                config.global_config.flash_attention_implementation = flash_attention_implementation
            if hasattr(config, 'heads') and hasattr(config.heads, 'diffusion'):
                config.heads.diffusion.eval.num_samples = num_diffusion_samples
            # Optional overrides present in upstream AF3 runner
            if hasattr(config, 'num_recycles'):
                config.num_recycles = num_recycles
            if hasattr(config, 'return_embeddings'):
                config.return_embeddings = return_embeddings
            if hasattr(config, 'return_distogram'):
                config.return_distogram = return_distogram
            return config

        if jax_compilation_cache_dir is not None:
            jax.config.update('jax_compilation_cache_dir', jax_compilation_cache_dir)

        gpu_devices = jax.local_devices(backend='gpu')
        if gpu_devices:
            compute_capability = float(gpu_devices[0].compute_capability)
            if compute_capability < 6.0:
                raise ValueError('AlphaFold 3 requires at least GPU compute capability 6.0.')
            elif 7.0 <= compute_capability < 8.0:
                xla_flags = os.environ.get('XLA_FLAGS')
                required_flag = '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'
                if not xla_flags or required_flag not in xla_flags:
                    raise ValueError(
                        'For devices with GPU compute capability 7.x (see'
                        ' https://developer.nvidia.com/cuda-gpus) the ENV XLA_FLAGS must'
                        f' include "{required_flag}".'
                    )
        logging.info(f'Found local devices: {gpu_devices}')
        logging.info('Building model from scratch...')

        model_runner = ModelRunner(
            model_class=MyNewModel,
            config=make_model_config(
                flash_attention_implementation=typing.cast(
                    attention.Implementation, flash_attention_implementation
                ),
                num_diffusion_samples=num_diffusion_samples,
                num_recycles=num_recycles,
                return_embeddings=return_embeddings,
                return_distogram=return_distogram,
            ),
            device=gpu_devices[0],
            model_dir=pathlib.Path(model_dir),
        )
        return {'model_runner': model_runner}

    @staticmethod
    def prepare_input(
        objects_to_model: list,  # Now a list of dicts with 'object' and 'output_dir'
        random_seed: int,
        af3_input_json: list = None,
        features_directory: str = None,
        debug_templates: bool | None = None,
        debug_msas: bool = False,
    ) -> list:
        """Prepare input for AlphaFold3 prediction."""
        logging.info(f"prepare_input called with {len(objects_to_model)} objects to model")
        for i, entry in enumerate(objects_to_model):
            logging.info(f"Object {i}: {entry}")

        @dataclasses.dataclass(frozen=True, slots=True)
        class _TranslatedMsaDebugRecord:
            chain_id: str
            chain_description: str
            chain_length: int
            paired_msa: str
            unpaired_msa: str
            paired_msa_row_count: int
            unpaired_msa_row_count: int
            paired_species_identifier_count: int
            paired_rows_without_species_identifier_count: int
            paired_rows_with_generated_accession_count: int

        @functools.lru_cache(maxsize=None)
        def _supported_chain_kwargs(chain_cls: type) -> frozenset[str]:
            return frozenset(inspect.signature(chain_cls).parameters)

        def _construct_chain(chain_cls: type, **kwargs):
            supported_kwargs = _supported_chain_kwargs(chain_cls)
            filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in supported_kwargs
            }
            return chain_cls(**filtered_kwargs)

        def _clone_chain_with_id(chain, new_id: str):
            description = getattr(chain, "description", None)
            if isinstance(chain, folding_input.ProteinChain):
                return _construct_chain(
                    folding_input.ProteinChain,
                    id=new_id,
                    sequence=chain.sequence,
                    description=description,
                    residue_ids=getattr(chain, "residue_ids", None),
                    ptms=chain.ptms,
                    paired_msa=chain.paired_msa,
                    unpaired_msa=chain.unpaired_msa,
                    templates=chain.templates,
                )
            if isinstance(chain, folding_input.RnaChain):
                return _construct_chain(
                    folding_input.RnaChain,
                    id=new_id,
                    sequence=chain.sequence,
                    description=description,
                    residue_ids=getattr(chain, "residue_ids", None),
                    modifications=chain.modifications,
                    unpaired_msa=chain.unpaired_msa,
                )
            if isinstance(chain, folding_input.DnaChain):
                return _construct_chain(
                    folding_input.DnaChain,
                    id=new_id,
                    sequence=chain.sequence,
                    description=description,
                    residue_ids=getattr(chain, "residue_ids", None),
                    modifications=chain.modifications(),
                )
            if isinstance(chain, folding_input.Ligand):
                return _construct_chain(
                    folding_input.Ligand,
                    id=new_id,
                    ccd_ids=chain.ccd_ids,
                    smiles=chain.smiles,
                    description=description,
                )
            raise TypeError(f"Unsupported chain type: {type(chain)}")

        def _slice_a3m_row_to_region(a3m_row: str, start: int, end: int) -> str:
            query_position = 0
            sliced_chars: list[str] = []
            for char in a3m_row:
                if char.islower():
                    # Lowercase A3M letters are insertions after the current
                    # query position, so keep only those between retained
                    # residues and drop insertions before `start` or after `end`.
                    if start <= query_position < end:
                        sliced_chars.append(char)
                    continue

                query_position += 1
                if query_position < start:
                    continue
                if query_position > end:
                    break
                sliced_chars.append(char)
            return "".join(sliced_chars)

        def _slice_a3m_to_region(
            a3m_text: str | None,
            start: int,
            end: int,
        ) -> str | None:
            if a3m_text in (None, ""):
                return a3m_text

            sequences, descriptions = af3_parsers.parse_fasta(a3m_text)
            sliced_sequences = [
                _slice_a3m_row_to_region(sequence, start, end)
                for sequence in sequences
            ]
            return (
                "\n".join(
                    f">{description}\n{sequence}"
                    for description, sequence in zip(
                        descriptions,
                        sliced_sequences,
                        strict=True,
                    )
                )
                + "\n"
            )

        def _slice_a3m_to_regions(
            a3m_text: str | None,
            regions: Sequence[tuple[int, int]],
        ) -> str | None:
            if a3m_text in (None, "") or not regions:
                return a3m_text

            sequences, descriptions = af3_parsers.parse_fasta(a3m_text)
            sliced_sequences = [
                "".join(
                    _slice_a3m_row_to_region(sequence, start, end)
                    for start, end in regions
                )
                for sequence in sequences
            ]
            return (
                "\n".join(
                    f">{description}\n{sequence}"
                    for description, sequence in zip(
                        descriptions,
                        sliced_sequences,
                        strict=True,
                    )
                )
                + "\n"
            )

        def _slice_templates_to_region(
            templates: Sequence[folding_input.Template] | None,
            start: int,
            end: int,
        ) -> Sequence[folding_input.Template] | None:
            if templates is None:
                return None

            start_index = start - 1
            sliced_templates = []
            for template in templates:
                remapped_indices = {
                    query_index - start_index: template_index
                    for query_index, template_index in template.query_to_template_map.items()
                    if start_index <= query_index < end
                }
                if remapped_indices:
                    sliced_templates.append(
                        folding_input.Template(
                            mmcif=template.mmcif,
                            query_to_template_map=remapped_indices,
                        )
                    )
            return sliced_templates

        def _slice_positioned_modifications_to_region(
            modifications: Sequence[tuple[str, int]],
            start: int,
            end: int,
        ) -> list[tuple[str, int]]:
            return [
                (modification_name, modification_position - start + 1)
                for modification_name, modification_position in modifications
                if start <= modification_position <= end
            ]

        def _slice_positioned_modifications_to_regions(
            modifications: Sequence[tuple[str, int]],
            regions: Sequence[tuple[int, int]],
        ) -> list[tuple[str, int]]:
            sliced_modifications: list[tuple[str, int]] = []
            offset = 0
            for start, end in regions:
                sliced_modifications.extend(
                    (
                        modification_name,
                        offset + modification_position - start + 1,
                    )
                    for modification_name, modification_position in modifications
                    if start <= modification_position <= end
                )
                offset += end - start + 1
            return sliced_modifications

        def _slice_templates_to_regions(
            templates: Sequence[folding_input.Template] | None,
            regions: Sequence[tuple[int, int]],
        ) -> Sequence[folding_input.Template] | None:
            if templates is None:
                return None

            sliced_templates = []
            for template in templates:
                remapped_indices = {}
                offset = 0
                for start, end in regions:
                    start_index = start - 1
                    remapped_indices.update({
                        offset + (query_index - start_index): template_index
                        for query_index, template_index in template.query_to_template_map.items()
                        if start_index <= query_index < end
                    })
                    offset += end - start + 1
                if remapped_indices:
                    sliced_templates.append(
                        folding_input.Template(
                            mmcif=template.mmcif,
                            query_to_template_map=remapped_indices,
                        )
                    )
            return sliced_templates

        def _chain_residue_ids(
            chain: (
                folding_input.ProteinChain
                | folding_input.RnaChain
                | folding_input.DnaChain
                | folding_input.Ligand
            ),
        ) -> list[int] | None:
            residue_ids = getattr(chain, "residue_ids", None)
            if residue_ids is not None:
                return [int(residue_id) for residue_id in residue_ids]
            if isinstance(
                chain,
                (
                    folding_input.ProteinChain,
                    folding_input.RnaChain,
                    folding_input.DnaChain,
                ),
            ):
                return list(range(1, len(chain.sequence) + 1))
            return None

        def _slice_af3_chain_to_regions(
            chain: (
                folding_input.ProteinChain
                | folding_input.RnaChain
                | folding_input.DnaChain
                | folding_input.Ligand
            ),
            *,
            regions: Sequence[tuple[int, int]],
            json_path: str,
        ):
            if isinstance(chain, folding_input.Ligand):
                raise ValueError(
                    f"Region ranges are not supported for ligand AF3 JSON inputs: {json_path}"
                )

            sequence_length = len(chain.sequence)
            for start, end in regions:
                if not 1 <= start <= end <= sequence_length:
                    raise ValueError(
                        f"Requested region {start}-{end} is outside the sequence "
                        f"length {sequence_length} for AF3 JSON input {json_path}."
                    )

            sliced_sequence = "".join(
                chain.sequence[start - 1:end]
                for start, end in regions
            )
            sliced_residue_ids = [
                residue_id
                for start, end in regions
                for residue_id in (_chain_residue_ids(chain) or [])[start - 1:end]
            ]
            if isinstance(chain, folding_input.ProteinChain):
                return _construct_chain(
                    folding_input.ProteinChain,
                    id=chain.id,
                    sequence=sliced_sequence,
                    ptms=_slice_positioned_modifications_to_regions(
                        chain.ptms,
                        regions,
                    ),
                    residue_ids=sliced_residue_ids,
                    unpaired_msa=_slice_a3m_to_regions(
                        chain.unpaired_msa,
                        regions,
                    ),
                    paired_msa=_slice_a3m_to_regions(
                        chain.paired_msa,
                        regions,
                    ),
                    templates=_slice_templates_to_regions(
                        chain.templates,
                        regions,
                    ),
                )

            if isinstance(chain, folding_input.RnaChain):
                return _construct_chain(
                    folding_input.RnaChain,
                    id=chain.id,
                    sequence=sliced_sequence,
                    modifications=_slice_positioned_modifications_to_regions(
                        chain.modifications,
                        regions,
                    ),
                    residue_ids=sliced_residue_ids,
                    unpaired_msa=_slice_a3m_to_regions(
                        chain.unpaired_msa,
                        regions,
                    ),
                )

            if isinstance(chain, folding_input.DnaChain):
                return _construct_chain(
                    folding_input.DnaChain,
                    id=chain.id,
                    sequence=sliced_sequence,
                    modifications=_slice_positioned_modifications_to_regions(
                        chain.modifications(),
                        regions,
                    ),
                    residue_ids=sliced_residue_ids,
                )

            raise TypeError(f"Unsupported chain type for AF3 JSON slicing: {type(chain)}")

        def _expand_json_input_chains(
            *,
            chains: Sequence[
                folding_input.ProteinChain
                | folding_input.RnaChain
                | folding_input.DnaChain
                | folding_input.Ligand
            ],
            json_path: str,
            regions: list[tuple[int, int]] | None,
        ) -> list[
            folding_input.ProteinChain
            | folding_input.RnaChain
            | folding_input.DnaChain
            | folding_input.Ligand
        ]:
            if not regions:
                return list(chains)

            if len(chains) != 1:
                raise ValueError(
                    "Region ranges for AF3 JSON feature inputs require exactly "
                    f"one chain per file, but {json_path} contains {len(chains)} chains."
                )

            merged_chain = _slice_af3_chain_to_regions(
                chains[0],
                regions=regions,
                json_path=json_path,
            )
            logging.info(
                "Collapsed AF3 JSON input %s into one gapped chain for regions %s",
                json_path,
                regions,
            )
            return [merged_chain]
        
        def get_chain_id(index: int) -> str:
            if index < 26:
                return chr(ord('A') + index)
            else:
                first_char = chr(ord('A') + (index // 26) - 1)
                second_char = chr(ord('A') + (index % 26))
                return first_char + second_char

        def get_next_available_chain_id(used_chain_ids: set, chain_id_counter_ref) -> str:
            """Get the next available chain ID that's not already used."""
            while True:
                chain_id = get_chain_id(chain_id_counter_ref[0])
                if chain_id not in used_chain_ids:
                    chain_id_counter_ref[0] += 1
                    return chain_id
                chain_id_counter_ref[0] += 1

        def _write_translated_msa_debug_artifacts(
            *,
            job_name: str,
            output_dir: str,
            chain_records: list[_TranslatedMsaDebugRecord],
            translation_results: list[Af2ToAf3TranslationResult],
        ) -> None:
            if not chain_records or not translation_results:
                return

            os.makedirs(output_dir, exist_ok=True)
            summary = {
                "job_name": job_name,
                "translation_modes": sorted(
                    {result.translation_mode for result in translation_results}
                ),
                "total_rows_considered": int(
                    sum(result.total_rows_considered for result in translation_results)
                ),
                "occupancy_histogram": {
                    "0": int(
                        sum(result.occupancy_histogram.get("0", 0) for result in translation_results)
                    ),
                    "1": int(
                        sum(result.occupancy_histogram.get("1", 0) for result in translation_results)
                    ),
                    "ge_2": int(
                        sum(result.occupancy_histogram.get("ge_2", 0) for result in translation_results)
                    ),
                },
                "paired_row_count": int(sum(result.paired_row_count for result in translation_results)),
                "invalid_paired_rows": int(
                    sum(result.invalid_paired_rows for result in translation_results)
                ),
                "invalid_unpaired_rows": int(
                    sum(result.invalid_unpaired_rows for result in translation_results)
                ),
                "paired_rows_valid": all(
                    result.invalid_paired_rows == 0 for result in translation_results
                ),
                "unpaired_rows_valid": all(
                    result.invalid_unpaired_rows == 0 for result in translation_results
                ),
                "chains": [],
            }

            for record in chain_records:
                chain_id = record.chain_id
                paired_path = os.path.join(
                    output_dir, f"{job_name}_chain-{chain_id}_paired_input.a3m"
                )
                unpaired_path = os.path.join(
                    output_dir, f"{job_name}_chain-{chain_id}_unpaired_input.a3m"
                )
                with open(paired_path, "wt") as handle:
                    handle.write(record.paired_msa)
                with open(unpaired_path, "wt") as handle:
                    handle.write(record.unpaired_msa)

                summary["chains"].append(
                    {
                        "chain_id": chain_id,
                        "chain_description": record.chain_description,
                        "chain_length": int(record.chain_length),
                        "paired_msa_row_count": int(record.paired_msa_row_count),
                        "unpaired_msa_row_count": int(record.unpaired_msa_row_count),
                        "paired_species_identifier_count": int(
                            record.paired_species_identifier_count
                        ),
                        "paired_rows_without_species_identifier_count": int(
                            record.paired_rows_without_species_identifier_count
                        ),
                        "paired_rows_with_generated_accession_count": int(
                            record.paired_rows_with_generated_accession_count
                        ),
                    }
                )

            summary_path = os.path.join(
                output_dir, f"{job_name}_af2_to_af3_translation_summary.json"
            )
            with open(summary_path, "wt") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)

        def insert_release_date_into_mmcif(
            mmcif_string: str, revision_date: str = '2100-01-01'
        ) -> str:
            lines = mmcif_string.splitlines()
            insert_index = None
            for i, line in enumerate(lines):
                if line.startswith('data_'):
                    insert_index = i + 1
                    break
            if insert_index is None:
                return mmcif_string
            revision_lines = [
                "",
                "loop_",
                "_pdbx_audit_revision_history.ordinal",
                "_pdbx_audit_revision_history.data_content_type", 
                "_pdbx_audit_revision_history.major_revision",
                "_pdbx_audit_revision_history.minor_revision",
                "_pdbx_audit_revision_history.revision_date",
                "1 'Structure model' 1 0 " + revision_date,
                ""
            ]
            lines[insert_index:insert_index] = revision_lines
            return '\n'.join(lines)

        def strip_entity_poly_seq_block(mmcif_string: str) -> str:
            """Remove _entity_poly_seq loop to let AF3 reconstruct scheme from _atom_site.

            AF3's parser can infer _pdbx_poly_seq_scheme directly from _atom_site and
            avoids mismatches when our synthetic _entity_poly_seq has UNK entries.
            """
            lines = mmcif_string.splitlines()
            out_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.strip() == 'loop_':
                    # Peek ahead to see if this loop is for _entity_poly_seq
                    j = i + 1
                    is_entity_poly_seq = False
                    while j < len(lines) and lines[j].strip().startswith('_'):
                        if lines[j].strip().startswith('_entity_poly_seq.'):
                            is_entity_poly_seq = True
                        # Stop header scan when we reach first data row (no leading underscore)
                        j += 1
                        if j < len(lines) and not lines[j].strip().startswith('_'):
                            break
                    if is_entity_poly_seq:
                        # Skip until next blank line or next loop_/category header
                        k = j
                        while k < len(lines):
                            if lines[k].strip() == '' or lines[k].strip() == '#':
                                k += 1
                                break
                            if lines[k].strip() == 'loop_':
                                break
                            k += 1
                        i = k
                        continue
                out_lines.append(line)
                i += 1
            return '\n'.join(out_lines)

        def _monomeric_to_chain(
            mono_obj: Union[MonomericObject, ChoppedObject],
            chain_id: str
        ) -> folding_input.ProteinChain:
            sequence = mono_obj.sequence
            feature_dict = mono_obj.feature_dict
            residue_ids = None
            residue_index = feature_dict.get("residue_index")
            if residue_index is not None:
                residue_ids = (
                    np.asarray(residue_index, dtype=np.int32).reshape(-1) + 1
                ).astype(int).tolist()
            msa_array = feature_dict.get('msa')
            deletion_matrix = feature_dict.get('deletion_matrix_int')
            if deletion_matrix is None:
                deletion_matrix = feature_dict.get('deletion_matrix')
            # Standalone AlphaPulldown monomer objects carry only a single custom MSA.
            # MultimericObject instances override this with an AF2->AF3 translation.
            unpaired_msa = (
                msa_rows_and_deletions_to_a3m(
                    msa_rows=np.asarray(msa_array),
                    deletion_rows=None if deletion_matrix is None else np.asarray(deletion_matrix),
                    query_sequence=sequence,
                )
                if msa_array is not None
                else ""
            )
            paired_msa = ""
            templates = []
            if 'template_aatype' in feature_dict:
                num_templates = feature_dict['template_aatype'].shape[0]
                for i in range(num_templates):
                    try:
                        pdb_code_chain = feature_dict["template_domain_names"][i].decode('utf-8')
                        if '_' in pdb_code_chain:
                            pdb_code, chain_id_template = pdb_code_chain.split('_')
                        else:
                            pdb_code = pdb_code_chain
                            chain_id_template = 'A'
                        template_sequence = feature_dict["template_sequence"][i]
                        if isinstance(template_sequence, bytes):
                            template_sequence = template_sequence.decode('utf-8')
                        template_mask = feature_dict["template_all_atom_masks"][i]
                        hh_ids = np.argmax(feature_dict["template_aatype"][i], axis=-1)
                        if np.sum(template_mask) == 0:
                            logging.info(f"Skipping template {i} ({pdb_code_chain}) - no atoms in region")
                            continue
                        tmpl_aatype = np.array([
                            residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[j]
                            for j in hh_ids
                        ], dtype=np.int32)
                        tmpl_aatype[tmpl_aatype == 21] = 20
                        if np.any(tmpl_aatype < 0) or np.any(tmpl_aatype >= len(residue_constants.resnames)):
                            raise ValueError(f"Invalid aatype: {tmpl_aatype}")
                        protein = Protein(
                            atom_positions=feature_dict["template_all_atom_positions"][i],
                            atom_mask=feature_dict["template_all_atom_masks"][i],
                            aatype=tmpl_aatype,
                            residue_index=feature_dict["residue_index"],
                            chain_index=np.zeros_like(feature_dict["residue_index"], dtype=int),
                            b_factors=np.zeros(template_mask.shape, dtype=float),
                        )
                        mmcif_string = to_mmcif(
                            prot=protein, file_id=pdb_code, model_type='Monomer'
                        )
                        new_mmcif_string = insert_release_date_into_mmcif(mmcif_string)
                        # Remove _entity_poly_seq to avoid mismatches during AF3 parsing
                        new_mmcif_string = strip_entity_poly_seq_block(new_mmcif_string)
                        # Optional debug dump of generated template mmCIF
                        if debug_templates:
                            try:
                                os.makedirs("templates_debug", exist_ok=True)
                                debug_fname = os.path.join(
                                    "templates_debug",
                                    f"generated_template_{pdb_code}_{chain_id_template}_chain{chain_id}_idx{i}.cif",
                                )
                                with open(debug_fname, "w") as f_dbg:
                                    f_dbg.write(new_mmcif_string)
                            except Exception:
                                # Best-effort debug; ignore failures
                                pass
                        # Build query->template mapping only for residues that have any atoms
                        # present in the template (avoid gaps causing index errors downstream).
                        try:
                            per_res_atom_mask = feature_dict["template_all_atom_masks"][i]
                            # Boolean mask: True if any atom present for this residue
                            present_res_mask = np.any(per_res_atom_mask > 0, axis=-1)
                            # Map compact template indices (0..N_present-1) to original residue indices
                            residue_indices_present = [idx for idx, present in enumerate(present_res_mask) if present]
                            compact_index_by_residue = {res_idx: compact_idx for compact_idx, res_idx in enumerate(residue_indices_present)}
                            # Query indices are positions in the template sequence aligned to the query
                            query_to_template_map = {res_idx: compact_index_by_residue[res_idx] for res_idx in residue_indices_present}
                        except Exception:
                            # Fallback to identity if masks missing, though this may error later
                            query_to_template_map = {j: j for j in range(len(template_sequence))}
                        templates.append(
                            folding_input.Template(
                                mmcif=new_mmcif_string,
                                query_to_template_map=query_to_template_map,
                            )
                        )
                    except Exception as e:
                        logging.error(f"Error processing template {i} for chain {chain_id}: {e}")
                        try:
                            os.makedirs("templates_debug", exist_ok=True)
                            error_filename = os.path.join("templates_debug", f"ERROR_template_{i}_chain_{chain_id}.txt")
                            with open(error_filename, "w") as error_file:
                                error_file.write(f"Error: {e}\n")
                                error_file.write(f"Template index: {i}\n")
                                error_file.write(f"Chain ID: {chain_id}\n")
                                if 'template_domain_names' in feature_dict:
                                    error_file.write(f"Domain name: {feature_dict['template_domain_names'][i]}\n")
                                if 'template_sequence' in feature_dict:
                                    error_file.write(f"Template sequence: {feature_dict['template_sequence'][i]}\n")
                        except Exception as save_error:
                            logging.error(f"Failed to save error info: {save_error}")
                        raise
            chain = _construct_chain(
                folding_input.ProteinChain,
                id=chain_id,
                sequence=sequence,
                ptms=[],
                residue_ids=residue_ids,
                description=mono_obj.description,
                unpaired_msa=unpaired_msa,
                paired_msa=paired_msa,
                templates=templates,
            )
            return chain

        def _expand_monomeric_object(
            mono_obj: Union[MonomericObject, ChoppedObject],
        ) -> list[Union[MonomericObject, ChoppedObject]]:
            return [mono_obj]

        def _process_single_object(obj, chain_id_counter_ref, used_chain_ids: set):
            nonlocal all_chains
            if isinstance(obj, dict) and 'json_input' in obj:
                json_path = obj['json_input']
                json_regions = obj.get('regions')
                logging.info(f"Processing JSON file: {json_path}")
                try:
                    with open(json_path, 'r') as f:
                        json_str = f.read()
                    input_obj = folding_input.Input.from_json(json_str)
                    chains_to_add = _expand_json_input_chains(
                        chains=input_obj.chains,
                        json_path=json_path,
                        regions=(
                            json_regions if isinstance(json_regions, list) and json_regions else None
                        ),
                    )
                    # Track the chain IDs from the JSON file
                    logging.info(
                        "JSON file %s contributes chains with IDs: %s",
                        json_path,
                        [chain.id for chain in chains_to_add],
                    )
                    
                    # Check for duplicate chain IDs and modify them if necessary
                    modified_chains = []
                    for chain in chains_to_add:
                        original_id = chain.id
                        new_id = original_id
                        
                        # If this chain ID is already used, generate a new one
                        if original_id in used_chain_ids:
                            new_id = get_next_available_chain_id(used_chain_ids, chain_id_counter_ref)
                            logging.info(f"Chain ID '{original_id}' already used, changing to '{new_id}'")
                        
                        used_chain_ids.add(new_id)
                        logging.info(f"Added chain ID '{new_id}' from JSON file {json_path}")
                        
                        modified_chains.append(_clone_chain_with_id(chain, new_id))
                    
                    all_chains.extend(modified_chains)
                except Exception as e:
                    logging.error(f"Failed to parse JSON file {json_path}: {e}")
                    raise
            elif isinstance(obj, (MonomericObject, ChoppedObject)):
                for expanded_obj in _expand_monomeric_object(obj):
                    chain_id = get_next_available_chain_id(used_chain_ids, chain_id_counter_ref)
                    used_chain_ids.add(chain_id)
                    logging.info(
                        f"Added chain ID '{chain_id}' for AlphaPulldown object "
                        f"{expanded_obj.description}"
                    )
                    all_chains.append(_monomeric_to_chain(expanded_obj, chain_id))
            elif isinstance(obj, MultimericObject):
                chains = []
                translated_result = None
                combined_msa = obj.feature_dict.get('msa')
                expanded_interactors = list(obj.interactors)

                translated_result = (
                    translate_af2_individual_chain_features_to_af3_msas_with_stats(
                        chain_feature_dicts=[
                            interactor.feature_dict for interactor in expanded_interactors
                        ],
                        chain_sequences=[
                            interactor.sequence for interactor in expanded_interactors
                        ],
                    )
                )
                num_pairable_chains = sum(
                    chain_stats.paired_species_identifier_count > 0
                    for chain_stats in translated_result.chain_stats
                )
                if num_pairable_chains < 2 and combined_msa is not None:
                    # Fall back to the merged AF2 multimer MSA transport path when the
                    # individual `_all_seq` features do not carry usable species IDs.
                    translated_result = (
                        translate_af2_complex_msa_to_af3_unpaired_chain_msas_with_stats(
                            merged_msa=np.asarray(combined_msa),
                            chain_sequences=[
                                interactor.sequence for interactor in obj.interactors
                            ],
                            num_alignments=obj.feature_dict.get('num_alignments'),
                            deletion_matrix=(
                                obj.feature_dict.get('deletion_matrix_int')
                                if obj.feature_dict.get('deletion_matrix_int') is not None
                                else obj.feature_dict.get('deletion_matrix')
                            ),
                            asym_id=obj.feature_dict.get('asym_id'),
                        )
                    )
                    expanded_interactors = list(obj.interactors)

                for chain_index, interactor in enumerate(expanded_interactors):
                    chain_id = get_next_available_chain_id(used_chain_ids, chain_id_counter_ref)
                    used_chain_ids.add(chain_id)
                    logging.info(f"Added chain ID '{chain_id}' for multimeric interactor")
                    base_chain = _monomeric_to_chain(interactor, chain_id)
                    if translated_result is not None:
                        chain_msas = translated_result.chain_msas[chain_index]
                        chain_stats = translated_result.chain_stats[chain_index]
                        base_chain = _construct_chain(
                            folding_input.ProteinChain,
                            id=base_chain.id,
                            sequence=base_chain.sequence,
                            ptms=base_chain.ptms,
                            residue_ids=getattr(base_chain, "residue_ids", None),
                            description=interactor.description,
                            unpaired_msa=chain_msas.unpaired_msa,
                            paired_msa=chain_msas.paired_msa,
                            templates=base_chain.templates,
                        )
                        af2_translated_msa_chain_ids.add(base_chain.id)
                        if debug_msas:
                            translation_debug_chain_records.append(
                                _TranslatedMsaDebugRecord(
                                    chain_id=base_chain.id,
                                    chain_description=interactor.description,
                                    chain_length=len(interactor.sequence),
                                    paired_msa=chain_msas.paired_msa,
                                    unpaired_msa=chain_msas.unpaired_msa,
                                    paired_msa_row_count=chain_stats.paired_msa_row_count,
                                    unpaired_msa_row_count=chain_stats.unpaired_msa_row_count,
                                    paired_species_identifier_count=(
                                        chain_stats.paired_species_identifier_count
                                    ),
                                    paired_rows_without_species_identifier_count=(
                                        chain_stats.paired_rows_without_species_identifier_count
                                    ),
                                    paired_rows_with_generated_accession_count=(
                                        chain_stats.paired_rows_with_generated_accession_count
                                    ),
                                )
                            )
                    chains.append(base_chain)
                if debug_msas and translated_result is not None:
                    translation_debug_results.append(translated_result)
                all_chains.extend(chains)
            else:
                raise TypeError(f"Unsupported object type for folding input conversion: {type(obj)}")

        prepared_inputs = []
        chain_id_counter = [0]  # Use a list to allow pass-by-reference
        used_chain_ids = set()  # Track used chain IDs
        all_chains = []
        # Track chains whose MSAs were translated from AF2 features; they must
        # not be rewritten by the promotion heuristic below.
        af2_translated_msa_chain_ids: set[str] = set()
        translation_debug_chain_records: list[_TranslatedMsaDebugRecord] = []
        translation_debug_results: list[Af2ToAf3TranslationResult] = []

        for entry in objects_to_model:
            object_to_model = entry['object']
            output_dir = entry['output_dir']

            # If object_to_model is a list, process each element
            if isinstance(object_to_model, list):
                for sub_obj in object_to_model:
                    _process_single_object(sub_obj, chain_id_counter, used_chain_ids)
                continue  # Done with this entry

            # Otherwise, process as a single object
            _process_single_object(object_to_model, chain_id_counter, used_chain_ids)

        # Debug: Print all chain IDs before creating the combined input
        logging.info(f"All chain IDs before creating combined input: {[chain.id for chain in all_chains]}")
        logging.info(f"Used chain IDs set: {used_chain_ids}")

        # Create a single combined input with all chains
        if all_chains:
            # Promote unpaired->paired only for chains not constructed from AF2 multimer features.
            promoted_chains: list[folding_input.ProteinChain | folding_input.RnaChain | folding_input.DnaChain | folding_input.Ligand] = []
            for ch in all_chains:
                if (
                    isinstance(ch, folding_input.ProteinChain)
                    and ch.id not in af2_translated_msa_chain_ids
                ):
                    try:
                        has_empty_paired = (getattr(ch, 'paired_msa', None) in (None, ''))
                        has_unpaired = bool(getattr(ch, 'unpaired_msa', ''))
                        if has_empty_paired and has_unpaired:
                            new_chain = _construct_chain(
                                folding_input.ProteinChain,
                                id=ch.id,
                                sequence=ch.sequence,
                                ptms=ch.ptms,
                                residue_ids=getattr(ch, "residue_ids", None),
                                description=getattr(ch, "description", None),
                                paired_msa=ch.unpaired_msa,
                                unpaired_msa='',
                                templates=ch.templates,
                            )
                            promoted_chains.append(new_chain)
                            continue
                    except Exception:
                        pass
                promoted_chains.append(ch)

            all_chains = promoted_chains
            job_name = _build_output_job_name(objects_to_model)
            combined_input = folding_input.Input(
                name=job_name,
                rng_seeds=[random_seed],
                chains=all_chains,
            )
            # Use the output directory from the first object
            first_output_dir = objects_to_model[0]['output_dir']
            if debug_msas:
                _write_translated_msa_debug_artifacts(
                    job_name=combined_input.sanitised_name(),
                    output_dir=first_output_dir,
                    chain_records=translation_debug_chain_records,
                    translation_results=translation_debug_results,
                )
            # Disable overlap resolution when we provide translated AF2 multimer MSAs.
            disable_overlaps = len(af2_translated_msa_chain_ids) > 0
            prepared_inputs.append({combined_input: (first_output_dir, not disable_overlaps)})

        return prepared_inputs

    @staticmethod
    def predict(
        model_runner: ModelRunner,
        objects_to_model: List[Dict[str, Union[MultimericObject, MonomericObject, ChoppedObject, 'folding_input.Input', str]]],
        random_seed: int,
        buckets: int,
        **kwargs,
    ):
        """Predicts structures for a list of objects using AlphaFold 3.
        Supports merging AlphaPulldown protein objects and input.json objects into a single job.
        """
        logging.info(f"predict called with {len(objects_to_model)} objects to model")
        for i, obj in enumerate(objects_to_model):
            logging.info(f"Object {i}: {obj}")
        
        if isinstance(buckets, int):
            buckets = [buckets]
        buckets = tuple(int(b) for b in buckets)

        # Prepare inputs
        logging.info("Calling prepare_input...")
        prepared_inputs = AlphaFold3Backend.prepare_input(
            objects_to_model=objects_to_model,
            random_seed=random_seed,
            af3_input_json=kwargs.get("af3_input_json"),
            features_directory=kwargs.get("features_directory"),
            debug_templates=kwargs.get("debug_templates", False),
            debug_msas=kwargs.get("debug_msas", False),
        )
        # Run predictions
        for mapping in prepared_inputs:
            logging.info(f"Processing mapping: {mapping}")
            if len(mapping) != 1:
                raise ValueError(f"Expected exactly one item in mapping, got {len(mapping)}: {mapping}")
            fold_input_obj, mapping_value = next(iter(mapping.items()))
            if isinstance(mapping_value, tuple):
                output_dir, resolve_overlaps_flag = mapping_value
            else:
                output_dir = mapping_value
                resolve_overlaps_flag = True
            
            # Expand to multiple seeds if num_seeds is specified
            num_seeds = kwargs.get('num_seeds')
            if num_seeds is not None:
                logging.info(f'Expanding fold job {fold_input_obj.name} to {num_seeds} seeds')
                fold_input_obj = fold_input_obj.with_multiple_seeds(num_seeds)
            
            # Do not write input JSON here; defer to process_fold_input for a single source of truth.

            logging.info(f'Processing fold input {fold_input_obj.name}')

            prediction_result = process_fold_input(
                fold_input=fold_input_obj,
                model_runner=model_runner,
                output_dir=output_dir,
                buckets=buckets,
                resolve_msa_overlaps=resolve_overlaps_flag,
                debug_msas=kwargs.get('debug_msas', False),
            )

            yield {
                'object': fold_input_obj,
                'output_dir': output_dir,
                'prediction_results': prediction_result
            }

    @staticmethod
    def postprocess(**kwargs) -> None:
        prediction_results = kwargs.get('prediction_results')
        output_dir = kwargs.get('output_dir')
        fold_input_obj = kwargs.get('multimeric_object')

        if prediction_results is None or output_dir is None or fold_input_obj is None:
            logging.warning('AF3 postprocess called with missing arguments; skipping.')
            return

        try:
            job_name = fold_input_obj.sanitised_name()
        except Exception:
            job_name = 'ranked_0'

        logging.info(
            f"Writing outputs for {job_name} with {len(prediction_results) if hasattr(prediction_results, '__len__') else 'unknown'} result(s)..."
        )
        write_outputs(
            all_inference_results=prediction_results,
            output_dir=output_dir,
            job_name=job_name,
        )
