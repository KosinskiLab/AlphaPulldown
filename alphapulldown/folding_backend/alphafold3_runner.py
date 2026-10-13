"""
Runs the AlphaFold 3 model on a fold input: featurisation, inference and structure
extraction per seed, adapted from upstream run_alphafold.py and kept close to it for
diffing. Each seed's outputs are written as it finishes.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""

import dataclasses
import functools
import importlib
import json
import logging
import os
import pathlib
import time
import typing
from collections.abc import Sequence
from typing import List

import haiku as hk
import jax
import numpy as np
from alphafold3.common import base_config
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.model import features, params
from alphafold3.model import model
from alphafold3.model.components import utils
from jax import numpy as jnp

from alphapulldown.folding_backend.alphafold3_debug import _write_final_msa_a3m
from alphapulldown.folding_backend.alphafold3_outputs import write_seed_outputs
from alphapulldown.prediction import af3_fused_triangles


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
    fused_triangles: af3_fused_triangles.FusedTriangleChoice = af3_fused_triangles.OFF

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
        result = jax.tree_util.tree_map(np.asarray, result)
        result = jax.tree_util.tree_map(
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


def _record_inference_kernels(
    output_dir: os.PathLike[str] | str,
    seed: int,
    model_runner: ModelRunner,
    example: features.BatchDict,
) -> None:
    """Record in inference_kernels.json which kernels a seed's triangle layers ran.

    Fused and original layers differ slightly in their numerics, so the record lets
    ranking and analysis tell them apart, like AlphaFold 2's. Entries are keyed
    ``seed-N`` like AF3's other outputs and are read back from the runner's own config,
    so they state what actually ran. The record never fails a finished prediction: an
    unreadable file is replaced, and any other failure is logged and skipped.
    """
    fused_triangles = getattr(model_runner, "fused_triangles", None)
    tokens = example.get("token_index", example.get("aatype"))
    if fused_triangles is None or tokens is None:  # not an AF3 runner (e.g. a test double)
        return
    key = f"seed-{seed}"
    try:
        record = af3_fused_triangles.metadata(
            model_runner.config.global_config, fused_triangles, num_tokens=int(tokens.shape[0])
        )
    except Exception as exc:  # provenance only; the prediction itself succeeded
        logging.warning("Not recording the inference kernels of %s: %s", key, exc)
        return
    path = pathlib.Path(output_dir) / "inference_kernels.json"
    try:
        records = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(records, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        logging.warning("Replacing unreadable %s: %s", path, exc)
        records = {}
    records[key] = record
    try:
        os.makedirs(output_dir, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    except OSError as exc:
        logging.warning("Not recording the inference kernels of %s: %s", key, exc)


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
    output_dir: os.PathLike[str] | str | None = None,
    resolve_msa_overlaps: bool = True,
    debug_msas: bool = False,
    on_seed_done: typing.Callable[[ResultsForSeed], None] | None = None,
) -> Sequence[ResultsForSeed]:
    """Run inference (featurisation + model) to predict structures for each seed.

    With ``on_seed_done``, each seed's results go to it as soon as they exist instead
    of being collected, so only one seed's are held in memory.
    """
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

    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        logging.info(f'Running model inference for seed {seed}...')
        inference_start_time = time.time()
        # If requested, dump the post-featurisation merged complex MSA for inspection.
        if debug_msas:
            _write_final_msa_a3m(
                example, seed, output_dir=output_dir, job_name=fold_input.sanitised_name()
            )

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
        if output_dir is not None:
            _record_inference_kernels(output_dir, seed, model_runner, example)

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

        results_for_seed = ResultsForSeed(
            seed=seed,
            inference_results=inference_results,
            full_fold_input=fold_input,
            embeddings=embeddings_out,
            distogram=dist_out,
        )
        if on_seed_done is None:
            all_inference_results.append(results_for_seed)
        else:
            on_seed_done(results_for_seed)
    logging.info(
        f'Inference + extraction for seeds {fold_input.rng_seeds} took {time.time() - all_inference_start_time:.2f} seconds.'
    )
    return all_inference_results


def _initialise_tokamax_trace_context() -> None:
    """Create tokamax's JAX user context now, before the model is first traced.

    tokamax creates it (``jax.make_user_context``) the first time an op consults its
    autotuning cache, which is during the first trace. A new user context adds an entry
    to JAX's trace context and so changes every jit cache key: the second prediction in a
    process missed the cache and re-traced and recompiled the whole model (~50 s). The
    hook is private to tokamax, so its absence is not an error.
    """
    try:
        tokamax_op = importlib.import_module("tokamax._src.ops.op")
    except ImportError:
        return
    initialise = getattr(tokamax_op, "get_autotuning_cache_overlay_state", None)
    if callable(initialise):
        initialise()


def process_fold_input(
    fold_input: folding_input.Input,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    resolve_msa_overlaps: bool = True,
    debug_msas: bool = False,
    seeds_done: Sequence[int] = (),
    modelcif_metadata: Sequence[dict] | None = None,
) -> folding_input.Input | list[int]:
    """Runs data pipeline and/or inference on a single fold input, writing each seed's
    outputs as it finishes; returns the fold's seeds, all now on disk. ``seeds_done``,
    written out by an interrupted run, are not predicted again.
    """
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

    missing_seeds = [seed for seed in fold_input.rng_seeds if seed not in seeds_done]
    if seeds_done:
        logging.info(
            f'Resuming {fold_input.name}: seed(s) {list(seeds_done)} are already predicted.'
        )
    if missing_seeds:
        logging.info(
            f'Predicting 3D structure for {fold_input.name} with seed(s) {missing_seeds}...'
        )
        job_name = fold_input.sanitised_name()
        # Featurisation is per seed, so the missing ones come out as in a single run.
        predict_structure(
            fold_input=dataclasses.replace(fold_input, rng_seeds=missing_seeds),
            model_runner=model_runner,
            buckets=buckets,
            output_dir=output_dir,
            resolve_msa_overlaps=resolve_msa_overlaps,
            debug_msas=debug_msas,
            on_seed_done=lambda results_for_seed: write_seed_outputs(
                results_for_seed, output_dir, job_name, modelcif_metadata
            ),
        )
    logging.info(f'Done processing fold input {fold_input.name}.')
    return list(fold_input.rng_seeds)
