"""
Implements structure prediction backend using AlphaFold 3.

Copyright (c) 2026 European Molecular Biology Laboratory

Author: Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""

import dataclasses
import logging
import os
import pathlib
import typing
from collections.abc import Sequence
from typing import List, Dict, Union, overload

import alphafold3.cpp
import jax
import tokamax
from alphafold3.common import folding_input
from alphafold3.model import model

from alphapulldown.folding_backend.folding_backend import FoldingBackend
from alphapulldown.folding_backend.alphafold3_debug import _write_translated_msa_debug_artifacts
from alphapulldown.folding_backend.alphafold3_inputs import (
    FoldInputBuilder,
    PreparedFold,
    _build_output_job_name,
)
from alphapulldown.folding_backend.alphafold3_outputs import (
    _modelcif_metadata,
    _start_or_resume_fold,
    write_fold_outputs,
)
# predict_structure and ResultsForSeed are re-exported for callers that import them from here.
from alphapulldown.folding_backend.alphafold3_runner import (
    ModelRunner,
    ResultsForSeed,
    _initialise_tokamax_trace_context,
    make_model_config,
    predict_structure,
    process_fold_input,
)
from alphapulldown.prediction import af3_fused_triangles
from alphapulldown.prediction.inference_flags import validate_model_configuration
from alphapulldown.prediction.jax_compilation_cache import enable_persistent_compilation_cache
from alphapulldown.prediction.resume import finish_fold_record
from alphapulldown.utils.output_paths import debug_dir
from alphapulldown.objects import MultimericObject, MonomericObject, ChoppedObject
from alphapulldown.folding_backend.alphafold3_storage import post_prediction_process_af3


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
        fast_kernels: str = "off",
        **other_backends_options,
    ) -> Dict:
        """Sets up the ModelRunner with the given configurations.

        The adapters give every backend the whole configuration; the options of other
        backends arrive in ``other_backends_options``, and an option no backend takes is an
        error rather than dropped.
        """
        validate_model_configuration(other_backends_options)
        enable_persistent_compilation_cache(jax_compilation_cache_dir)
        _initialise_tokamax_trace_context()

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
        fused_triangles = af3_fused_triangles.resolve(fast_kernels, gpu_devices[0])
        logging.info('Building model from scratch...')

        model_class = model.Model
        model_runner = ModelRunner(
            model_class=model_class,
            config=make_model_config(
                model_class=model_class,
                flash_attention_implementation=typing.cast(
                    tokamax.DotProductAttentionImplementation,
                    flash_attention_implementation,
                ),
                num_diffusion_samples=num_diffusion_samples,
                num_recycles=num_recycles,
                return_embeddings=return_embeddings,
                return_distogram=return_distogram,
                global_config_update=fused_triangles.global_config_update(),
            ),
            device=gpu_devices[0],
            model_dir=pathlib.Path(model_dir),
            fused_triangles=fused_triangles,
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
    ) -> list[PreparedFold]:
        """Prepare input for AlphaFold3 prediction."""
        logging.info(f"prepare_input called with {len(objects_to_model)} objects to model")
        for i, entry in enumerate(objects_to_model):
            logging.info(f"Object {i}: {entry}")

        # A fresh builder per call: its chain IDs and caches belong to this preparation.
        builder = FoldInputBuilder(
            features_directory=features_directory,
            debug_templates=debug_templates,
            debug_msas=debug_msas,
            debug_dir=debug_dir(objects_to_model[0]['output_dir']) if objects_to_model else None,
        )
        for entry in objects_to_model:
            object_to_model = entry['object']
            # If object_to_model is a list, process each element
            if isinstance(object_to_model, list):
                for sub_obj in object_to_model:
                    builder.add(sub_obj)
                continue  # Done with this entry

            # Otherwise, process as a single object
            builder.add(object_to_model)

        # Debug: Print all chain IDs before creating the combined input
        logging.info(f"All chain IDs before creating combined input: {[chain.id for chain in builder.chains]}")
        logging.info(f"Used chain IDs set: {builder.used_chain_ids}")

        # Create a single combined input with all chains
        if not builder.chains:
            return []
        combined_input = builder.build(
            job_name=_build_output_job_name(objects_to_model), random_seed=random_seed
        )
        # Use the output directory from the first object
        first_output_dir = objects_to_model[0]['output_dir']
        if debug_msas:
            _write_translated_msa_debug_artifacts(
                job_name=combined_input.sanitised_name(),
                output_dir=first_output_dir,
                chain_records=builder.translation_debug_chain_records,
                translation_results=builder.translation_debug_results,
            )
        # Disable overlap resolution when we provide translated AF2 multimer MSAs.
        disable_overlaps = len(builder.af2_translated_msa_chain_ids) > 0
        return [
            PreparedFold(
                fold_input=combined_input,
                output_dir=first_output_dir,
                resolve_msa_overlaps=not disable_overlaps,
            )
        ]

    @staticmethod
    def predict(
        model_runner: ModelRunner,
        objects_to_model: List[Dict[str, Union[MultimericObject, MonomericObject, ChoppedObject, 'folding_input.Input', str]]],
        random_seed: int,
        buckets: int | Sequence[int],
        *,
        allow_resume: bool = False,
        postprocess_flags: Dict | None = None,
        features_directory=None,
        debug_templates: bool = False,
        debug_msas: bool = False,
        num_seeds: int | None = None,
        num_diffusion_samples: int | None = None,
        num_recycles: int | None = None,
        return_embeddings: bool = False,
        return_distogram: bool = False,
        **other_backends_options,
    ):
        """Predicts structures for a list of objects using AlphaFold 3.
        Supports merging AlphaPulldown protein objects and input.json objects into a single job.

        ``random_seed=None`` reuses an earlier run's seed when resuming. With
        ``allow_resume``, a fold already complete is neither predicted nor yielded.
        ``num_diffusion_samples``, ``num_recycles`` and the embedding and distogram
        switches, set up in ``setup``, are recorded with each fold for resume. Options of
        other backends arrive in ``other_backends_options``; one no backend takes is an
        error rather than dropped.
        """
        validate_model_configuration(other_backends_options)
        logging.info(f"predict called with {len(objects_to_model)} objects to model")
        for i, obj in enumerate(objects_to_model):
            logging.info(f"Object {i}: {obj}")

        if isinstance(buckets, int):
            buckets = [buckets]
        buckets = tuple(int(b) for b in buckets)
        postprocess_flags = postprocess_flags or {}
        model_flags = {
            'num_seeds': num_seeds,
            'num_diffusion_samples': num_diffusion_samples,
            'num_recycles': num_recycles,
            'return_embeddings': return_embeddings,
            'return_distogram': return_distogram,
        }

        # Prepare inputs
        logging.info("Calling prepare_input...")
        prepared_inputs = AlphaFold3Backend.prepare_input(
            objects_to_model=objects_to_model,
            # A placeholder: _start_or_resume_fold settles each fold's seeds.
            random_seed=0 if random_seed is None else random_seed,
            features_directory=features_directory,
            debug_templates=debug_templates,
            debug_msas=debug_msas,
        )
        # Run predictions
        for prepared in prepared_inputs:
            output_dir = prepared.output_dir
            started = _start_or_resume_fold(
                prepared.fold_input, output_dir, random_seed, allow_resume, model_flags, postprocess_flags
            )
            if started is None:
                continue
            fold_input_obj, seeds_done = started

            # Do not write input JSON here; defer to process_fold_input for a single source of truth.

            logging.info(f'Processing fold input {fold_input_obj.name}')

            prediction_result = process_fold_input(
                fold_input=fold_input_obj,
                model_runner=model_runner,
                output_dir=output_dir,
                buckets=buckets,
                resolve_msa_overlaps=prepared.resolve_msa_overlaps,
                debug_msas=debug_msas,
                seeds_done=seeds_done,
                modelcif_metadata=(
                    _modelcif_metadata(fold_input_obj, output_dir)
                    if postprocess_flags.get('convert_to_modelcif', True)
                    else None
                ),
            )

            yield {
                'object': fold_input_obj,
                'output_dir': output_dir,
                'prediction_results': prediction_result
            }

    @staticmethod
    def postprocess(
        *,
        prediction_results=None,
        output_dir=None,
        multimeric_object=None,
        storage_mode: str = 'vanilla',
        **_ignored,
    ) -> None:
        """Write the best model under the job name, apply the storage preset and mark the
        fold complete. ``prediction_results`` are the fold's seeds, already on disk.

        The adapters pass every backend's post-processing options; those AlphaFold 3
        does not use, such as ``models_to_relax``, are ignored.
        """
        fold_input_obj = multimeric_object
        if prediction_results is None or output_dir is None or fold_input_obj is None:
            logging.warning('AF3 postprocess called with missing arguments; skipping.')
            return

        try:
            job_name = fold_input_obj.sanitised_name()
        except Exception:
            job_name = 'ranked_0'

        # The samples, ModelCIF metadata included, were written as each seed finished
        # (write_seed_outputs); the best one now goes under the job name.
        logging.info(f"Writing the best model of {job_name}...")
        write_fold_outputs(output_dir=output_dir, job_name=job_name)

        # Apply the AF3 storage preset (no-op for the default 'vanilla').
        post_prediction_process_af3(output_dir, job_name, storage_mode=storage_mode)
        # Last: a fold recorded complete is skipped by --allow_resume.
        finish_fold_record(output_dir)
