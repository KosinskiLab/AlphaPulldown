"""
Implements structure prediction backend using AlphaFold 3.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""

import csv
import dataclasses
import datetime
import functools
import logging
import os
import pathlib
import time
import typing
from collections.abc import Sequence
from typing import List, Dict, Union
import json
import itertools
import string

import alphafold3.cpp
import haiku as hk
import jax
import numpy as np
from alphafold3.common import base_config
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.jax.attention import attention
from alphafold3.model import features, params, post_processing
from alphafold3.model import model
from alphafold3.model.components import utils
from jax import numpy as jnp

from alphafold.common import residue_constants
from alphafold.common.protein import Protein, to_mmcif
from alphapulldown.folding_backend.folding_backend import FoldingBackend
from alphapulldown.objects import MultimericObject, MonomericObject, ChoppedObject



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
    """Writes outputs to the specified output directory."""
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
            post_processing.write_output(
                inference_result=result, output_dir=sample_dir
            )
            ranking_score = float(result.metadata['ranking_score'])
            ranking_scores.append((seed, sample_idx, ranking_score))
            if max_ranking_score is None or ranking_score > max_ranking_score:
                max_ranking_score = ranking_score
                max_ranking_result = result

    if max_ranking_result is not None:  # True iff ranking_scores non-empty.
        post_processing.write_output(
            inference_result=max_ranking_result,
            output_dir=output_dir,
            terms_of_use=output_terms,
            name=job_name,
        )
        with open(os.path.join(output_dir, 'ranking_scores.csv'), 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(['seed', 'sample', 'ranking_score'])
            writer.writerows(ranking_scores)


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
    """Run inference (featurisation + model) to predict structures for each seed."""
    logging.info(f'Featurising data for seeds {fold_input.rng_seeds}...')
    featurisation_start_time = time.time()
    ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input, buckets=buckets, ccd=ccd, verbose=True
    )
    logging.info(
        f'Featurising took {time.time() - featurisation_start_time:.2f} seconds.'
    )

    all_inference_start_time = time.time()
    all_inference_results = []
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        logging.info(f'Running model inference for seed {seed}...')
        inference_start_time = time.time()
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

        all_inference_results.append(
            ResultsForSeed(
                seed=seed,
                inference_results=inference_results,
                full_fold_input=fold_input,
            )
        )
    logging.info(
        f'Inference + extraction for seeds {fold_input.rng_seeds} took {time.time() - all_inference_start_time:.2f} seconds.'
    )
    return all_inference_results


def process_fold_input(
    fold_input: folding_input.Input,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
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

    # Write input JSON
    logging.info(f'Writing model input JSON to {output_dir}')
    with open(
        os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json'), 'wt'
    ) as f:
        f.write(fold_input.to_json())

    if model_runner is None:
        logging.info('Skipping inference...')
        return fold_input

    # Run inference
    logging.info(
        f'Predicting 3D structure for {fold_input.name} with seed(s) {fold_input.rng_seeds}...'
    )
    all_inference_results = predict_structure(
        fold_input=fold_input,
        model_runner=model_runner,
        buckets=buckets,
    )

    # Write outputs
    logging.info(
        f'Writing outputs for {fold_input.name} for seed(s) {fold_input.rng_seeds}...'
    )
    write_outputs(
        all_inference_results=all_inference_results,
        output_dir=output_dir,
        job_name=fold_input.sanitised_name(),
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
        ):
            # The new code approach:
            config = model_class.Config()
            if hasattr(config, 'global_config'):
                config.global_config.flash_attention_implementation = flash_attention_implementation
            if hasattr(config, 'heads') and hasattr(config.heads, 'diffusion'):
                config.heads.diffusion.eval.num_samples = num_diffusion_samples
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
    ) -> list:
        """Prepare input for AlphaFold3 prediction."""
        def get_chain_id(index: int) -> str:
            if index < 26:
                return chr(ord('A') + index)
            else:
                first_char = chr(ord('A') + (index // 26) - 1)
                second_char = chr(ord('A') + (index % 26))
                return first_char + second_char

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

        def msa_array_to_a3m(msa_array):
            msa_sequences = []
            for i, msa_seq in enumerate(msa_array):
                seq_str = ''.join([residue_constants.ID_TO_HHBLITS_AA.get(int(aa), 'X') for aa in msa_seq])
                msa_sequences.append(f'>sequence_{i}\n{seq_str}')
            return '\n'.join(msa_sequences)

        def _monomeric_to_chain(
            mono_obj: Union[MonomericObject, ChoppedObject],
            chain_id: str
        ) -> folding_input.ProteinChain:
            sequence = mono_obj.sequence
            feature_dict = mono_obj.feature_dict
            msa_array = feature_dict.get('msa')
            unpaired_msa = msa_array_to_a3m(msa_array) if msa_array is not None else ""
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
                        chain_index_array = np.zeros_like(feature_dict["residue_index"], dtype=int)
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
            chain = folding_input.ProteinChain(
                id=chain_id,
                sequence=sequence,
                ptms=[],
                unpaired_msa=unpaired_msa,
                paired_msa=paired_msa,
                templates=templates,
            )
            return chain

        def _process_single_object(obj, chain_id_counter_ref):
            nonlocal all_chains, job_name
            if isinstance(obj, dict) and 'json_input' in obj:
                json_path = obj['json_input']
                try:
                    with open(json_path, 'r') as f:
                        json_str = f.read()
                    input_obj = folding_input.Input.from_json(json_str)
                    all_chains.extend(input_obj.chains)
                    if len(all_chains) == len(input_obj.chains):
                        job_name = input_obj.name
                    else:
                        job_name = f"{job_name}_and_{input_obj.name}"
                except Exception as e:
                    logging.error(f"Failed to parse JSON file {json_path}: {e}")
                    raise
            elif isinstance(obj, (MonomericObject, ChoppedObject)):
                chain_id = get_chain_id(chain_id_counter_ref[0])
                chain_id_counter_ref[0] += 1
                chains = [_monomeric_to_chain(obj, chain_id)]
                all_chains.extend(chains)
            elif isinstance(obj, MultimericObject):
                chains = []
                for interactor in obj.interactors:
                    chain_id = get_chain_id(chain_id_counter_ref[0])
                    chain_id_counter_ref[0] += 1
                    chain = _monomeric_to_chain(interactor, chain_id)
                    chains.append(chain)
                all_chains.extend(chains)
            else:
                raise TypeError(f"Unsupported object type for folding input conversion: {type(obj)}")

        prepared_inputs = []
        chain_id_counter = [0]  # Use a list to allow pass-by-reference
        all_chains = []
        job_name = "combined_prediction"

        for entry in objects_to_model:
            object_to_model = entry['object']
            output_dir = entry['output_dir']

            # If object_to_model is a list, process each element
            if isinstance(object_to_model, list):
                for sub_obj in object_to_model:
                    _process_single_object(sub_obj, chain_id_counter)
                continue  # Done with this entry

            # Otherwise, process as a single object
            _process_single_object(object_to_model, chain_id_counter)

        # Create a single combined input with all chains
        if all_chains:
            combined_input = folding_input.Input(
                name=job_name,
                rng_seeds=[random_seed],
                chains=all_chains,
            )
            # Use the output directory from the first object
            first_output_dir = objects_to_model[0]['output_dir']
            prepared_inputs.append({combined_input: first_output_dir})

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
        """Predicts structures for a list of objects using AlphaFold 3.
        Supports merging AlphaPulldown protein objects and input.json objects into a single job.
        """
        if isinstance(buckets, int):
            buckets = [buckets]
        buckets = tuple(int(b) for b in buckets)

        # Prepare inputs
        prepared_inputs = AlphaFold3Backend.prepare_input(
            objects_to_model=objects_to_model,
            random_seed=random_seed,
            af3_input_json=kwargs.get("af3_input_json"),
            features_directory=kwargs.get("features_directory"),
        )

        # Run predictions
        for mapping in prepared_inputs:
            (fold_input_obj, output_dir), = mapping.items()
        # Prepare inputs
        prepared_inputs = AlphaFold3Backend.prepare_input(
            objects_to_model=objects_to_model,
            random_seed=random_seed,
            af3_input_json=kwargs.get("af3_input_json"),
            features_directory=kwargs.get("features_directory"),
        )

        # Run predictions
        for mapping in prepared_inputs:
            (fold_input_obj, output_dir), = mapping.items()
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                logging.error(f'Failed to create output directory {output_dir}: {e}')
                raise

            logging.info(f'Processing fold input {fold_input_obj.name}')

            prediction_result = process_fold_input(
                fold_input=fold_input_obj,
                model_runner=model_runner,
                output_dir=output_dir,
                buckets=buckets,
            )

            yield {
                'object': fold_input_obj,
                'output_dir': output_dir,
                'prediction_results': prediction_result
            }

    @staticmethod
    def postprocess(**kwargs) -> None:
        pass

