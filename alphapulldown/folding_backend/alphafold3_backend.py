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
# Global Constants and Type Definitions
# -----------------------------------------------------------------------------

HHBLITS_TO_INTERNAL_AATYPE = np.array([
    0,   # A -> ALA
    1,   # C -> CYS
    2,   # D -> ASP
    3,   # E -> GLU
    4,   # F -> PHE
    5,   # G -> GLY
    6,   # H -> HIS
    7,   # I -> ILE
    8,   # K -> LYS
    9,   # L -> LEU
    10,  # M -> MET
    11,  # N -> ASN
    12,  # P -> PRO
    13,  # Q -> GLN
    14,  # R -> ARG
    15,  # S -> SER
    16,  # T -> THR
    17,  # V -> VAL
    18,  # W -> TRP
    19,  # Y -> TYR
    20,  # X -> UNK
    20,  # - -> UNK
], dtype=np.int32)


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


# -----------------------------------------------------------------------------
# Core Logic Functions (Conversion, Prediction, Processing, Output)
# -----------------------------------------------------------------------------

def _convert_to_fold_input(
    object_to_model: Union[MonomericObject, ChoppedObject, MultimericObject],
    random_seed: int,
) -> folding_input.Input:
    """Convert a given object to AlphaFold3 fold input."""
    # (Unchanged from your original code, except for any references to base_model)
    # ...
    chain_id_gen = (chr(c) for c in range(ord('A'), ord('Z') + 1))

    def insert_release_date_into_mmcif(
        mmcif_string: str, revision_date: str = '2100-01-01'
    ) -> str:
        pdb_data = mmcif_string.splitlines()
        header_lines = [
            "_entry.id   pdb",
            "_pdbx_audit_revision_history.revision_date"
        ]
        release_date_line = f"{revision_date}"
        pdb_data.insert(2, "\n".join(header_lines))
        pdb_data.insert(3, release_date_line)
        return "\n".join(pdb_data)

    def msa_array_to_a3m(msa_array):
        """Converts MSA numpy array to A3M formatted string."""
        msa_sequences = []
        for i, msa_seq in enumerate(msa_array):
            seq_str = ''.join([residue_constants.ID_TO_HHBLITS_AA.get(int(aa), 'X') for aa in msa_seq])
            msa_sequences.append(f'>sequence_{i}\n{seq_str}')
        return '\n'.join(msa_sequences)

    def _monomeric_to_chain(
        mono_obj: Union[MonomericObject, ChoppedObject],
        chain_id: str
    ) -> folding_input.ProteinChain:
        """Converts a single MonomericObject or ChoppedObject into a ProteinChain."""
        sequence = mono_obj.sequence
        feature_dict = mono_obj.feature_dict

        # Convert MSA arrays to A3M.
        msa_array = feature_dict.get('msa')
        unpaired_msa = msa_array_to_a3m(msa_array) if msa_array is not None else ""
        paired_msa = ""  # For this simplified logic, no paired MSA is handled here.

        # Process templates if present
        templates = []
        if 'template_aatype' in feature_dict:
            num_templates = feature_dict['template_aatype'].shape[0]
            for i in range(num_templates):
                pdb_code_chain = feature_dict["template_domain_names"][i].decode('utf-8')
                if '_' in pdb_code_chain:
                    pdb_code, chain_id_template = pdb_code_chain.split('_')
                else:
                    pdb_code = pdb_code_chain
                    chain_id_template = 'A'

                template_aatype_onehot = feature_dict["template_aatype"][i]
                template_aatype_hhblits_id = np.argmax(template_aatype_onehot, axis=-1)
                template_aatype_int = HHBLITS_TO_INTERNAL_AATYPE[template_aatype_hhblits_id]

                unique_aatypes = np.unique(template_aatype_int)
                if not np.all((unique_aatypes >= 0) & (unique_aatypes <= 20)):
                    raise ValueError(f"Unexpected aatype indices found: {unique_aatypes}")

                template_mask = feature_dict["template_all_atom_masks"][i]
                chain_index_array = np.zeros_like(feature_dict["residue_index"], dtype=int)
                template_sequence = feature_dict["template_sequence"][i]
                if isinstance(template_sequence, bytes):
                    template_sequence = template_sequence.decode('utf-8')

                protein = Protein(
                    atom_positions=feature_dict["template_all_atom_positions"][i],
                    atom_mask=template_mask,
                    aatype=template_aatype_int,
                    residue_index=feature_dict["residue_index"],
                    chain_index=chain_index_array,
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

        chain = folding_input.ProteinChain(
            id=chain_id,
            sequence=sequence,
            ptms=[],
            unpaired_msa=unpaired_msa,
            paired_msa=paired_msa,
            templates=templates,
        )
        return chain

    if isinstance(object_to_model, (MonomericObject, ChoppedObject)):
        chain_id = next(chain_id_gen)
        chains = [_monomeric_to_chain(object_to_model, chain_id)]
    elif isinstance(object_to_model, MultimericObject):
        chains = []
        for interactor in object_to_model.interactors:
            chain_id = next(chain_id_gen)
            chain = _monomeric_to_chain(interactor, chain_id)
            chains.append(chain)
    else:
        logging.error("Unsupported object type for folding input conversion.")
        raise TypeError("Unsupported object type for folding input conversion.")

    return folding_input.Input(
        name=object_to_model.description,
        rng_seeds=[random_seed],
        chains=chains,
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

    # Handle output directory naming
    if os.path.exists(output_dir) and os.listdir(output_dir):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_output_dir = f'{output_dir}_{timestamp}'
        logging.warning(
            f'Output directory {output_dir} exists and is non-empty, using {new_output_dir} instead.'
        )
        output_dir = new_output_dir

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
    os.makedirs(output_dir, exist_ok=True)
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
    def predict(
        model_runner: ModelRunner,
        objects_to_model: List[Dict[Union[MultimericObject, MonomericObject, ChoppedObject], str]],
        random_seed: int,
        buckets: int,
        **kwargs,
    ):
        """Predicts structures for a list of objects using AlphaFold 3."""
        if isinstance(buckets, int):
            buckets = [buckets]
        buckets = tuple(int(b) for b in buckets)

        for mapping in objects_to_model:
            (object_to_model, output_dir), = mapping.items()
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                logging.error(f'Failed to create output directory {output_dir}: {e}')
                raise

            fold_input_obj = _convert_to_fold_input(object_to_model, random_seed)
            logging.info(f'Processing fold input {fold_input_obj.name}')

            prediction_result = process_fold_input(
                fold_input=fold_input_obj,
                model_runner=model_runner,
                output_dir=output_dir,
                buckets=buckets,
            )

            yield {
                object_to_model: {
                    "prediction_results": prediction_result,
                    "output_dir": output_dir
                }
            }

    @staticmethod
    def postprocess(**kwargs) -> None:
        pass

