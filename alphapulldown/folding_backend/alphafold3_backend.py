"""Implements structure prediction backend using AlphaFold3.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""

import os
import pathlib
import time
import typing
from datetime import datetime
from typing import Dict, List, Sequence, Union, Optional, Tuple, Any

import dataclasses
import functools
from xml.dom import NotFoundErr

import jax
import numpy as np
from jax import numpy as jnp

from alphafold3.data.templates import Templates
from alphafold3.data import structure_stores

from alphafold3.common import base_config
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.jax.attention import attention
from alphafold3.model import features
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import base_model
from alphafold3.model.components import utils
from alphafold3.model.diffusion import model as diffusion_model
import haiku as hk

from alphapulldown.objects import MultimericObject, MonomericObject, ChoppedObject

from absl import logging

from  alphapulldown.folding_backend.folding_backend import FoldingBackend

# Suppress specific warnings by setting verbosity to ERROR
logging.set_verbosity(logging.ERROR)


def _insert_release_date_into_mmcif(mmcif_string: str, revision_date: str = '2100-01-01') -> str:
    """
    Inserts the release date into the mmCIF string at the specified positions.

    Args:
        mmcif_string (str): The original mmCIF string.
        revision_date (str, optional): The release date in 'YYYY-MM-DD' format.
                                       Defaults to '2100-01-01'.

    Returns:
        str: The updated mmCIF string with the release date inserted.
    """

    # Split the mmCIF string into a list of lines
    pdb_data = mmcif_string.splitlines()

    # Define the lines to insert
    header_lines = [
        "_entry.id   pdb",
        "_pdbx_audit_revision_history.revision_date"
    ]
    release_date_line = f'"{revision_date}"'

    # Insert the header lines at index 2
    pdb_data.insert(2, "\n".join(header_lines))

    # Insert the release date at index 3
    pdb_data.insert(3, release_date_line)

    # Rejoin the lines into a single string
    updated_mmcif_string = "\n".join(pdb_data)

    return updated_mmcif_string


# Define the HHBLITS to Internal AA Type Mapping
HHBLITS_TO_INTERNAL_AATYPE = np.array([
    0,  # 0: 'A' -> 'ALA' -> 0
    1,  # 1: 'C' -> 'CYS' -> 1
    2,  # 2: 'D' -> 'ASP' -> 2
    3,  # 3: 'E' -> 'GLU' -> 3
    4,  # 4: 'F' -> 'PHE' -> 4
    5,  # 5: 'G' -> 'GLY' -> 5
    6,  # 6: 'H' -> 'HIS' -> 6
    7,  # 7: 'I' -> 'ILE' -> 7
    8,  # 8: 'K' -> 'LYS' -> 8
    9,  # 9: 'L' -> 'LEU' -> 9
    10,  # 10: 'M' -> 'MET' -> 10
    11,  # 11: 'N' -> 'ASN' -> 11
    12,  # 12: 'P' -> 'PRO' -> 12
    13,  # 13: 'Q' -> 'GLN' -> 13
    14,  # 14: 'R' -> 'ARG' -> 14
    15,  # 15: 'S' -> 'SER' -> 15
    16,  # 16: 'T' -> 'THR' -> 16
    17,  # 17: 'V' -> 'VAL' -> 17
    18,  # 18: 'W' -> 'TRP' -> 18
    19,  # 19: 'Y' -> 'TYR' -> 19
    20,  # 20: 'X' -> 'UNK' -> 20
    20,  # 21: '-' -> 'UNK' -> 20
], dtype=np.int32)

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
        result: base_model.ModelResult,
        target_name: str = '',
    ) -> typing.Iterable[base_model.InferenceResult]:
        ...


ModelT = typing.TypeVar('ModelT', bound=ConfigurableModel)


def make_model_config(
    *,
    model_class: type[ModelT] = diffusion_model.Diffuser,
    flash_attention_implementation: attention.Implementation = 'xla',  # Changed to 'xla'
    num_diffusion_samples: int = 5,
):
    """Returns a model config with some defaults overridden."""
    config = model_class.Config()
    if hasattr(config, 'global_config'):
        config.global_config.flash_attention_implementation = (
            flash_attention_implementation
        )
    if hasattr(config, 'heads'):
        config.heads.diffusion.eval.num_samples = num_diffusion_samples
    return config


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
    ) -> typing.Callable[[jnp.ndarray, features.BatchDict], base_model.ModelResult]:
        """Loads model parameters and returns a jitted model forward pass."""
        assert isinstance(self.config, self.model_class.Config)

        @hk.transform
        def forward_fn(batch):
            result = self.model_class(self.config)(batch)
            result['__identifier__'] = self.model_params['__meta__']['__identifier__']
            return result

        return functools.partial(
            jax.jit(forward_fn.apply, device=self.device), self.model_params
        )

    def run_inference(
        self, featurised_example: features.BatchDict, rng_key: jnp.ndarray
    ) -> base_model.ModelResult:
        """Computes a forward pass of the model on a featurised example."""
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
        result['__identifier__'] = result['__identifier__'].tobytes()
        return result

    def extract_structures(
        self,
        batch: features.BatchDict,
        result: base_model.ModelResult,
        target_name: str,
    ) -> List[base_model.InferenceResult]:
        """Generates structures from model outputs."""
        return list(
            self.model_class.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
    """Stores the inference results (diffusion samples) for a single seed.

    Attributes:
      seed: The seed used to generate the samples.
      inference_results: The inference results, one per sample.
      full_fold_input: The fold input that must also include the results of
        running the data pipeline - MSA and templates.
    """

    seed: int
    inference_results: Sequence[base_model.InferenceResult]
    full_fold_input: folding_input.Input


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
    """Runs the full inference pipeline to predict structures for each seed."""

    logging.info(f'Featurising data for seeds {fold_input.rng_seeds}...')
    featurisation_start_time = time.time()
    ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input, buckets=buckets, ccd=ccd, verbose=True
    )
    logging.info(
        f'Featurising data for seeds {fold_input.rng_seeds} took '
        f' {time.time() - featurisation_start_time:.2f} seconds.'
    )
    all_inference_start_time = time.time()
    all_inference_results = []
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        logging.info(f'Running model inference for seed {seed}...')
        inference_start_time = time.time()
        rng_key = jax.random.PRNGKey(seed)
        try:
            result = model_runner.run_inference(example, rng_key)
        except Exception as e:
            logging.error(f"Failed during inference for seed {seed}: {e}")
            continue
        logging.info(
            f'Running model inference for seed {seed} took '
            f' {time.time() - inference_start_time:.2f} seconds.'
        )
        logging.info(f'Extracting output structures (one per sample) for seed {seed}...')
        extract_structures_start_time = time.time()
        try:
            inference_results = model_runner.extract_structures(
                batch=example, result=result, target_name=fold_input.name
            )
        except Exception as e:
            logging.error(f"Failed to extract structures for seed {seed}: {e}")
            continue
        logging.info(
            f'Extracting output structures (one per sample) for seed {seed} took '
            f' {time.time() - extract_structures_start_time:.2f} seconds.'
        )
        all_inference_results.append(
            ResultsForSeed(
                seed=seed,
                inference_results=inference_results,
                full_fold_input=fold_input,
            )
        )
        logging.info(
            'Running model inference and extracting output structures for seed'
            f' {seed} took  {time.time() - inference_start_time:.2f} seconds.'
        )
    logging.info(
        'Running model inference and extracting output structures for seeds'
        f' {fold_input.rng_seeds} took '
        f' {time.time() - all_inference_start_time:.2f} seconds.'
    )
    return all_inference_results


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
    """Writes outputs to the specified output directory."""
    import csv
    import alphafold3.cpp

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
            # The output terms of use are the same for all seeds/samples.
            terms_of_use=output_terms,
            name=job_name,
        )
        # Save csv of ranking scores with seeds and sample indices, to allow easier
        # comparison of ranking scores across different runs.
        with open(os.path.join(output_dir, 'ranking_scores.csv'), 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(['seed', 'sample', 'ranking_score'])
            writer.writerows(ranking_scores)


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> folding_input.Input | Sequence[ResultsForSeed]:
    """Runs data pipeline and/or inference on a single fold input.

    Args:
      fold_input: Fold input to process.
      data_pipeline_config: Data pipeline config to use. If None, skip the data
        pipeline.
      model_runner: Model runner to use. If None, skip inference.
      output_dir: Output directory to write to.
      buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
        of the model. If None, calculate the appropriate bucket size from the
        number of tokens. If not None, must be a sequence of at least one integer,
        in strictly increasing order. Will raise an error if the number of tokens
        is more than the largest bucket size.

    Returns:
      The processed fold input, or the inference results for each seed.

    Raises:
      ValueError: If the fold input has no chains.
    """
    import datetime

    logging.info(f'Processing fold input {fold_input.name}')

    if not fold_input.chains:
        logging.error('Fold input has no chains.')
        raise ValueError('Fold input has no chains.')

    if os.path.exists(output_dir) and os.listdir(output_dir):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_output_dir = f'{output_dir}_{timestamp}'
        logging.warning(
            f'Output directory {output_dir} exists and is non-empty, using instead '
            f'{new_output_dir}.'
        )
        output_dir = new_output_dir

    if model_runner is not None:
        # If we're running inference, check we can load the model parameters before
        # (possibly) launching the data pipeline.
        logging.info('Checking we can load the model parameters...')
        try:
            _ = model_runner.model_params
        except Exception as e:
            logging.error(f'Failed to load model parameters: {e}')
            raise

    if data_pipeline_config is None:
        logging.info('Skipping data pipeline...')
    else:
        logging.info('Running data pipeline...')
        fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

    logging.info(f'Output directory: {output_dir}')
    logging.info(f'Writing model input JSON to {output_dir}')
    write_fold_input_json(fold_input, output_dir)
    if model_runner is None:
        logging.info('Skipping inference...')
        output = fold_input
    else:
        logging.info(
            f'Predicting 3D structure for {fold_input.name} for seed(s)'
            f' {fold_input.rng_seeds}...'
        )
        all_inference_results = predict_structure(
            fold_input=fold_input,
            model_runner=model_runner,
            buckets=buckets,
        )
        logging.info(
            f'Writing outputs for {fold_input.name} for seed(s)'
            f' {fold_input.rng_seeds}...'
        )
        write_outputs(
            all_inference_results=all_inference_results,
            output_dir=output_dir,
            job_name=fold_input.sanitised_name(),
        )
        output = all_inference_results

    logging.info(f'Done processing fold input {fold_input.name}.')
    return output


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
    """Writes the input JSON to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json'), 'wt'
    ) as f:
        f.write(fold_input.to_json())


def get_structure_sequence(mmcif_file: str, chain_code: str) -> str:
    """Extracts the sequence for a given chain from an mmCIF file.

    Args:
        mmcif_file (str): Path to the mmCIF file.
        chain_code (str): Chain identifier.

    Returns:
        str: The amino acid sequence of the chain.

    Raises:
        ValueError: If the chain is not found or sequence extraction fails.
    """
    try:
        from alphafold3 import structure
        with open(mmcif_file, 'r') as f:
            cif_content = f.read()
        logging.info(cif_content)
        cif = structure.from_mmcif(
            mmcif_string=cif_content,  # Corrected keyword argument
            fix_mse_residues=True,
            fix_arginines=True,
            include_water=False,
            include_other=True,
            include_bonds=False,
        )
        sequence = cif.chain_single_letter_sequence().get(chain_code)
        if sequence is None:
            raise ValueError(f"Chain {chain_code} not found in {mmcif_file}.")
        return sequence
    except TypeError as te:
        logging.error(f"TypeError in from_mmcif: {te}")
        raise
    except Exception as e:
        logging.error(f"Error extracting sequence from {mmcif_file} for chain {chain_code}: {e}")
        raise


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AlphaFold3Backend(FoldingBackend):
    """A backend to perform structure prediction using AlphaFold 3."""

    def __init__(self):
        pass

    @staticmethod
    def setup(
        model_dir: str,
        num_diffusion_samples: int = 5,
        flash_attention_implementation: str = 'xla',  # Changed to 'xla'
        **kwargs,
    ) -> Dict:
        """
        Sets up the ModelRunner with the given configurations.

        Args:
            model_dir (str): Path to the directory containing model parameters.
            num_diffusion_samples (int): Number of diffusion samples.
            flash_attention_implementation (str): Flash attention implementation to use.
            mmcif_database_path (str): Path to the directory containing mmCIF files.

        Returns:
            Dict: A dictionary containing the ModelRunner and mmcif_database_path.
        """
        devices = jax.local_devices()
        device = devices[0]
        model_config = make_model_config(
            flash_attention_implementation=typing.cast(
                attention.Implementation, flash_attention_implementation
            ),
            num_diffusion_samples=num_diffusion_samples,
        )
        model_runner = ModelRunner(
            model_class=diffusion_model.Diffuser,
            config=model_config,
            device=device,
            model_dir=pathlib.Path(model_dir),
        )
        # Store the mmcif_database_path in the ModelRunner for accessibility
        # (Assuming it's needed elsewhere)
        return {'model_runner': model_runner}

    def predict(
        self,
        model_runner: ModelRunner,
        objects_to_model: List[Dict[Union[MultimericObject, MonomericObject, ChoppedObject], str]],
        random_seed: int = 42,
        data_pipeline_config: pipeline.DataPipelineConfig | None = None,
        buckets: Sequence[int] | None = None,
        **kwargs,
    ):
        """Predicts structures for a list of objects using AlphaFold 3.

        Args:
            model_runner (ModelRunner): The model runner instance.
            objects_to_model (List[Dict[Union[MultimericObject, MonomericObject, ChoppedObject], str]]):
                List of dictionaries mapping objects to model to their respective output directories.
            random_seed (int): The random seed for inference.
            data_pipeline_config (pipeline.DataPipelineConfig | None): Configuration for the data pipeline.
            buckets (Sequence[int] | None): Bucket sizes for padding.
            **kwargs: Additional arguments.

        Yields:
            Dict: A dictionary mapping the object to its prediction results and output directory.
        """
        for obj_dict in objects_to_model:
            for object_to_model, output_dir in obj_dict.items():
                # Convert object_to_model to fold_input.Input
                fold_input = self._convert_to_fold_input(
                    object_to_model, random_seed
                )
                # Run prediction
                #try:
                result = process_fold_input(
                    fold_input=fold_input,
                    data_pipeline_config=data_pipeline_config,
                    model_runner=model_runner,
                    output_dir=output_dir,
                    buckets=buckets,
                )
                yield {object_to_model: {"prediction_results": result, "output_dir": output_dir}}
               # except Exception as e:
               #     logging.error(f"Failed to predict structure for {object_to_model}: {e}")
               #     continue


    def _convert_to_fold_input(
            self,
            object_to_model,
            random_seed: int,
    ) -> folding_input.Input:
        import string
        import logging
        from alphafold.common import residue_constants
        import numpy as np

        def msa_array_to_a3m(msa_array):
            """Converts MSA numpy array to A3M formatted string."""
            msa_sequences = []
            for i, msa_seq in enumerate(msa_array):
                seq_str = ''.join([residue_constants.ID_TO_HHBLITS_AA.get(int(aa), 'X') for aa in msa_seq])
                msa_sequences.append(f'>sequence_{i}\n{seq_str}')
            a3m_str = '\n'.join(msa_sequences)
            return a3m_str

        # Define the chain ID generator
        def chain_id_generator():
            chain_letters = string.ascii_uppercase  # 'A' to 'Z'
            # First, yield single-letter IDs
            for c in chain_letters:
                yield c
            # Then, yield two-letter IDs in the specified order
            for first_letter in chain_letters:
                for second_letter in chain_letters:
                    yield first_letter + second_letter

        chain_id_gen = chain_id_generator()

        if isinstance(object_to_model, (MonomericObject, ChoppedObject)):
            chain_id = next(chain_id_gen)
            sequence = object_to_model.sequence
            # Get msa array
            msa_array = object_to_model.feature_dict.get('msa')
            if msa_array is not None:
                unpaired_msa = msa_array_to_a3m(msa_array)
            else:
                unpaired_msa = ''
            # Paired MSA is empty
            paired_msa = ''

            feature_dict = object_to_model.feature_dict
            templates = []
            if 'template_aatype' in feature_dict:
                num_templates = feature_dict['template_aatype'].shape[0]
                for i in range(num_templates):
                    # Parse PDB code and chain ID from template_domain_name
                    pdb_code_chain = str(feature_dict["template_domain_names"][i])
                    if '_' in pdb_code_chain:
                        pdb_code, chain_id_template = pdb_code_chain.split('_')
                    else:
                        pdb_code = pdb_code_chain
                        chain_id_template = 'A'  # Default to 'A' if no chain ID is specified

                    from alphafold.common.protein import Protein, to_mmcif

                    # Convert aatype from one-hot to integer indices using the new mapping
                    template_aatype_onehot = feature_dict["template_aatype"][i]  # Shape: (110, 22)
                    template_aatype_hhblits_id = np.argmax(template_aatype_onehot, axis=-1)  # Shape: (110,)

                    # Map HHBLITS IDs to internal AA type indices
                    template_aatype_int = HHBLITS_TO_INTERNAL_AATYPE[template_aatype_hhblits_id]
                    # Validation: Ensure all aatype indices are within 0-20
                    unique_aatypes = np.unique(template_aatype_int)
                    logging.debug(f"Template {i} Unique aatype indices:", unique_aatypes)

                    if not np.all((unique_aatypes >= 0) & (unique_aatypes <= 20)):
                        raise ValueError(f"Unexpected aatype indices found: {unique_aatypes}")

                    # Create the Protein object
                    template_mask = feature_dict["template_all_atom_masks"][i]  # Shape: (110, 37)
                    chain_index_array = np.zeros_like(feature_dict["residue_index"], dtype=int)  # Shape: (110,)

                    # Decode the template sequence if it's in bytes
                    template_sequence = feature_dict["template_sequence"][i]
                    if isinstance(template_sequence, bytes):
                        template_sequence = template_sequence.decode('utf-8')

                    template_protein = Protein(
                        atom_positions=feature_dict["template_all_atom_positions"][i],  # Shape: (110, 37, 3)
                        atom_mask=template_mask,  # Shape: (110, 37)
                        aatype=template_aatype_int,  # Shape: (110,), dtype: int32
                        residue_index=feature_dict["residue_index"],  # Shape: (110,)
                        chain_index=chain_index_array,  # Shape: (110,), dtype: int32
                        b_factors=np.zeros(template_mask.shape, dtype=float),  # Shape: (110, 37)
                    )
                    template_sequence = feature_dict["template_sequence"][i]
                    # Convert to mmCIF string using actual PDB code and chain ID
                    mmcif_string = to_mmcif(
                        prot=template_protein,
                        file_id=pdb_code,
                        model_type='Monomer',
                    )
                    new_mmcif_string = _insert_release_date_into_mmcif(mmcif_string)
                    # It's always one-to-one correspondence
                    query_to_template_map = {j: j for j in range(len(template_sequence))}

                    # Create the Template object
                    template = folding_input.Template(
                        mmcif=new_mmcif_string,
                        query_to_template_map=query_to_template_map,
                    )
                    templates.append(template)
            else:
                templates = []

            chain = folding_input.ProteinChain(
                id=chain_id,
                sequence=sequence,
                ptms=[],  # Provide PTMs if available
                unpaired_msa=unpaired_msa,
                paired_msa=paired_msa,
                templates=templates,
            )
            chains = [chain]
        else:
            logging.error("Unsupported object type for folding input conversion.")
            raise TypeError("Unsupported object type for folding input conversion.")

        # Create the fold input
        fold_input = folding_input.Input(
            name=object_to_model.description,
            rng_seeds=[random_seed],
            chains=chains,
        )
        return fold_input

    def postprocess(**kwargs) -> None:
        return None