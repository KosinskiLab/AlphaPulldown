#!/usr/bin/env python

""" CLI inferface for performing structure prediction.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
            Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""
import pickle

import jax
gpus = jax.local_devices(backend='gpu')
from absl import flags, app
import os
from os import makedirs
from typing import Dict, List, Union, Tuple
from os.path import join, basename
from absl import logging
import glob
import shutil
import lzma
import random
import sys
from alphapulldown.folding_backend import backend
from alphapulldown.folding_backend.alphafold_backend import ModelsToRelax
from alphapulldown.objects import MultimericObject, MonomericObject, ChoppedObject
from alphapulldown.utils.modelling_setup import create_interactors, create_custom_info, parse_fold

logging.set_verbosity(logging.INFO)

# Required arguments
flags.DEFINE_list(
    'input', None,
    'Folds in format [fasta_path:number:start-stop],[...],.',
    short_name='i')
flags.DEFINE_list(
    'output_directory', None,
    'Path to output directory. Will be created if not exists.',
    short_name='o')
flags.DEFINE_string(
    'data_directory', None,
    'Path to directory containing model weights and parameters.')
flags.DEFINE_list(
    'features_directory', None,
    'Path to computed monomer features.')

# AlphaFold settings
flags.DEFINE_integer('num_cycle', 3,
                     'Number of recycles, defaults to 3.')
flags.DEFINE_integer('num_predictions_per_model', 1,
                     'Number of predictions per model, defaults to 1.')
flags.DEFINE_boolean('pair_msa', True,
                     'Whether to pair the MSAs when constructing multimer objects. Default is True')
flags.DEFINE_boolean('save_features_for_multimeric_object', False,
                     'Whether to save features for multimeric object.')
flags.DEFINE_boolean('skip_templates', False,
                     'Do not use template features when modelling')
flags.DEFINE_boolean('msa_depth_scan', False,
                     'Run predictions for each model with logarithmically distributed MSA depth.')
flags.DEFINE_boolean('multimeric_template', False,
                     'Whether to use multimeric templates.')
flags.DEFINE_list('model_names', None,
                    'A list of names of models to use, e.g. model_2_multimer_v3 (default: all models).')
flags.DEFINE_integer('msa_depth', None,
                     'Number of sequences to use from the MSA (by default is taken from AF model config).')
flags.DEFINE_string('description_file', None,
                    'Path to the text file with multimeric template instruction.')
flags.DEFINE_string('path_to_mmt', None,
                    'Path to directory with multimeric template mmCIF files.')
flags.DEFINE_integer('desired_num_res', None,
                     'A desired number of residues to pad')
flags.DEFINE_integer('desired_num_msa', None,
                     'A desired number of msa to pad')
flags.DEFINE_enum_class(
    "models_to_relax",
    ModelsToRelax.NONE,
    ModelsToRelax,
    "The models to run the final relaxation step on. "
    "If `all`, all models are relaxed, which may be time "
    "consuming. If `best`, only the most confident model "
    "is relaxed. If `none`, relaxation is not run. Turning "
    "off relaxation might result in predictions with "
    "distracting stereochemical violations but might help "
    "in case you are having issues with the relaxation "
    "stage.",
)
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_boolean('convert_to_modelcif', True,
                     'Whether to convert predicted pdb files to modelcif format. Default True.')
flags.DEFINE_boolean('allow_resume', True,
                     'Whether to allow resuming predictions from previous runs or start anew. Default True.')
# AlphaLink2 settings
flags.DEFINE_string('crosslinks', None, 'Path to crosslink information pickle for AlphaLink.')

# AlphaFold3 settings
# JAX inference performance tuning.
flags.DEFINE_string(
    'jax_compilation_cache_dir',
    None,
    'Path to a directory for the JAX compilation cache.',
)
flags.DEFINE_list(
    'buckets',
    # pyformat: disable
    ['64', '128', '256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072',
     '3584', '4096', '4608', '5120'],
    # pyformat: enable
    'Strictly increasing order of token sizes for which to cache compilations.'
    ' For any input with more tokens than the largest bucket size, a new bucket'
    ' is created for exactly that number of tokens.',
)
flags.DEFINE_enum(
    'flash_attention_implementation',
    default='triton',
    enum_values=['triton', 'cudnn', 'xla'],
    help=(
        "Flash attention implementation to use. 'triton' and 'cudnn' uses a"
        ' Triton and cuDNN flash attention implementation, respectively. The'
        ' Triton kernel is fastest and has been tested more thoroughly. The'
        " Triton and cuDNN kernels require Ampere GPUs or later. 'xla' uses an"
        ' XLA attention implementation (no flash attention) and is portable'
        ' across GPU devices.'
    ),
)
flags.DEFINE_integer(
    'num_diffusion_samples',
    5,
    'Number of diffusion samples to generate.',
)

# Post-processing settings
flags.DEFINE_boolean('compress_result_pickles', False,
                     'Whether the result pickles are going to be gzipped. Default False.')
flags.DEFINE_boolean('remove_result_pickles', False,
                     'Whether the result pickles are going to be removed')
flags.DEFINE_boolean('remove_keys_from_pickles',True,
                     'Whether to remove aligned_confidence_probs, distogram and masked_msa from pickles')
flags.DEFINE_boolean('use_ap_style', False,
                     'Change output directory to include a description of the fold '
                     'as seen in previous alphapulldown versions.')
flags.DEFINE_boolean('use_gpu_relax', True,
                     'Whether to run Amber relaxation on GPU. Default is True')

# Global settings
flags.DEFINE_string('protein_delimiter', '+', 'Delimiter for proteins of a single fold.')
flags.DEFINE_string('fold_backend', 'alphafold',
                    'Folding backend that should be used for structure prediction.')

FLAGS = flags.FLAGS

def predict_structure(
    objects_to_model: List[Dict[str, Union[MultimericObject, MonomericObject, ChoppedObject, str]]],
    model_flags: Dict,
    postprocess_flags: Dict,
    fold_backend: str = "alphafold"
) -> None:
    """
    Predict structural features of multimers using specified models and configurations.

    Parameters
    ----------
    objects_to_model : A list of dictionaries. Each dictionary has a key of either:
        - MultimericObject, MonomericObject, or ChoppedObject for AlphaPulldown objects
        - Dict[str, str] with 'json_input' key for JSON inputs
       The value of each dictionary is the corresponding output_dir to save the modelling results.
    model_flags : Dict
        Dictionary of flags passed to the respective backend's predict function.
    postprocess_flags : Dict
        Dictionary of flags passed to the respective backend's postprocess function.
    fold_backend : str, optional
        Backend used for folding, defaults to alphafold.
    """
    backend.change_backend(backend_name=fold_backend)
    model_runners_and_configs = backend.setup(**model_flags)
    if FLAGS.random_seed is not None:
        random_seed = FLAGS.random_seed
    else:
        if fold_backend in ['alphafold', 'alphalink']:
            random_seed = random.randrange(sys.maxsize // len(model_runners_and_configs["model_runners"]))
        elif fold_backend=='alphafold3':
            random_seed = random.randrange(2**32 - 1)
    predicted_jobs = backend.predict(
        **model_runners_and_configs,
        objects_to_model=objects_to_model,
        random_seed=random_seed,
        **model_flags
    )

    for predicted_job in predicted_jobs:
        object_to_model = predicted_job['object']
        prediction_results = predicted_job['prediction_results']
        output_dir = predicted_job['output_dir']
        backend.postprocess(
            **postprocess_flags,
            multimeric_object=object_to_model,
            prediction_results=prediction_results,
            output_dir=output_dir
        )

def pre_modelling_setup(
    interactors : List[Union[MonomericObject, ChoppedObject]], output_dir) -> Tuple[Union[MultimericObject,MonomericObject, ChoppedObject], str]:
    """
    A function that sets up objects to be modelled and handles output directory preparation.

    Args:
    interactors: A list of MonomericObject or ChoppedObject. If len(interactors) == 1, 
    that means a monomeric modelling job should be done. Otherwise, it will be a multimeric modelling job
    output_dir: base output directory

    Return:
    A MultimericObject or MonomericObject
    output_directory for this particular modelling job
    """
    if len(interactors) > 1:
        # this means it's going to be a MultimericObject
        object_to_model = MultimericObject(
            interactors=interactors,
            pair_msa=FLAGS.pair_msa,
            multimeric_template=FLAGS.multimeric_template,
            multimeric_template_meta_data=FLAGS.description_file,
            multimeric_template_dir=FLAGS.path_to_mmt,
        )
        if FLAGS.save_features_for_multimeric_object:
            pickle.dump(MultimericObject.feature_dict, open(join(output_dir, "multimeric_object_features.pkl"), "wb"))
    else:
        # means it's going to be a MonomericObject or a ChoppedObject
        object_to_model= interactors[0]
        object_to_model.input_seqs = [object_to_model.sequence]

    if FLAGS.use_ap_style:
        list_oligo = object_to_model.description.split("_and_")
        if len(list_oligo) == len(set(list_oligo)) : #no homo-oligomer
           output_dir = join(output_dir, object_to_model.description)
        else :
            old_output_dir = output_dir
            for oligo in set(list_oligo) :
                number_oligo = list_oligo.count(oligo)
                if output_dir == old_output_dir :
                    if number_oligo != 1 :
                        output_dir += f"/{oligo}_homo_{number_oligo}er"
                    else :
                        output_dir += f"/{oligo}"
                else :
                    if number_oligo != 1 :
                        output_dir += f"_and_{oligo}_homo_{number_oligo}er"
                    else :
                        output_dir += f"_and_{oligo}"
    if len(output_dir) > 4096: #max path length for most filesystems
        logging.warning(f"Output directory path is too long: {output_dir}."
                        "Please use a shorter path with --output_directory.")
    
    # Create parent directories first
    parent_dir = os.path.dirname(output_dir)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy features metadata to output directory
    for interactor in interactors:
        for feature_dir in FLAGS.features_directory:
            # meta.json is named the same way as the pickle file
            if isinstance(interactor, ChoppedObject):
                description = interactor.monomeric_description
            elif isinstance(interactor, MonomericObject):
                description = interactor.description
            meta_json = glob.glob(
                join(feature_dir, f"{description}_feature_metadata_*.json*")
            )
        if meta_json:
            # sort by modification time to take the latest
            meta_json.sort(key=os.path.getmtime, reverse=True)

            for feature_json in meta_json:
                output_path = join(output_dir, basename(feature_json))

                if feature_json.endswith(".json.xz"):
                    # Decompress before copying
                    decompressed_path = output_path.rstrip(".xz")
                    logging.info(f"Decompressing {feature_json} to {decompressed_path}")

                    with lzma.open(feature_json, "rb") as xz_file, open(decompressed_path, "wb") as json_file:
                        json_file.write(xz_file.read())
                else:
                    # Copy without decompression
                    logging.info(f"Copying {feature_json} to {output_dir}")
                    shutil.copyfile(feature_json, output_path)
        else:
            logging.warning(f"No feature metadata found for {interactor.description} in {feature_dir}")

    return object_to_model, output_dir

def main(argv):
    # Parse inputs and build interactor objects
    parsed = parse_fold(FLAGS.input, FLAGS.features_directory, FLAGS.protein_delimiter)
    data = create_custom_info(parsed)
    all_interactors = create_interactors(data, FLAGS.features_directory)
    n = len(FLAGS.input)
    # Normalize output dirs
    if len(FLAGS.output_directory) == 1:
        out_dirs = FLAGS.output_directory * n
    elif len(FLAGS.output_directory) == n:
        out_dirs = FLAGS.output_directory
    else:
        raise ValueError(
            "Either specify one output_directory for all folds or one per fold."
        )

    # Define default model and postprocess flags
    default_model_flags = {
        "model_name": "monomer_ptm",
        "num_cycle": FLAGS.num_cycle,
        "model_dir": FLAGS.data_directory,
        "num_predictions_per_model": FLAGS.num_predictions_per_model,
        "crosslinks": FLAGS.crosslinks,
        "desired_num_res": FLAGS.desired_num_res,
        "desired_num_msa": FLAGS.desired_num_msa,
        "skip_templates": FLAGS.skip_templates,
        "allow_resume": FLAGS.allow_resume,
        "num_diffusion_samples": FLAGS.num_diffusion_samples,
        "flash_attention_implementation": FLAGS.flash_attention_implementation,
        "buckets": FLAGS.buckets,
        "jax_compilation_cache_dir": FLAGS.jax_compilation_cache_dir,
        "features_directory": FLAGS.features_directory,
    }

    default_postprocess_flags = {
        "compress_pickles": FLAGS.compress_result_pickles,
        "remove_pickles": FLAGS.remove_result_pickles,
        "remove_keys_from_pickles": FLAGS.remove_keys_from_pickles,
        "use_gpu_relax": FLAGS.use_gpu_relax,
        "models_to_relax": FLAGS.models_to_relax,
        "features_directory": FLAGS.features_directory,
        "convert_to_modelcif": FLAGS.convert_to_modelcif
    }

    # Prepare the list of jobs
    objects_to_model = []
    final_model_flags = default_model_flags
    final_postprocess_flags = default_postprocess_flags
    
    for interactors, out_dir in zip(all_interactors, out_dirs):
        if not interactors:
            continue

        # Separate JSON-only entries
        json_dicts = [
            d
            for d in interactors
            if isinstance(d, dict) and 'json_input' in d
        ]
        prot_objs = [
            x for x in interactors
            if not (isinstance(x, dict) and 'json_input' in x)
        ]

        # First handle any protein objects
        if prot_objs:
            obj, real_out = pre_modelling_setup(
                prot_objs, output_dir=out_dir
            )
            objects_to_model.append({'object': obj, 'output_dir': real_out})
            
            # Update final flags based on object type
            final_model_flags = default_model_flags.copy()
            final_postprocess_flags = default_postprocess_flags.copy()
            
            if isinstance(obj, MultimericObject):
                final_model_flags.update({
                    "model_name": "multimer",
                    "msa_depth_scan": FLAGS.msa_depth_scan,
                    "model_names_custom": FLAGS.model_names,
                    "msa_depth": FLAGS.msa_depth
                })
        # Then handle any number of JSON inputs
        for json_dict in json_dicts:
            objects_to_model.append({'object': json_dict, 'output_dir': out_dir})

    if objects_to_model:
        predict_structure(
            objects_to_model=objects_to_model,
            model_flags=final_model_flags,
            postprocess_flags=final_postprocess_flags,
            fold_backend=FLAGS.fold_backend
        )


if __name__ == '__main__':
    flags.mark_flag_as_required('input')
    flags.mark_flag_as_required('output_directory')
    flags.mark_flag_as_required('data_directory')
    flags.mark_flag_as_required('features_directory')
    app.run(main)
