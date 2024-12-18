#!/usr/bin/env python
"""
    Copyright (c) 2024 European Molecular Biology Laboratory

    Email: alphapulldown@embl-hamburg.de
"""
from absl import flags, app
from os import makedirs
from typing import Dict, List, Union, Tuple
from os.path import join, basename
from absl import logging
import glob
import shutil

from alphapulldown.folding_backend import backend
from alphapulldown.folding_backend.backend_flags import (define_alphafold_flags,
                                                             define_alphafold3_flags,
                                                             define_alphalink_flags)
from alphapulldown.objects import MultimericObject, MonomericObject, ChoppedObject
from alphapulldown.utils.modelling_setup import create_interactors, create_custom_info, parse_fold


logging.set_verbosity(logging.INFO)

# Global / generic flags
flags.DEFINE_string('protein_delimiter', '+', 'Delimiter for proteins of a single fold.')
flags.DEFINE_string('fold_backend', 'alphafold',
                    'Folding backend that should be used for structure prediction.')

# Required arguments
flags.DEFINE_list('input', None, 'Folds: [fasta_path:number:start-stop],[...].', short_name='i')
flags.DEFINE_list('output_directory', None, 'Path to output directory.', short_name='o')
flags.DEFINE_string('data_directory', None, 'Path to directory with model weights.')
flags.DEFINE_list('features_directory', None, 'Path to computed monomer features.')

# Common flags
flags.DEFINE_boolean('use_ap_style', False, 'Change output directory style.')
flags.DEFINE_boolean('compress_result_pickles', False, 'Gzip-compress result pickles.')
flags.DEFINE_boolean('remove_result_pickles', False, 'Remove result pickles after prediction.')
flags.DEFINE_boolean('remove_keys_from_pickles', True, 'Remove large keys from pickles.')
flags.DEFINE_boolean('use_gpu_relax', True, 'Use GPU for Amber relaxation.')

FLAGS = flags.FLAGS
FLAGS(app.parse_flags_with_usage)  # Parse known flags to identify backend

# Dynamically load backend-specific flags
if FLAGS.fold_backend == 'alphafold':
    define_alphafold_flags()
elif FLAGS.fold_backend == 'alphafold3':
    define_alphafold_flags()
    define_alphafold3_flags()
elif FLAGS.fold_backend == 'alphalink':
    define_alphalink_flags()
else:
    logging.warning(f"No specific flags defined for backend {FLAGS.fold_backend}.")

FLAGS(app.parse_flags_with_usage)  # Re-parse after backend flags are defined

def predict_structure(
    objects_to_model: List[Dict[Union[MultimericObject, MonomericObject, ChoppedObject], str]],
    model_flags: Dict,
    postprocess_flags: Dict,
    random_seed: int = 42,
    fold_backend: str = "alphafold"
) -> None:
    backend.change_backend(backend_name=fold_backend)
    model_runners_and_configs = backend.setup(**model_flags)

    predicted_jobs = backend.predict(
        **model_runners_and_configs,
        objects_to_model=objects_to_model,
        random_seed=random_seed,
        **model_flags
    )

    for predicted_job in predicted_jobs:
        object_to_model, prediction_results = next(iter(predicted_job.items()))
        backend.postprocess(
            **postprocess_flags,
            multimeric_object=object_to_model,
            prediction_results=prediction_results['prediction_results'],
            output_dir=prediction_results['output_dir']
        )

def pre_modelling_setup(
    interactors: List[Union[MonomericObject, ChoppedObject]], flags, output_dir
) -> Tuple[Union[MultimericObject,MonomericObject, ChoppedObject], dict, dict, str]:

    if len(interactors) > 1:
        object_to_model = MultimericObject(
            interactors=interactors,
            pair_msa= flags.pair_msa,
            multimeric_template=flags.multimeric_template,
            multimeric_template_meta_data=flags.description_file,
            multimeric_template_dir=flags.path_to_mmt,
        )
    else:
        object_to_model= interactors[0]
        object_to_model.input_seqs = [object_to_model.sequence]

    flags_dict = {
        "model_name": "monomer_ptm",
        "num_cycle": flags.num_cycle,
        "model_dir": flags.data_directory,
        "num_multimer_predictions_per_model": flags.num_predictions_per_model,
        "crosslinks": flags.crosslinks,
        "desired_num_res": flags.desired_num_res,
        "desired_num_msa": flags.desired_num_msa,
        "skip_templates": flags.skip_templates,
        "allow_resume": flags.allow_resume,
        "num_diffusion_samples": getattr(flags, 'num_diffusion_samples', None),
        "flash_attention_implementation": getattr(flags, 'flash_attention_implementation', None),
        "buckets": getattr(flags, 'buckets', None),
        "jax_compilation_cache_dir": getattr(flags, 'jax_compilation_cache_dir', None),
    }

    if isinstance(object_to_model, MultimericObject):
        flags_dict["model_name"] = "multimer"
        flags_dict["msa_depth_scan"] = flags.msa_depth_scan
        flags_dict["model_names_custom"] = flags.model_names
        flags_dict["msa_depth"] = flags.msa_depth

    postprocess_flags = {
        "compress_pickles": flags.compress_result_pickles,
        "remove_pickles": flags.remove_result_pickles,
        "remove_keys_from_pickles": flags.remove_keys_from_pickles,
        "use_gpu_relax": flags.use_gpu_relax,
        "models_to_relax": flags.models_to_relax,
        "features_directory": flags.features_directory,
        "convert_to_modelcif": flags.convert_to_modelcif
    }

    if flags.use_ap_style:
        output_dir = join(output_dir, object_to_model.description)
    if len(output_dir) > 4096:
        logging.warning(f"Output directory path is too long: {output_dir}.")
    makedirs(output_dir, exist_ok=True)

    for interactor in interactors:
        for feature_dir in flags.features_directory:
            if isinstance(interactor, ChoppedObject):
                description = interactor.monomeric_description
            else:
                description = interactor.description
            meta_json = glob.glob(
                join(feature_dir, f"{description}_feature_metadata_*.json")
            )
            if meta_json:
                feature_json = meta_json[0]
                logging.info(f"Copying {feature_json} to {output_dir}")
                shutil.copyfile(feature_json, join(output_dir, basename(feature_json)))
            else:
                logging.warning(f"No feature metadata found for {interactor.description} in {output_dir}")

    return object_to_model, flags_dict, postprocess_flags, output_dir

def main(argv):
    parsed_input = parse_fold(FLAGS.input, FLAGS.features_directory, FLAGS.protein_delimiter)
    data = create_custom_info(parsed_input)
    all_interactors = create_interactors(data, FLAGS.features_directory)
    objects_to_model = []

    if len(FLAGS.input) != len(FLAGS.output_directory):
        FLAGS.output_directory *= len(FLAGS.input)

    if len(FLAGS.input) != len(FLAGS.output_directory):
        raise ValueError(
            "Either specify one output_directory per fold or one for all folds."
        )

    for index, interactors in enumerate(all_interactors):
        object_to_model, flags_dict, postprocess_flags, output_dir = pre_modelling_setup(
            interactors, FLAGS, output_dir=FLAGS.output_directory[index]
        )
        objects_to_model.append({object_to_model: output_dir})

    predict_structure(
        objects_to_model=objects_to_model,
        model_flags=flags_dict,
        fold_backend=FLAGS.fold_backend,
        postprocess_flags=postprocess_flags
    )

if __name__ == '__main__':
    flags.mark_flag_as_required('input')
    flags.mark_flag_as_required('output_directory')
    flags.mark_flag_as_required('data_directory')
    flags.mark_flag_as_required('features_directory')
    app.run(main)
