#!/usr/bin/env python

""" CLI inferface for performing structure prediction.

    Copyright (c) 2024 European Molecular Biology Laboratory
    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
            Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""
import os
import pickle
import glob
import shutil
import lzma
import random
import sys
from pathlib import Path
import jax
from absl import app, flags, logging
from typing import Tuple, Union

from alphapulldown.folding_backend import backend
from alphapulldown.folding_backend.alphafold_backend import ModelsToRelax
from alphapulldown.objects import MonomericObject, MultimericObject
from alphapulldown.utils.modelling_setup import create_interactors, create_custom_info, parse_fold

logging.set_verbosity(logging.INFO)

# JAX GPU setup
jax.local_devices(backend='gpu')

# CLI flags
flags.DEFINE_list('input', None, 'List of folds: fasta_path:number:start-stop,...')
flags.DEFINE_list('output_directory', None, 'Output dirs (one per fold or common)')
flags.DEFINE_string('data_directory', None, 'Model weights and parameters dir')
flags.DEFINE_list('features_directory', None, 'Directories with precomputed monomer features')

# Folding settings
flags.DEFINE_integer('num_cycle', 3, 'Recycle count')
flags.DEFINE_integer('num_predictions_per_model', 1, 'Predictions per model')
flags.DEFINE_boolean('pair_msa', True, 'Pair MSAs in multimers')
flags.DEFINE_boolean('skip_templates', False, 'Ignore template features')
flags.DEFINE_boolean('msa_depth_scan', False, 'Log-scale MSA depth scans')
flags.DEFINE_list('model_names', None, 'Custom model names list')
flags.DEFINE_integer('msa_depth', None, 'Limit MSA sequences')
flags.DEFINE_enum('model_preset', 'monomer', ['monomer','monomer_casp14','monomer_ptm','multimer'], 'Model preset')
flags.DEFINE_enum_class('models_to_relax', ModelsToRelax.NONE, ModelsToRelax, 'Relaxation strategy')
flags.DEFINE_boolean('convert_to_modelcif', True, 'Convert PDB to mmCIF')
flags.DEFINE_boolean('allow_resume', True, 'Resume or restart predictions')
flags.DEFINE_boolean('benchmark', False, 'Measure GPU timing vs. compile time')
flags.DEFINE_integer('random_seed', None, 'Random seed for reproducibility')
flags.DEFINE_string('crosslinks', None, 'AlphaLink2 crosslink pickle')
flags.DEFINE_string('jax_compilation_cache_dir', None, 'JAX compilation cache dir')
flags.DEFINE_list('buckets', ['256','512','768','1024','1280','1536','2048','2560','3072','3584','4096','4608','5120'],
                  'JAX token bucket sizes')
flags.DEFINE_enum('flash_attention_implementation', 'triton', ['triton','cudnn','xla'], 'Flash attention impl')
flags.DEFINE_integer('num_diffusion_samples', 5, 'Diffusion samples count')

# Post-process settings
flags.DEFINE_boolean('compress_result_pickles', False, 'Gzip result pickles')
flags.DEFINE_boolean('remove_result_pickles', False, 'Remove pickles after save')
flags.DEFINE_boolean('remove_keys_from_pickles', True, 'Strip large keys from pickles')
flags.DEFINE_boolean('use_gpu_relax', True, 'Amber relax on GPU')

# Global
flags.DEFINE_string('protein_delimiter', '+', 'Multi-chain delimiter')
flags.DEFINE_string('fold_backend', 'alphafold', 'Folding backend')

FLAGS = flags.FLAGS


def predict_structure(
    jobs: list,
    model_flags: dict,
    postprocess_flags: dict,
    backend_name: str
):
    backend.change_backend(backend_name)
    runners = backend.setup(**model_flags)

    seed = FLAGS.random_seed
    if seed is None:
        seed = random.randrange(sys.maxsize // len(runners['model_runners']))

    predictions = backend.predict(
        **runners,
        objects_to_model=jobs,
        random_seed=seed,
        **model_flags
    )

    for job in predictions:
        obj, res = next(iter(job.items()))
        backend.postprocess(
            **postprocess_flags,
            multimeric_object=obj,
            prediction_results=res['prediction_results'],
            output_dir=res['output_dir']
        )


def pre_modelling_setup(
    interactors: list,
    flags,
    outdir: str
) -> Tuple[Union[MonomericObject, MultimericObject], dict, dict, str]:
    # Build monomeric or multimeric container
    if len(interactors) > 1:
        container = MultimericObject(
            interactors=interactors,
            pair_msa=flags.pair_msa,
            multimeric_template=flags.model_preset=='multimer',
            multimeric_template_meta_data=flags.description_file,
            multimeric_template_dir=flags.path_to_mmt
        )
    else:
        container = interactors[0]

    os.makedirs(outdir, exist_ok=True)

    # Copy latest meta .json for each monomer
    for feat_dir in flags.features_directory:
        for mono in (interactors if isinstance(container, MultimericObject) else [container]):
            desc = mono.description
            matches = glob.glob(f"{feat_dir}/{desc}_feature_metadata_*.json*")
            if not matches:
                logging.warning(f"No metadata for {desc} in {feat_dir}")
                continue
            latest = max(matches, key=os.path.getmtime)
            dest = Path(outdir) / Path(latest).name.replace('.xz','')
            if latest.endswith('.xz'):
                with lzma.open(latest,'rb') as src, open(dest,'wb') as dst:
                    dst.write(src.read())
            else:
                shutil.copy(latest, dest)

    # Prepare flags for backend
    common = {
        'model_dir': flags.data_directory,
        'num_cycle': flags.num_cycle,
        'num_multimer_predictions_per_model': flags.num_predictions_per_model,
        'crosslinks': flags.crosslinks,
        'allow_resume': flags.allow_resume,
        'jax_compilation_cache_dir': flags.jax_compilation_cache_dir,
        'buckets': flags.buckets,
        'flash_attention_implementation': flags.flash_attention_implementation,
        'num_diffusion_samples': flags.num_diffusion_samples,
        'skip_templates': flags.skip_templates,
        'benchmark': flags.benchmark,
    }
    if isinstance(container, MultimericObject):
        common.update({
            'msa_depth_scan': flags.msa_depth_scan,
            'model_names_custom': flags.model_names,
            'msa_depth': flags.msa_depth
        })

    post = {
        'compress_pickles': flags.compress_result_pickles,
        'remove_pickles': flags.remove_result_pickles,
        'remove_keys_from_pickles': flags.remove_keys_from_pickles,
        'use_gpu_relax': flags.use_gpu_relax,
        'models_to_relax': flags.models_to_relax,
        'convert_to_modelcif': flags.convert_to_modelcif,
    }

    return container, common, post, outdir


def main(_argv):
    parsed = parse_fold(FLAGS.input, FLAGS.features_directory, FLAGS.protein_delimiter)
    data = create_custom_info(parsed)
    all_interactors = create_interactors(data, FLAGS.features_directory)

    # Pair outputs
    outs = FLAGS.output_directory or []
    if len(outs) not in (1, len(all_interactors)):
        raise ValueError("Provide one or matching output dirs.")
    outs = outs * (len(all_interactors) // len(outs))

    jobs = []
    for interactors, out in zip(all_interactors, outs):
        container, m_flags, p_flags, od = pre_modelling_setup(interactors, FLAGS, out)
        jobs.append({container: od})

    predict_structure(
        jobs=jobs,
        model_flags=m_flags,
        postprocess_flags=p_flags,
        backend_name=FLAGS.fold_backend
    )


if __name__ == '__main__':
    flags.mark_flags_as_required(['input','output_directory','data_directory','features_directory'])
    app.run(main)
