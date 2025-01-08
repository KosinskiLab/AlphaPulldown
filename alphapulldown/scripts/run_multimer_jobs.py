#!/usr/bin/env python3
""" A script to perform structure prediction.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Authors: Dingquan Yu, Valentin Maurer <name.surname@embl-hamburg.de>
"""
import io
import warnings
import subprocess
from absl import app, logging, flags
import os
import sys
from alphapulldown.scripts.run_structure_prediction import FLAGS
from alphapulldown.utils.create_combinations import process_files

logging.set_verbosity(logging.INFO)


flags.DEFINE_enum("mode", "pulldown", ["pulldown", "all_vs_all", "homo-oligomer", "custom"],
                  "choose the mode of running multimer jobs")
flags.DEFINE_string("oligomer_state_file", None, "path to oligomer state files")
flags.DEFINE_list("protein_lists", None, "protein list files")
flags.DEFINE_string("alphalink_weight", None, "Path to AlphaLink neural network weights")
flags.DEFINE_string("unifold_param", None, "Path to UniFold neural network weights")
flags.DEFINE_boolean("use_unifold", False,
                     "Whether unifold models are going to be used. Default it False")
flags.DEFINE_boolean("use_alphalink", False,
                     "Whether alphalink models are going to be used. Default it False")
flags.DEFINE_enum("unifold_model_name", "multimer_af2",
                  ["multimer_af2", "multimer_ft", "multimer", "multimer_af2_v3", "multimer_af2_model45_v3"],
                  "choose unifold model structure")
flags.DEFINE_integer("job_index", None, "index of sequence in the fasta file, starting from 1")
flags.DEFINE_boolean("dry_run", False, "Report number of jobs that would be run and exit without running them")

#Different flag names from alphafold
flags.DEFINE_list("monomer_objects_dir", None, "a list of directories where monomer objects are stored")
flags.DEFINE_string("output_path", None, "output directory where the region data is going to be stored")
flags.DEFINE_string("data_dir", None, "Path to params directory")
del(FLAGS.models_to_relax)
flags.DEFINE_enum("models_to_relax",'None',['None','All','Best'],
                  "Which models to relax. Default is None, meaning no model will be relaxed")

def main(argv):
    FLAGS(argv)
    protein_lists = FLAGS.protein_lists
    if FLAGS.mode == "all_vs_all":
        protein_lists = [FLAGS.protein_lists[0], FLAGS.protein_lists[0]]
    elif FLAGS.mode == "homo-oligomer":
        protein_lists = [FLAGS.oligomer_state_file]
        warnings.warn(
            "Mode homo-oligomer is deprecated. Please switch to the new custom format.",
            DeprecationWarning,
        )

    buffer = io.StringIO()
    _ = process_files(
        input_files=protein_lists,
        output_path=buffer,
        exclude_permutations = True
    )
    buffer.seek(0)
    all_folds = buffer.readlines()
    all_folds = [x.strip().replace(",", ":") for x in all_folds]
    all_folds = [x.strip().replace(";", "+") for x in all_folds]
    if FLAGS.dry_run:
        logging.info(f"Dry run: the total number of jobs to be run: {len(all_folds)}")
        sys.exit(0)
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    base_command = [f"python3 {parent_dir}/run_structure_prediction.py"]

    fold_backend, model_dir = "alphafold", FLAGS.data_dir
    if FLAGS.use_alphalink:
        fold_backend, model_dir = "alphalink", FLAGS.alphalink_weight
    elif FLAGS.use_unifold:
        fold_backend, model_dir = "unifold", FLAGS.unifold_param

    constant_args = {
        "--input": None,
        "--output_directory": FLAGS.output_path,
        "--num_cycle": FLAGS.num_cycle,
        "--num_predictions_per_model": FLAGS.num_predictions_per_model,
        "--data_directory": model_dir,
        "--features_directory": FLAGS.monomer_objects_dir,
        "--pair_msa": FLAGS.pair_msa,
        "--msa_depth_scan": FLAGS.msa_depth_scan,
        "--multimeric_template": FLAGS.multimeric_template,
        "--model_names": FLAGS.model_names,
        "--msa_depth": FLAGS.msa_depth,
        "--crosslinks": FLAGS.crosslinks,
        "--fold_backend": fold_backend,
        "--description_file": FLAGS.description_file,
        "--path_to_mmt": FLAGS.path_to_mmt,
        "--compress_result_pickles": FLAGS.compress_result_pickles,
        "--remove_result_pickles": FLAGS.remove_result_pickles,
        "--remove_keys_from_pickles" : FLAGS.remove_keys_from_pickles,
        "--use_ap_style": True,
        "--use_gpu_relax": FLAGS.use_gpu_relax,
        "--protein_delimiter": FLAGS.protein_delimiter,
        "--desired_num_res": FLAGS.desired_num_res,
        "--desired_num_msa": FLAGS.desired_num_msa,
        "--models_to_relax": FLAGS.models_to_relax
    }

    command_args = {}
    for k, v in constant_args.items():
        if v is None:
            continue
        elif v is False:
            updated_key = f"--no{k.split('--')[-1]}"
            command_args[updated_key] = ""
        elif v is True:
            command_args[k] = ""
        elif isinstance(v, list):
            command_args[k] = ",".join([str(x) for x in v])
        else:
            command_args[k] = v

    job_indices = list(range(len(all_folds)))
    if FLAGS.job_index is not None:
        job_index = FLAGS.job_index - 1
        if job_index >= len(all_folds):
            raise IndexError(
                f"Job index can be no larger than {len(all_folds)} (got {job_index})."
            )
        job_indices = [job_index, ]

    if (FLAGS.desired_num_res is not None) and (FLAGS.desired_num_msa is not None):
        # This means all the folds in all_folds are going to be modelled together
        # then no need to iterate through the job_indices
        command_args["--input"] = ",".join(all_folds)
        command = base_command.copy()
        for arg, value in command_args.items():
            command.extend([str(arg), str(value)])
        subprocess.run(" ".join(command), check=True, shell=True)
    else:
        for job_index in job_indices:
            command_args["--input"] = all_folds[job_index]
            command = base_command.copy()
            for arg, value in command_args.items():
                command.extend([str(arg), str(value)])
            subprocess.run(" ".join(command), check=True, shell=True)


if __name__ == "__main__":
    app.run(main)
