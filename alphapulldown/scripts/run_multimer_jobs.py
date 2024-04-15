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
from alphapulldown.folding_backend.alphafold_backend import ModelsToRelax
from alphapulldown.utils.create_combinations import process_files

logging.set_verbosity(logging.INFO)


flags.DEFINE_enum(
    "mode",
    "pulldown",
    ["pulldown", "all_vs_all", "homo-oligomer", "custom"],
    "choose the mode of running multimer jobs",
)
flags.DEFINE_string(
    "output_path", None, "output directory where the region data is going to be stored"
)
flags.DEFINE_string("oligomer_state_file", None,
                    "path to oligomer state files")
flags.DEFINE_list(
    "monomer_objects_dir",
    None,
    "a list of directories where monomer objects are stored",
)
flags.DEFINE_list("protein_lists", None, "protein list files")

delattr(flags.FLAGS, "data_dir")
flags.DEFINE_string("data_dir", None, "Path to params directory")

flags.DEFINE_integer("num_cycle", 3, help="number of recycles")
flags.DEFINE_integer(
    "num_predictions_per_model", 1, "How many predictions per model. Default is 1"
)
flags.DEFINE_integer(
    "job_index", None, "index of sequence in the fasta file, starting from 1"
)
flags.DEFINE_boolean(
    "no_pair_msa", False, "do not pair the MSAs when constructing multimer objects"
)
flags.DEFINE_boolean(
    "multimeric_mode",
    False,
    "Run with multimeric template ",
)
flags.DEFINE_boolean(
    "msa_depth_scan",
    False,
    "Run predictions for each model with logarithmically distributed MSA depth",
)
flags.DEFINE_string(
    "model_names",
    None,
    "Names of models to use, e.g. model_2_multimer_v3 (default: all models)",
)
flags.DEFINE_integer(
    "msa_depth",
    None,
    "Number of sequences to use from the MSA (by default is taken from AF model config)",
)
flags.DEFINE_boolean(
    "use_unifold",
    False,
    "Whether unifold models are going to be used. Default it False",
)

flags.DEFINE_boolean(
    "use_alphalink",
    False,
    "Whether alphalink models are going to be used. Default it False",
)
flags.DEFINE_string("crosslinks", None, "Path to crosslink information pickle")
flags.DEFINE_string(
    "alphalink_weight", None, "Path to AlphaLink neural network weights"
)
flags.DEFINE_string("unifold_param", None,
                    "Path to UniFold neural network weights")
flags.DEFINE_boolean(
    "compress_result_pickles",
    False,
    "Whether the result pickles are going to be gzipped. Default False",
)
flags.DEFINE_boolean(
    "remove_result_pickles",
    False,
    "Whether the result pickles that do not belong to the best model are going to be removed. Default is False",
)
flags.DEFINE_string(
    "description_file",
    None,
    "Path to the text file with multimeric template instructions",
)
flags.DEFINE_string(
    "path_to_mmt", None, "Path to directory with multimeric template mmCIF files"
)
flags.DEFINE_string(
    "protein_delimiter", "+", "Delimiter that separate different prediction jobs. Default is +"
)
flags.DEFINE_integer(
    "desired_num_res", None, "A desired number of residues to pad"
)
flags.DEFINE_integer(
    "desired_num_msa", None, "A desired number of residues to pad"
)
flags.DEFINE_enum(
    "unifold_model_name",
    "multimer_af2",
    [
        "multimer_af2",
        "multimer_ft",
        "multimer",
        "multimer_af2_v3",
        "multimer_af2_model45_v3",
    ],
    "choose unifold model structure",
)
flags.mark_flag_as_required("output_path")

delattr(flags.FLAGS, "models_to_relax")
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
flags.DEFINE_boolean(
    "use_ap_style",
    True,
    "Whether to use multimeric object's description to create output folder"
    "Remember to turn it off if you are using snakemake"
)

unused_flags = (
    "bfd_database_path",
    "db_preset",
    "fasta_paths",
    "hhblits_binary_path",
    "hhsearch_binary_path",
    "hmmbuild_binary_path",
    "hmmsearch_binary_path",
    "jackhmmer_binary_path",
    "kalign_binary_path",
    "max_template_date",
    "mgnify_database_path",
    "num_multimer_predictions_per_model",
    "obsolete_pdbs_path",
    "output_dir",
    "pdb70_database_path",
    "pdb_seqres_database_path",
    "small_bfd_database_path",
    "template_mmcif_dir",
    "uniprot_database_path",
    "uniref30_database_path",
    "uniref90_database_path",
)

for flag in unused_flags:
    delattr(flags.FLAGS, flag)

FLAGS = flags.FLAGS


def main(argv):
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
    _ = process_files(input_files=protein_lists, output_path=buffer)
    buffer.seek(0)
    all_folds = buffer.readlines()
    all_folds = [x.strip().replace(",", ":") for x in all_folds]
    all_folds = [x.strip().replace(";", "+") for x in all_folds]

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
        "--no_pair_msa": FLAGS.no_pair_msa,
        "--msa_depth_scan": FLAGS.msa_depth_scan,
        "--multimeric_template": FLAGS.multimeric_mode,
        "--model_names": FLAGS.model_names,
        "--msa_depth": FLAGS.msa_depth,
        "--crosslinks": FLAGS.crosslinks,
        "--fold_backend": fold_backend,
        "--description_file": FLAGS.description_file,
        "--path_to_mmt": FLAGS.path_to_mmt,
        "--compress_result_pickles": FLAGS.compress_result_pickles,
        "--remove_result_pickles": FLAGS.remove_result_pickles,
        "--use_ap_style": FLAGS.use_ap_style,
        "--use_gpu_relax": FLAGS.use_gpu_relax,
        "--protein_delimiter": FLAGS.protein_delimiter,
        "--desired_num_res": FLAGS.desired_num_res,
        "--desired_num_msa": FLAGS.desired_num_msa
    }

    command_args = {}
    for k, v in constant_args.items():
        if v is None or v is False:
            continue
        elif v is True:
            command_args[k] = ""
        elif isinstance(v, list):
            command_args[k] = " ".join([str(x) for x in v])
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
        command_args["--input"] = " ".join(all_folds)
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
