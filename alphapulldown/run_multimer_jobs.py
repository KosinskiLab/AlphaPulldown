#!/usr/bin/env python3

# Author: Dingquan Yu
# A script to create region information for create_multimer_features.py
# #

import itertools
from re import I
from absl import app, flags, logging
from alphapulldown.utils import *
from itertools import combinations
from alphapulldown.objects import MultimericObject
import os
import shutil
from pathlib import Path
from alphapulldown.predict_structure import predict


flags.DEFINE_enum(
    "mode",
    "pulldown",
    ["pulldown", "all_vs_all", "homo-oligomer", "custom"],
    "choose the mode of running multimer jobs",
)
flags.DEFINE_string(
    "output_path", None, "output directory where the region data is going to be stored"
)
flags.DEFINE_string("oligomer_state_file", None, "path to oligomer state files")
flags.DEFINE_list(
    "monomer_objects_dir",
    None,
    "a list of directories where monomer objects are stored",
)
flags.DEFINE_list("protein_lists", None, "protein list files")
flags.DEFINE_string("data_dir", None, "Path to params directory")
flags.DEFINE_boolean(
    "random_seed",
    False,
    "Run multiple JAX model evaluations "
    "to obtain a timing that excludes the compilation time, "
    "which should be more indicative of the time required for "
    "inferencing many proteins.",
)
flags.DEFINE_integer("num_cycle", 3, help="number of recycles")
flags.DEFINE_boolean(
    "amber_relax",
    False,
    "Run multiple JAX model evaluations "
    "to obtain a timing that excludes the compilation time, "
    "which should be more indicative of the time required for "
    "inferencing many proteins.",
)
flags.DEFINE_boolean(
    "benchmark",
    False,
    "Run multiple JAX model evaluations "
    "to obtain a timing that excludes the compilation time, "
    "which should be more indicative of the time required for "
    "inferencing many proteins.",
)
flags.DEFINE_enum(
    "model_preset",
    "multimer",
    ["monomer", "monomer_casp14", "monomer_ptm", "multimer"],
    "Choose preset model configuration - the monomer model, "
    "the monomer model with extra ensembling, monomer model with "
    "pTM head, or multimer model",
)
flags.DEFINE_integer(
    "num_predictions_per_model", 1, "How many predictions per model. Default is 1"
)
flags.DEFINE_integer(
    "job_index", None, "index of sequence in the fasta file, starting from 1"
)
flags.mark_flag_as_required("output_path")
FLAGS = flags.FLAGS


def create_pulldown_info(
    bait_proteins: list, candidate_proteins: list, job_index=None
) -> dict:
    """
    A function to create apms info

    Args:
    all_proteins: list of all proteins in the fasta file parsed by read_all_proteins()
    bait_protein: name of the bait protein
    job_index: whether there is a job_index specified or not
    """
    all_protein_pairs = list(itertools.product(*[bait_proteins, *candidate_proteins]))
    num_cols = len(candidate_proteins) + 1
    data = dict()

    if job_index is None:
        for i in range(num_cols):
            curr_col = []
            for pair in all_protein_pairs:
                curr_col.append(pair[i])
            update_dict = {f"col_{i+1}": curr_col}
            data.update(update_dict)

    elif isinstance(job_index, int):
        target_pair = all_protein_pairs[job_index-1]
        for i in range(num_cols):
            update_dict = {f"col_{i+1}": [target_pair[i]]}
            data.update(update_dict)
    return data


def create_all_vs_all_info(all_proteins: list, job_index=None):
    """A function to create all against all i.e. every possible pair of interaction"""
    all_possible_pairs = list(combinations(all_proteins, 2))
    if job_index is not None:
        job_index = job_index - 1
        combs = [all_possible_pairs[job_index-1]]
    else:
        combs = all_possible_pairs

    col1 = []
    col2 = []
    for comb in combs:
        col1.append(comb[0])
        col2.append(comb[1])

    data = {"col1": col1, "col2": col2}
    return data


def create_custom_info(all_proteins):
    """
    A function to create 'data' for custom input file
    """
    num_cols = len(all_proteins)
    data = dict()
    for i in range(num_cols):
        data[f"col_{i+1}"] = [all_proteins[i]]
    return data


def create_multimer_objects(data, monomer_objects_dir):
    """
    A function to create multimer objects

    Arg
    data: a dictionary created by create_all_vs_all_info() or create_apms_info()
    monomer_objects_dir: a directory where pre-computed monomer objects are stored
    """

    multimers = []
    num_jobs = len(data[list(data.keys())[0]])
    job_idxes = list(range(num_jobs))

    for job_idx in job_idxes:
        interactors = create_interactors(data, monomer_objects_dir, job_idx)
        if len(interactors) > 1:
            multimer = MultimericObject(interactors=interactors)
            multimer.create_all_chain_features()
            logging.info(f"done creating multimer {multimer.description}")
            multimers.append(multimer)
        else:
            logging.info(f"done loading monomer {interactors[0].description}")
            multimers.append(interactors[0])

    return multimers


def create_homooligomers(oligomer_state_file, monomer_objects_dir, job_index=None):
    """a function to read homooligomer state"""
    multimers = []
    monomer_dir_dict = make_dir_monomer_dictionary(monomer_objects_dir)
    with open(oligomer_state_file) as f:
        lines = list(f.readlines())
        if job_index is not None:
            job_idxes = [job_index - 1]
        else:
            job_idxes = list(range(len(lines)))

        for job_idx in job_idxes:
            l = lines[job_idx]
            if len(l.strip()) > 0:
                if len(l.rstrip().split(",")) > 1:
                    protein_name = l.rstrip().split(",")[0]
                    num_units = int(l.rstrip().split(",")[1])
                else:
                    protein_name = l.rstrip().split(",")[0]
                    num_units = 1

                if num_units > 1:
                    monomer = load_monomer_objects(monomer_dir_dict, protein_name)
                    interactors = [monomer] * num_units
                    homooligomer = MultimericObject(interactors)
                    homooligomer.description = f"{protein_name}_homo_{num_units}er"
                    homooligomer.create_all_chain_features()
                    multimers.append(homooligomer)
                    logging.info(
                        f"finished creating homooligomer {homooligomer.description}"
                    )
                elif num_units == 1:
                    monomer = load_monomer_objects(monomer_dir_dict, protein_name)
                    multimers.append(monomer)
                    logging.info(f"finished loading monomer: {protein_name}")
        f.close()
    return multimers


def create_custom_jobs(custom_input_file, monomer_objects_dir, job_index=None):
    """
    A function to create multimers under custom mode

    Args
    custom_input_file: A list of input_files from FLAGS.protein_lists

    """
    lines = []
    for file in custom_input_file:
        with open(file) as f:
            lines = lines + list(f.readlines())
            f.close()
    if job_index is not None:
        job_idxes = [job_index - 1]
    else:
        job_idxes = list(range(len(lines)))

    for job_idx in job_idxes:
        l = lines[job_idx]
        if len(l.strip()) > 0:
            all_proteins = read_custom(l)
            data = create_custom_info(all_proteins)
            multimers = create_multimer_objects(data, monomer_objects_dir)
    return multimers


def predict_individual_jobs(multimer_object, output_path, model_runners, random_seed):
    output_path = os.path.join(output_path, multimer_object.description)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"now running prediction on {multimer_object.description}")

    if not isinstance(multimer_object, MultimericObject):
        multimer_object.input_seqs = [multimer_object.sequence]

    predict(
        model_runners,
        output_path,
        multimer_object.feature_dict,
        random_seed,
        FLAGS.benchmark,
        fasta_name=multimer_object.description,
        amber_relaxer=FLAGS.amber_relax,
        seqs=multimer_object.input_seqs,
    )
    create_and_save_pae_plots(multimer_object, output_path)


def predict_multimers(multimers):
    """
    Final function to predict multimers

    Args
    multimers: A list of multimer objects created by create_multimer_objects()
    or create_custom_jobs() or create_homooligomers()
    """
    for object in multimers:
        if isinstance(object, MultimericObject):
            model_runners, random_seed = create_model_runners_and_random_seed(
                "multimer",
                FLAGS.num_cycle,
                FLAGS.random_seed,
                FLAGS.data_dir,
                FLAGS.num_predictions_per_model,
            )
            predict_individual_jobs(
                object,
                FLAGS.output_path,
                model_runners=model_runners,
                random_seed=random_seed,
            )
        else:
            model_runners, random_seed = create_model_runners_and_random_seed(
                "monomer_ptm",
                FLAGS.num_cycle,
                FLAGS.random_seed,
                FLAGS.data_dir,
                FLAGS.num_predictions_per_model,
            )
            logging.info("will run in monomer mode")
            predict_individual_jobs(
                object,
                FLAGS.output_path,
                model_runners=model_runners,
                random_seed=random_seed,
            )


def main(argv):
    check_output_dir(FLAGS.output_path)

    if FLAGS.mode == "pulldown":
        bait_proteins = read_all_proteins(FLAGS.protein_lists[0])
        candidate_proteins = []
        for file in FLAGS.protein_lists[1:]:
            candidate_proteins.append(read_all_proteins(file))
        data = create_pulldown_info(
            bait_proteins, candidate_proteins, job_index=FLAGS.job_index
        )
        multimers = create_multimer_objects(data, FLAGS.monomer_objects_dir)

    elif FLAGS.mode == "all_vs_all":
        all_proteins = read_all_proteins(FLAGS.protein_lists[0])
        data = create_all_vs_all_info(all_proteins, job_index=FLAGS.job_index)
        multimers = create_multimer_objects(data, FLAGS.monomer_objects_dir)

    elif FLAGS.mode == "homo-oligomer":
        multimers = create_homooligomers(
            FLAGS.oligomer_state_file,
            FLAGS.monomer_objects_dir,
            job_index=FLAGS.job_index,
        )

    elif FLAGS.mode == "custom":
        multimers = create_custom_jobs(
            FLAGS.protein_lists, FLAGS.monomer_objects_dir, job_index=FLAGS.job_index
        )

    predict_multimers(multimers)


if __name__ == "__main__":
    app.run(main)
