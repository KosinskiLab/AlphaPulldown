#
# Author: Dingquan Yu
# A script containing utility functions
# #

from alphafold.data.tools import jackhmmer
from alphapulldown.objects import ChoppedObject
import json
import os
import pickle
import logging
from alphapulldown.plot_pae import plot_pae
from alphafold.model import config
from alphafold.model import model
from alphafold.model import data
import random
import sys
from alphafold.data import parsers
from pathlib import Path


def create_uniprot_runner(jackhmmer_binary_path, uniprot_database_path):
    """create a uniprot runner object"""
    return jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path, database_path=uniprot_database_path
    )


def make_dir_monomer_dictionary(monomer_objects_dir):
    """
    a function to gather all monomers across different monomer_objects_dir

    args
    monomer_objects_dir: a list of directories where monomer objects are stored, given by FLAGS.monomer_objects_dir
    """
    output_dict = dict()
    for dir in monomer_objects_dir:
        monomers = os.listdir(dir)
        for m in monomers:
            output_dict[m] = dir
    return output_dict


def load_monomer_objects(monomer_dir_dict, protein_name):
    """
    a function to load monomer an object from its pickle

    args
    monomer_dir_dict: a dictionary recording protein_name and its directory. created by make_dir_monomer_dictionary()
    """
    target_path = monomer_dir_dict[f"{protein_name}.pkl"]
    target_path = os.path.join(target_path, f"{protein_name}.pkl")
    return pickle.load(open(target_path, "rb"))


def read_all_proteins(fasta_path) -> list:
    """
    A function to read all proteins in the file

    Args:
    fasta_path: path to the fasta file where all proteins are in one file
    """
    all_proteins = []
    with open(fasta_path, "r") as f:
        lines = list(f.readlines())
        if any(l.startswith(">") for l in lines):
            # this mean the file is a fasta file
            with open(fasta_path, "r") as input_file:
                sequences, descriptions = parsers.parse_fasta(input_file.read())
                for desc in descriptions:
                    all_proteins.append({desc: "all"})
        else:
            for l in lines:
                if len(l.strip()) > 0:
                    curr_list = l.rstrip().split(",")
                    if len(curr_list) == 1:
                        all_proteins.append({l.rstrip().split(",")[0]: "all"})

                    elif len(curr_list) > 1:
                        protein_name = curr_list[0]
                        regions = curr_list[1:]
                        output_region = []
                        for r in regions:
                            output_region.append(
                                (int(r.split("-")[0]), int(r.split("-")[1]))
                            )
                        all_proteins.append({protein_name: output_region})
    return all_proteins


def obtain_region(input_string):
    """
    A function that extract regions from the input string

    Args
    input_string: format is 'protein_n,1-100,2-200'
    or 'protein_n'
    """
    curr_list = input_string.split(",")
    if len(curr_list) == 1:
        return {input_string.rstrip().split(",")[0]: "all"}

    elif len(curr_list) > 1:
        protein_name = curr_list[0]
        regions = curr_list[1:]
        output_region = []
        for r in regions:
            output_region.append((int(r.split("-")[0]), int(r.split("-")[1])))
        return {protein_name: output_region}


def read_custom(line) -> list:
    """
    A function to input file under the mode: custom

    Args:
    line: each individual line in the custom input file
    """
    all_proteins = []
    curr_list = line.rstrip().split(";")
    for substring in curr_list:
        curr_protein = obtain_region(substring)
        all_proteins.append(curr_protein)

    return all_proteins


def check_existing_objects(output_dir, pickle_name):
    """check whether the wanted monomer object already exists in the output_dir"""
    logging.info(f"checking if {os.path.join(output_dir,pickle_name)} already exists")
    return os.path.isfile(os.path.join(output_dir, pickle_name))


def create_interactors(data, monomer_objects_dir, i):
    """
    A function to a list of monomer objects

    Args
    data: a dictionary object storing interactors' names and regions
    """
    interactors = []
    monomer_dir_dict = make_dir_monomer_dictionary(monomer_objects_dir)
    for k in data.keys():
        for curr_interactor_name, curr_interactor_region in data[k][i].items():
            if curr_interactor_region == "all":
                monomer = load_monomer_objects(monomer_dir_dict, curr_interactor_name)
                interactors.append(monomer)
            elif (
                isinstance(curr_interactor_region, list)
                and len(curr_interactor_region) != 0
            ):
                monomer = load_monomer_objects(monomer_dir_dict, curr_interactor_name)
                chopped_object = ChoppedObject(
                    monomer.description,
                    monomer.sequence,
                    monomer.feature_dict,
                    curr_interactor_region,
                )
                chopped_object.prepare_final_sliced_feature_dict()
                interactors.append(chopped_object)
    return interactors


def check_output_dir(path):
    """
    A function to automatically the output directory provided by the user
    if the user hasn't already created the directory
    """
    logging.info(f"checking if output_dir exists {path}")
    if not os.path.isdir(path):
        Path(path).mkdir(parents=True, exist_ok=True)


def create_and_save_pae_plots(multimer_object, output_dir):
    """A function to produce pae plots"""
    ranking_path = os.path.join(output_dir, "ranking_debug.json")
    if not os.path.isfile(ranking_path):
        logging.info(
            "Predictions have failed. please check standard error and output and run again."
        )
    else:
        order = json.load(open(ranking_path, "r"))["order"]
        plot_pae(
            multimer_object.input_seqs, order, output_dir, multimer_object.description
        )


def create_model_runners_and_random_seed(
    model_preset, num_cycle, random_seed, data_dir, num_multimer_predictions_per_model
):
    num_ensemble = 1
    model_runners = {}
    model_names = config.MODEL_PRESETS[model_preset]
    for model_name in model_names:
        model_config = config.model_config(model_name)
        model_config.model.num_ensemble_eval = num_ensemble
        model_config["model"].update({"num_recycle": num_cycle})
        model_config.model.num_ensemble_eval = num_ensemble
        model_params = data.get_model_haiku_params(
            model_name=model_name, data_dir=data_dir
        )
        model_runner = model.RunModel(model_config, model_params)
        for i in range(num_multimer_predictions_per_model):
            model_runners[f"{model_name}_pred_{i}"] = model_runner
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize // len(model_names))
        logging.info("Using random seed %d for the data pipeline", random_seed)
    return model_runners, random_seed


def save_meta_data(flag_dict, outfile):
    """A function to print out metadata"""
    with open(outfile, "w") as f:
        # if shutil.which('git') is not None:
        #     label = subprocess.check_output(["git", "describe", '--always']).decode('utf-8').rstrip()
        #     print(f"git_label:{label}",file=f)

        for k, v in flag_dict.items():
            print(f"{k}:{v}", file=f)


def parse_fasta(fasta_string: str):
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.


    Note:
      This function was built upon alhpafold.data.parsers.parse_fasta in order
      to accomodamte naming convention in this package.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            line.replace(" ", "_")
            unwanted_symbols = ["|", "=", "&", "*", "@", "#", "`", ":", ";", "$", "?"]
            for symbol in unwanted_symbols:
                if symbol in line:
                    line.replace(symbol, "_")
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions
