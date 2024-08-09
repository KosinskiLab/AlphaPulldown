#
# Author: Dingquan Yu
# A script containing utility functions
# 

import os
import sys
import random
import pickle
import logging
import importlib.util
from pathlib import Path
from typing import List,Dict,Union
import numpy as np
import alphafold
from alphafold.data import parsers
from alphafold.data.tools import jackhmmer
from alphafold.model import config
from alphafold.model import model
from alphafold.model import data
from alphafold.data import templates
from alphapulldown.objects import MonomericObject
from os.path import exists,join
from alphapulldown.objects import ChoppedObject
from alphapulldown.utils.file_handling import make_dir_monomer_dictionary
from absl import logging
logging.set_verbosity(logging.INFO)


def parse_fold(input, features_directory, protein_delimiter):
    """
    Parses a list of protein fold specifications and returns structured folding jobs.

    Args:
        input_list (list): List of protein fold specifications as strings.
        features_directory (list): List of directories to search for protein feature files.
        protein_delimiter (str): Delimiter used to separate different protein folds.

    Returns:
        list: A list of folding jobs, each represented by a list of dictionaries.

    Raises:
        FileNotFoundError: If any required protein features are missing.
    """
    all_folding_jobs = []
    for i in input:
        formatted_folds, missing_features, unique_features = [], [], []
        protein_folds = [x.split(":") for x in i.split(protein_delimiter)]
        for protein_fold in protein_folds:
            name, number, region = None, 1, "all"

            if len(protein_fold) ==1:
                # protein_fold is in this format: [protein_name]
                name = protein_fold[0]
            elif len(protein_fold) > 1:
                name, number= protein_fold[0], protein_fold[1]
                if ("-") in protein_fold[1]:
                    # protein_fold is in this format: [protein_name:1-10:14-30:40-100:etc]
                    try:
                        number = 1
                        region = protein_fold[1:]
                        region = [tuple(int(x) for x in r.split("-")) for r in region]
                    except Exception as e:
                        logging.error(f"Your format: {i} is wrong. The programme will terminate.")
                        sys.exit()
                else:
                    # protein_fold is in this format: [protein_name:copy_number:1-10:14-30:40-100:etc]
                    try:
                        number = protein_fold[1]
                        if len(protein_fold[2:]) > 0:
                            region = protein_fold[2:]
                            region = [tuple(int(x) for x in r.split("-")) for r in region]
                    except Exception as e:
                        logging.error(f"Your format: {i} is wrong. The programme will terminate.")
                        sys.exit()
            
            number = int(number)
            unique_features.append(name)
            if not any([exists(join(monomer_dir, f"{name}.pkl")) for monomer_dir in features_directory]):
                missing_features.append(name)

            formatted_folds.extend([{name: region} for _ in range(number)])
        all_folding_jobs.append(formatted_folds)
        missing_features = set(missing_features)
        if len(missing_features):
            raise FileNotFoundError(
                f"{missing_features} not found in {features_directory}"
            )
    return all_folding_jobs

def pad_input_features(feature_dict: dict, 
                       desired_num_res : int, desired_num_msa : int) -> None:
    
    """
    A function that pads input feature numpy arrays based on desired number of residues 
    and desired number of msas 

    Args:
        feature_dict : feature_dict attribute from either a MonomericObject or a MultimericObject
        desired_num_res: desired number of residues 
        desired_num_msa: desired number of msa 
    """
    def pad_individual_matrix(v, axis_indexes, shape, nums_to_add):
        pad_width = [(0,0) for _ in shape]
        for idx, num_to_add in zip(axis_indexes, nums_to_add):
            pad_width[idx] = (0,num_to_add)
        return np.pad(v, pad_width=pad_width)

    assembly_num_chains = feature_dict.pop('assembly_num_chains')
    num_templates = feature_dict.pop('num_templates')
    seq_length = feature_dict.pop('seq_length')
    num_alignments = feature_dict.pop('num_alignments')
    original_num_msa , original_num_res = feature_dict['msa'].shape
    num_res_to_pad = desired_num_res - original_num_res
    num_msa_to_pad = desired_num_msa - original_num_msa

    for k,v in feature_dict.items():
        axies_to_pad = []
        nums_to_pad = []
        if original_num_msa in v.shape:
            msa_axis = v.shape.index(original_num_msa)
            axies_to_pad.append(msa_axis)
            nums_to_pad.append(num_msa_to_pad)
        if original_num_res in v.shape:
            res_axis = v.shape.index(original_num_res)
            nums_to_pad.append(num_res_to_pad)
            axies_to_pad.append(res_axis)
        output = pad_individual_matrix(v, axies_to_pad, v.shape, nums_to_pad)
        feature_dict[k] = output 
    feature_dict['seq_length'] = np.array([desired_num_res])
    feature_dict['num_alignments'] = np.array([desired_num_msa])
    feature_dict['assembly_num_chains'] = assembly_num_chains
    feature_dict['num_templates'] = num_templates

def create_custom_info(all_proteins : List[List[Dict[str, str]]]) -> List[Dict[str, List[str]]]:
    """
    Create a dictionary representation of data for a custom input file.

    Parameters
    ----------
    all_proteins : List[List[Dict[str, str]]]
       A list of lists of protein names or sequences. Each element
       of the list is a nother list of dictionaries thats should be included in the data.

    Returns
    -------
     List[Dict[str, List[str]]]
        A list of dictionaries. Within each dictionary: each key is a column name following the
        pattern 'col_X' where X is the column index starting from 1.
        Each key maps to a list containing a single protein name or
        sequence from the input list.

    """
    output = []
    def process_single_dictionary(all_proteins):
        num_cols = len(all_proteins)
        data = dict()
        for i in range(num_cols):
            data[f"col_{i + 1}"] = [all_proteins[i]]
        return data
    for i in all_proteins:
        curr_data = process_single_dictionary(i)
        output.append(curr_data)
    return output

def get_run_alphafold():
    """
    A function that imports run_alphafold
    """
    def load_module(file_name, module_name):
        if module_name in sys.modules:
            return sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, file_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    # First try
    PATH_TO_RUN_ALPHAFOLD = os.path.join(os.path.dirname(alphafold.__file__), "run_alphafold.py")

    if not os.path.exists(PATH_TO_RUN_ALPHAFOLD):
        # Adjust path if file not found
        PATH_TO_RUN_ALPHAFOLD = os.path.join(os.path.dirname(os.path.dirname(alphafold.__file__)), "run_alphafold.py")

    try:
        run_af = load_module(PATH_TO_RUN_ALPHAFOLD, "run_alphafold")
        return run_af
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find 'run_alphafold.py' at {PATH_TO_RUN_ALPHAFOLD}") from e

def create_uniprot_runner(jackhmmer_binary_path, uniprot_database_path):
    """create a uniprot runner object"""
    return jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path, database_path=uniprot_database_path
    )


def check_empty_templates(feature_dict: dict) -> bool:
    """A function to check wether the pickle has empty templates"""
    if "template_all_atom_masks" in feature_dict:
        return (feature_dict["template_all_atom_masks"].size == 0) or (
                feature_dict["template_aatype"].size == 0
        )
    elif "template_all_atom_mask" in feature_dict:
        return (feature_dict["template_all_atom_mask"].size == 0) or (
                feature_dict["template_aatype"].size == 0
        )


def mk_mock_template(feature_dict: dict):
    """
    Modified based upon colabfold mk_mock_template():
    https://github.com/sokrypton/ColabFold/blob/05c0cb38d002180da3b58cdc53ea45a6b2a62d31/colabfold/batch.py#L121-L155
    """
    num_temp = 1  # number of fake templates
    ln = feature_dict["aatype"].shape[0]
    output_templates_sequence = "A" * ln

    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_domain_names": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    feature_dict.update(template_features)
    return feature_dict


def load_monomer_objects(monomer_dir_dict, protein_name):
    """
    a function to load monomer an object from its pickle

    args
    monomer_dir_dict: a dictionary recording protein_name and its directory. created by make_dir_monomer_dictionary()
    """
    target_path = monomer_dir_dict[f"{protein_name}.pkl"]
    target_path = os.path.join(target_path, f"{protein_name}.pkl")
    monomer = pickle.load(open(target_path, "rb"))
    if check_empty_templates(monomer.feature_dict):
        monomer.feature_dict = mk_mock_template(monomer.feature_dict)
    return monomer


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
    logging.info(f"checking if {os.path.join(output_dir, pickle_name)} already exists")
    return os.path.isfile(os.path.join(output_dir, pickle_name))


def create_interactors(data : List[Dict[str, List[str]]], 
                       monomer_objects_dir : List[str], i : int = 0) -> List[List[Union[MonomericObject, ChoppedObject]]]:
    """
    A function to a list of monomer objects

    Args
    data: a dictionary object storing interactors' names and regions

    Return:
    A list in which each element is a list of MonomericObject/ChoppedObject
    """
    def process_each_dict(data,monomer_objects_dir):
        interactors = []
        monomer_dir_dict = make_dir_monomer_dictionary(monomer_objects_dir)
        for k in data.keys():
            for curr_interactor_name, curr_interactor_region in data[k][i].items():
                monomer = load_monomer_objects(monomer_dir_dict, curr_interactor_name)
                if check_empty_templates(monomer.feature_dict):
                    monomer.feature_dict = mk_mock_template(monomer.feature_dict)
                else:
                    if curr_interactor_region == "all":
                        interactors.append(monomer)
                    elif (
                            isinstance(curr_interactor_region, list)
                            and len(curr_interactor_region) != 0
                    ):
                        chopped_object = ChoppedObject(
                            monomer.description,
                            monomer.sequence,
                            monomer.feature_dict,
                            curr_interactor_region,
                        )
                        chopped_object.prepare_final_sliced_feature_dict()
                        interactors.append(chopped_object)
        return interactors

    interactors = []
    for d in data:
        interactors.append(process_each_dict(d, monomer_objects_dir))
    return interactors


def check_output_dir(path):
    """
    A function to automatically the output directory provided by the user
    if the user hasn't already created the directory
    """
    logging.info(f"checking if output_dir exists {path}")
    if not os.path.isdir(path):
        Path(path).mkdir(parents=True, exist_ok=True)


def compute_msa_ranges(num_msa, num_extra_msa, num_multimer_predictions):
    """
    Denser for smaller num_msa, sparser for larger num_msa
    """
    msa_ranges = np.rint(np.logspace(np.log10(16), np.log10(num_msa),
                                     num_multimer_predictions)).astype(int).tolist()
    extra_msa_ranges = np.rint(np.logspace(np.log10(32), np.log10(num_extra_msa),
                                           num_multimer_predictions)).astype(int).tolist()
    return msa_ranges, extra_msa_ranges


def update_model_config(model_config, num_msa, num_extra_msa):
    embeddings_and_evo = model_config["model"]["embeddings_and_evoformer"]
    embeddings_and_evo.update({"num_msa": num_msa, "num_extra_msa": num_extra_msa})


def create_model_runners_and_random_seed(
        model_preset, num_cycle, random_seed, data_dir,
        num_multimer_predictions_per_model,
        gradient_msa_depth=False, model_names_custom=None,
        msa_depth=None):
    num_ensemble = 1
    model_runners = {}
    model_names = config.MODEL_PRESETS[model_preset]

    if model_names_custom:
        model_names_custom = tuple(model_names_custom.split(","))
        if all(x in model_names for x in model_names_custom):
            model_names = model_names_custom
        else:
            raise Exception(f"Provided model names {model_names_custom} not part of available {model_names}")

    for model_name in model_names:
        model_config = config.model_config(model_name)
        model_config.model.num_ensemble_eval = num_ensemble
        model_config["model"].update({"num_recycle": num_cycle})

        model_params = data.get_model_haiku_params(model_name=model_name, data_dir=data_dir)
        model_runner = model.RunModel(model_config, model_params)

        if gradient_msa_depth or msa_depth:
            num_msa, num_extra_msa = get_default_msa(model_config)
            msa_ranges, extra_msa_ranges = compute_msa_ranges(num_msa, num_extra_msa,
                                                              num_multimer_predictions_per_model)

        for i in range(num_multimer_predictions_per_model):
            if msa_depth or gradient_msa_depth:
                if msa_depth:
                    num_msa = int(msa_depth)
                    num_extra_msa = num_msa * 4  # approx. 4x the number of msa, as in the AF2 config file
                elif gradient_msa_depth:
                    num_msa = msa_ranges[i]
                    num_extra_msa = extra_msa_ranges[i]
                update_model_config(model_config, num_msa, num_extra_msa)
                logging.info(
                    f"Model {model_name} is running {i} prediction with num_msa={num_msa} "
                    f"and num_extra_msa={num_extra_msa}")
                model_runners[f"{model_name}_pred_{i}_msa_{num_msa}"] = model_runner
                #model_runners[f"{model_name}_pred_{i}"] = model_runner
            else:
                logging.info(
                    f"Model {model_name} is running {i} prediction with default MSA depth")
                model_runners[f"{model_name}_pred_{i}"] = model_runner

    if random_seed is None:
        random_seed = random.randrange(sys.maxsize // len(model_runners))
        logging.info("Using random seed %d for the data pipeline", random_seed)

    return model_runners, random_seed


def get_default_msa(model_config):
    embeddings_and_evo = model_config["model"]["embeddings_and_evoformer"]
    return embeddings_and_evo["num_msa"], embeddings_and_evo["num_extra_msa"]
