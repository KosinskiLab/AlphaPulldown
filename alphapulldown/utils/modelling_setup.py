#
# Author: Dingquan Yu
# A script containing utility functions
# 

import os
import sys
import pickle
import lzma
import importlib.util
from typing import List,Dict,Union
import numpy as np
import alphafold
from alphafold.data.tools import jackhmmer
from alphafold.data import templates
from alphapulldown.objects import MonomericObject
from os.path import exists,join
from alphapulldown.objects import ChoppedObject
from alphapulldown.utils.file_handling import make_dir_monomer_dictionary
from absl import logging
logging.set_verbosity(logging.INFO)


def parse_fold(input_list, features_directory, protein_delimiter):
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
    missing_features = set()  # Initialize as a set to collect unique missing features
    for i in input_list:
        formatted_folds = []
        protein_folds = [x.split(":") for x in i.split(protein_delimiter)]
        for protein_fold in protein_folds:
            name, number, region = None, 1, "all"

            if len(protein_fold) == 1:
                # Format: [protein_name]
                name = protein_fold[0]
            elif len(protein_fold) > 1:
                name = protein_fold[0]
                if "-" in protein_fold[1]:
                    # Format: [protein_name:1-10:14-30:40-100:etc]
                    try:
                        number = 1
                        region = protein_fold[1:]
                        region = [tuple(int(x) for x in r.split("-")) for r in region]
                    except Exception:
                        logging.error(f"Your format: {i} is wrong. The program will terminate.")
                        sys.exit()
                else:
                    # Format: [protein_name:copy_number:1-10:14-30:40-100:etc]
                    try:
                        number = int(protein_fold[1])
                        if len(protein_fold) > 2:
                            region = protein_fold[2:]
                            region = [tuple(int(x) for x in r.split("-")) for r in region]
                    except Exception:
                        logging.error(f"Your format: {i} is wrong. The program will terminate.")
                        sys.exit()

            number = int(number)
            # Check for missing features
            if not any(
                exists(join(monomer_dir, f"{name}{ext}"))
                for monomer_dir in features_directory
                for ext in [".pkl", ".pkl.xz"]
            ):
                missing_features.add(name)  # Use .add() since missing_features is a set

            formatted_folds.extend([{name: region} for _ in range(number)])
        all_folding_jobs.append(formatted_folds)

    if missing_features:
        raise FileNotFoundError(
            f"{sorted(missing_features)} not found in {features_directory}"
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
    feature_dict.pop('seq_length')
    feature_dict.pop('num_alignments')
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
    A function to load a monomer object from its pickle file.
    If the file is compressed with .xz, it will decompress it first.

    Args:
    monomer_dir_dict: a dictionary recording protein_name and its directory.
                      Created by make_dir_monomer_dictionary().
    protein_name: the name of the protein to load.

    Returns:
    monomer: the loaded monomer object.
    """
    # Check if the .pkl or .pkl.xz file exists in the dictionary
    if f"{protein_name}.pkl" in monomer_dir_dict:
        target_path = monomer_dir_dict[f"{protein_name}.pkl"]
        target_path = os.path.join(target_path, f"{protein_name}.pkl")
    elif f"{protein_name}.pkl.xz" in monomer_dir_dict:
        target_path = monomer_dir_dict[f"{protein_name}.pkl.xz"]
        target_path = os.path.join(target_path, f"{protein_name}.pkl.xz")
    else:
        raise FileNotFoundError(f"No file found for {protein_name}")

    # Load the monomer object from either the .pkl or .pkl.xz file
    if target_path.endswith('.pkl'):
        with open(target_path, "rb") as f:
            monomer = pickle.load(f)
    elif target_path.endswith('.pkl.xz') :
        with lzma.open(target_path, "rb") as f:
            monomer = pickle.load(f)
    else:
        raise FileNotFoundError(f"Neither .pkl nor .pkl.xz file found for {protein_name}")

    # Check and potentially modify the feature_dict
    if check_empty_templates(monomer.feature_dict):
        monomer.feature_dict = mk_mock_template(monomer.feature_dict)

    return monomer


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

