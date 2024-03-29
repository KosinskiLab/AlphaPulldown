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
from alphapulldown.objects import MonomericObject, ChoppedObject
from alphafold.model.tf.data_transforms import make_fixed_size
from os.path import exists,join
from alphapulldown.objects import ChoppedObject
from alphapulldown.utils.file_handling import make_dir_monomer_dictionary
from ml_collections import ConfigDict
from absl import logging
import tensorflow as tf
logging.set_verbosity(logging.INFO)

def parse_fold(args):
    all_folding_jobs = []
    for i in args.input:
        formatted_folds, missing_features, unique_features = [], [], []
        protein_folds = [x.split(":") for x in i.split(args.protein_delimiter)]
        for protein_fold in protein_folds:
            name, number, region = None, 1, "all"

            match len(protein_fold):
                case 1:
                    name = protein_fold[0]
                case 2:
                    name, number = protein_fold[0], protein_fold[1]
                    if ("-") in protein_fold[1]:
                        number = 1
                        region = protein_fold[1].split("-")
                case 3:
                    name, number, region = protein_fold
            
            number = int(number)
            if len(region) != 2 and region != "all":
                raise ValueError(f"Region {region} is malformatted expected start-stop.")

            if len(region) == 2:
                region = [tuple(int(x) for x in region)]

            unique_features.append(name)
            if not any([exists(join(monomer_dir, f"{name}.pkl")) for monomer_dir in args.features_directory]):
                missing_features.append(name)

            formatted_folds.extend([{name: region} for _ in range(number)])
        all_folding_jobs.append(formatted_folds)
        missing_features = set(missing_features)
        if len(missing_features):
            raise FileNotFoundError(
                f"{missing_features} not found in {args.features_directory}"
            )
    args.parsed_input = all_folding_jobs
    return args

def update_muiltimer_model_config(multimer_model_config : ConfigDict) -> None:
    """
    A function that update multimer model based on the schema from 
    monomer models config before padding 

    Args:
        model_config: a ConfigDict from alphafold.model.config
    """
    model_config = config.model_config("model_1_ptm")
    multimer_model_config.update({'eval': model_config.data.eval}) # added eval to multimer config

    # below update multimer-specific settings
    ONE_DIMENTIONAL_PADDINGS = ['asym_id','sym_id','entity_id','entity_mask','deletion_mean','num_alignments']
    multimer_model_config['eval']['feat'].update({"msa":multimer_model_config['eval']['feat']['msa_feat'][0:2],
                                                  "template_all_atom_mask":multimer_model_config['eval']['feat']['template_all_atom_masks'],
                                                  "deletion_matrix":multimer_model_config['eval']['feat']['msa_feat'][0:2],
                                                  "cluster_bias_mask":multimer_model_config['eval']['feat']['msa_feat'][0:1]})
    
    for k in ONE_DIMENTIONAL_PADDINGS:
        multimer_model_config['eval']['feat'].update({k:multimer_model_config['eval']['feat']['residue_index']})

def pad_input_features(model_config : ConfigDict, feature_dict: dict, 
                       desired_num_res : int, desired_num_msa : int) -> None:
    
    """
    A function that pads input feature numpy arrays based on desired number of residues 
    and desired number of msas 

    Args:
        model_config: ConfigDict from alphafold.model.config
        feature_dict : feature_dict attribute from either a MonomericObject or a MultimericObject
        desired_num_res: desired number of residues 
        desired_num_msa: desired number of msa 
    """
    NUM_EXTRA_MSA = 2048
    NUM_TEMPLATES = 4
    make_fixed_size_fn = make_fixed_size(model_config.eval.feat, 
                                         desired_num_msa,NUM_EXTRA_MSA,
                                         desired_num_res,NUM_TEMPLATES)
    
    assembly_num_chains = feature_dict.pop('assembly_num_chains')
    num_templates = feature_dict.pop('num_templates')
    make_fixed_size_fn(feature_dict)
    # make sure all matrices are numpy ndarray otherwise throught dtype errors
    for k,v in feature_dict.items():
        if isinstance(v, tf.Tensor):
            feature_dict[k] = v.numpy()
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
    return (feature_dict["template_all_atom_masks"].size == 0) or (
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