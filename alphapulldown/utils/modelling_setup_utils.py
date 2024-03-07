#
# Author: Dingquan Yu
# A script containing utility functions
# #
from alphafold.data.tools import jackhmmer
from alphapulldown.core.objects import ChoppedObject
import os
import pickle
import logging
import alphafold
from alphafold.model import config
from alphafold.model import model
from alphafold.model import data
from alphafold.data import templates
import random
from alphafold.data import parsers
from pathlib import Path
import numpy as np
import sys
import importlib.util
from alphapulldown.utils.file_handling_utils import make_dir_monomer_dictionary

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