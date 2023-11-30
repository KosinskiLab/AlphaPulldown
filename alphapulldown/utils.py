#
# Author: Dingquan Yu
# A script containing utility functions
# #

from alphafold.data.tools import jackhmmer
from alphapulldown.objects import ChoppedObject
from alphapulldown import __version__ as AP_VERSION
from alphafold.version import __version__ as AF_VERSION
import json
import os
import pickle
import logging
from alphapulldown.plot_pae import plot_pae
import alphafold
from alphafold.model import config
from alphafold.model import model
from alphafold.model import data
from alphafold.data import templates
import random
import subprocess
from alphafold.data import parsers
from pathlib import Path
import numpy as np
import sys
import datetime
import re
import hashlib
import glob
import importlib.util

COMMON_PATTERNS = [
    r"[Vv]ersion\s*(\d+\.\d+(?:\.\d+)?)",  # version 1.0 or version 1.0.0
    r"\b(\d+\.\d+(?:\.\d+)?)\b"  # just the version number 1.0 or 1.0.0
]
BFD_HASH_HHM_FFINDEX = "799f308b20627088129847709f1abed6"

DB_NAME_TO_URL = {
    'UniRef90' : ["ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz"],
    'UniRef30' : ["https://storage.googleapis.com/alphafold-databases/v2.3/UniRef30_{release_date}.tar.gz"],
    'MGnify' : ["https://storage.googleapis.com/alphafold-databases/v2.3/mgy_clusters_{release_date}.fa.gz"],
    'BFD' : ["https://storage.googleapis.com/alphafold-databases/casp14_versions/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz"],
    'Reduced BFD' : ["https://storage.googleapis.com/alphafold-databases/reduced_dbs/bfd-first_non_consensus_sequences.fasta.gz"],
    'PDB70' : ["http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz"],
    'UniProt' : [
        "ftp://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz",
        "ftp://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
        ],
    'PDB seqres' : ["ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt"],
    'ColabFold' : ["https://wwwuser.gwdg.de/~compbiol/colabfold/colabfold_envdb_202108.tar.gz"],
}

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

def post_prediction_process(output_path,zip_pickles = False,remove_pickles = False):
    """A function to process resulted files after the prediction"""
    if remove_pickles:
        remove_irrelavent_pickles(output_path)
    if zip_pickles:
        zip_result_pickles(output_path)

def remove_irrelavent_pickles(output_path):
    """Remove result pickles that do not belong to the best model"""
    try:
        best_model = json.load(open(os.path.join(output_path,"ranking_debug.json"),'rb'))['order'][0]
        pickle_to_remove = [i for i in os.listdir(output_path) if (i.endswith('pkl')) and (best_model not in i)]
        cmd = ['rm'] + pickle_to_remove
        results = subprocess.run(cmd)
    except FileNotFoundError:
        print(f"ranking_debug.json does not exist in : {output_path}. Please check your inputs.")
    except subprocess.CalledProcessError as e:
        print(f"Error while removing result pickles: {e.returncode}")
        print(f"Command output: {e.output}")      

def zip_result_pickles(output_path):
    """A function that remove results pickles in the output directory"""
    cmd = f"cd {output_path} && gzip --force --verbose *.pkl"
    try:
        results = subprocess.run(cmd,shell=True,capture_output=True,text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while compressing result pickles: {e.returncode}")
        print(f"Command output: {e.output}")

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


def get_last_modified_date(path):
    """
    Get the last modified date of a file or the most recently modified file in a directory.
    """
    try:
        if not os.path.exists(path):
            logging.warning(f"Path does not exist: {path}")
            return None

        if os.path.isfile(path):
            return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')

        logging.info(f"Getting last modified date for {path}")
        most_recent_timestamp = max((entry.stat().st_mtime for entry in glob.glob(path + '*') if entry.is_file()),
                                    default=0.0)

        return datetime.datetime.fromtimestamp(most_recent_timestamp).strftime(
            '%Y-%m-%d %H:%M:%S') if most_recent_timestamp else None

    except Exception as e:
        logging.warning(f"Error processing {path}: {e}")
        return None


def parse_version(output):
    """Parse version information from a given output string."""
    for pattern in COMMON_PATTERNS:
        match = re.search(pattern, output)
        if match:
            return match.group(1)

    match = re.search(r"Kalign\s+version\s+(\d+\.\d+)", output)
    if match:
        return match.group(1)

    return None


def get_hash(filename):
    """Get the md5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(filename, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
        return (md5_hash.hexdigest())


def get_program_version(binary_path):
    """Get version information for a given binary."""
    for cmd_suffix in ["--help", "-h"]:
        cmd = [binary_path, cmd_suffix]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            version = parse_version(result.stdout + result.stderr)
            if version:
                return version
        except Exception as e:
            logging.debug(f"Error while processing {cmd}: {e}")

    logging.warning(f"Cannot parse version from {binary_path}")
    return None


def get_metadata_for_binary(k, v):
    name = k.replace("_binary_path", "")
    return {name: {"version": get_program_version(v)}}


def get_metadata_for_database(k, v):
    name = k.replace("_database_path", "").replace("_dir", "")

    specific_databases = ["pdb70", "bfd"]
    if name in specific_databases:
        name = name.upper()
        url = DB_NAME_TO_URL[name]
        fn = v + "_hhm.ffindex"
        hash_value = get_hash(fn)
        release_date = get_last_modified_date(fn)
        if release_date == "NA":
            release_date = None
        if hash_value == BFD_HASH_HHM_FFINDEX:
            release_date = "AF2"
        return {name: {"release_date": release_date, "version": hash_value, "location_url": url}}

    other_databases = ["small_bfd", "uniprot", "uniref90", "pdb_seqres"]
    if name in other_databases:
        if name == "small_bfd":
            name = "Reduced BFD"
        elif name == "uniprot":
            name = "UniProt"
        elif name == "uniref90":
            name = "UniRef90"
        elif name == "pdb_seqres":
            name = "PDB seqres"
        url = DB_NAME_TO_URL[name]
        # here we ignore pdb_mmcif assuming it's version is identical to pdb_seqres
        return {name: {"release_date": get_last_modified_date(v),
                       "version": None if name != "PDB seqres" else get_hash(v), "location_url": url}}

    if name in ["uniref30", "mgnify"]:
        if name == "uniref30":
            name = "UniRef30"
        elif name == "mgnify":
            name = "MGnify"
        hash_value = None
        release_date = None
        match = re.search(r"(\d{4}_\d{2})", v)
        if match:
            #release_date = match.group(1)
            url_release_date = match.group(1)
            url = [DB_NAME_TO_URL[name][0].format(release_date=url_release_date)]
            if name == "UniRef30":
                hash_value = get_hash(v + "_hhm.ffindex")
                if not hash_value:
                    hash_value = url_release_date
            if name == "MGnify":
                hash_value = url_release_date
        return {name: {"release_date": release_date, "version": hash_value, "location_url": url}}
    return {}


def save_meta_data(flag_dict, outfile):
    """Save metadata in JSON format."""
    metadata = {
        "databases": {},
        "software": {"AlphaPulldown": {"version": AP_VERSION},
                     "AlphaFold": {"version": AF_VERSION}},
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "other": {},
    }

    for k, v in flag_dict.items():
        if v is None:
            continue
        if k == "use_cprofile_for_profiling" or k.startswith("test_") or k.startswith("help"):
            continue
        metadata["other"][k] = str(v)
        if "_binary_path" in k:
            metadata["software"].update(get_metadata_for_binary(k, v))
        elif "_database_path" in k or "template_mmcif_dir" in k:
            metadata["databases"].update(get_metadata_for_database(k, v))
        elif k == "use_mmseqs2":
            url = DB_NAME_TO_URL["ColabFold"]
            metadata["databases"].update({"ColabFold":
                                              {"version": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                               "release_date": None,
                                               "location_url": url}
                                          })

    with open(outfile, "w") as f:
        json.dump(metadata, f, indent=2)


def convert_fasta_description_to_protein_name(line):
    line = line.replace(" ", "_")
    unwanted_symbols = ["|", "=", "&", "*", "@", "#", "`", ":", ";", "$", "?"]
    for symbol in unwanted_symbols:
        if symbol in line:
            line = line.replace(symbol, "_")
    return line[1:]  # Remove the '>' at the beginning.


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
            descriptions.append(convert_fasta_description_to_protein_name(line))
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions
