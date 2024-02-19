#!/usr/bin/env python3
# coding: utf-8
# Create features for AlphaFold from fasta file(s) or a csv file with descriptions for multimeric templates
# #

import contextlib
import csv
import os
import pickle
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from absl import logging, app
from alphafold.data import templates
from alphafold.data.pipeline import DataPipeline
from alphafold.data.tools import hmmsearch
from colabfold.utils import DEFAULT_API_SERVER

from alphapulldown.create_custom_template_db import create_db
from alphapulldown.objects import MonomericObject
from alphapulldown.utils import (
    convert_fasta_description_to_protein_name,
    create_uniprot_runner,
    get_run_alphafold,
    parse_fasta,
    save_meta_data
)

# Initialize and define flags
run_af = get_run_alphafold()
flags = run_af.flags

# All flags
flags.DEFINE_bool("use_mmseqs2", False, "Use mmseqs2 remotely or not. Default is False")
flags.DEFINE_bool("save_msa_files", False, "Save MSA output or not")
flags.DEFINE_bool("skip_existing", False, "Skip existing monomer feature pickles or not")
flags.DEFINE_string("new_uniclust_dir", None, "Directory where new version of uniclust is stored")
flags.DEFINE_integer("seq_index", None, "Index of sequence in the fasta file, starting from 1")

# Flags related to TrueMultimer
flags.DEFINE_string("path_to_mmt", None, "Path to directory with multimeric template mmCIF files")
flags.DEFINE_string("description_file", None, "Path to the text file with descriptions")
flags.DEFINE_float("threshold_clashes", 1000,
                   "Threshold for VDW overlap to identify clashes (default: 1000, i.e. no threshold, for thresholding, use 0.9)")
flags.DEFINE_float("hb_allowance", 0.4, "Allowance for hydrogen bonding (default: 0.4)")
flags.DEFINE_float("plddt_threshold", 0, "Threshold for pLDDT score (default: 0)")

FLAGS = flags.FLAGS
MAX_TEMPLATE_HITS = 20

flags_dict = FLAGS.flag_values_dict()


def ensure_directory_exists(directory):
    """
    Ensures that a directory exists. If the directory does not exist, it is created.

    Args:
    directory (str): The path of the directory to check or create.
    """
    if not os.path.exists(directory):
        logging.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)


@contextlib.contextmanager
def output_meta_file(file_path):
    """
    A context manager that ensures the directory for a file exists and then opens the file for writing.

    Args:
    file_path (str): The path of the file to be opened.

    Yields:
    Generator[str]: The name of the file opened.
    """
    ensure_directory_exists(os.path.dirname(file_path))
    with open(file_path, "w") as outfile:
        yield outfile.name


def get_database_path(flag_value, default_subpath):
    """
    Retrieves the database path based on a flag value or a default subpath.

    Args:
    flag_value (str): The value of the flag specifying the database path.
    default_subpath (str): The default subpath to use if the flag value is not set.

    Returns:
    str: The final path to the database.
    """
    return flag_value or os.path.join(FLAGS.data_dir, default_subpath)


def create_arguments(local_path_to_custom_template_db=None):
    """
    Updates the (global) flags dictionary with paths to various databases required for AlphaFold. If a local path to a
    custom template database is provided, pdb-related paths are set to this local database.

    Args:
    local_path_to_custom_template_db (str, optional): Path to a local custom template database. Defaults to None.
    """
    global use_small_bfd

    FLAGS.uniref30_database_path = get_database_path(FLAGS.uniref30_database_path, "uniref30/UniRef30_2023_02")
    FLAGS.uniref90_database_path = get_database_path(FLAGS.uniref90_database_path, "uniref90/uniref90.fasta")
    FLAGS.mgnify_database_path = get_database_path(FLAGS.mgnify_database_path, "mgnify/mgy_clusters_2022_05.fa")
    FLAGS.bfd_database_path = get_database_path(FLAGS.bfd_database_path,
                                                "bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt")
    FLAGS.small_bfd_database_path = get_database_path(FLAGS.small_bfd_database_path,
                                                      "small_bfd/bfd-first_non_consensus_sequences.fasta")
    FLAGS.pdb70_database_path = get_database_path(FLAGS.pdb70_database_path, "pdb70/pdb70")

    use_small_bfd = FLAGS.db_preset == "reduced_dbs"
    flags_dict.update({"use_small_bfd": use_small_bfd})

    # Update pdb related flags
    if local_path_to_custom_template_db:
        FLAGS.pdb_seqres_database_path = os.path.join(local_path_to_custom_template_db, "pdb_seqres", "pdb_seqres.txt")
        flags_dict.update({"pdb_seqres_database_path": FLAGS.pdb_seqres_database_path})
        FLAGS.template_mmcif_dir = os.path.join(local_path_to_custom_template_db, "pdb_mmcif", "mmcif_files")
        flags_dict.update({"template_mmcif_dir": FLAGS.template_mmcif_dir})
        FLAGS.obsolete_pdbs_path = os.path.join(local_path_to_custom_template_db, "pdb_mmcif", "obsolete.dat")
        flags_dict.update({"obsolete_pdbs_path": FLAGS.obsolete_pdbs_path})
    else:
        FLAGS.pdb_seqres_database_path = get_database_path(FLAGS.pdb_seqres_database_path, "pdb_seqres/pdb_seqres.txt")
        flags_dict.update({"pdb_seqres_database_path": FLAGS.pdb_seqres_database_path})
        FLAGS.template_mmcif_dir = get_database_path(FLAGS.template_mmcif_dir, "pdb_mmcif/mmcif_files")
        flags_dict.update({"template_mmcif_dir": FLAGS.template_mmcif_dir})
        FLAGS.obsolete_pdbs_path = get_database_path(FLAGS.obsolete_pdbs_path, "pdb_mmcif/obsolete.dat")
        flags_dict.update({"obsolete_pdbs_path": FLAGS.obsolete_pdbs_path})


def create_custom_db(temp_dir, protein, templates, chains):
    """
    Creates a custom template database for a specific protein using given templates and chains.

    Args:
    temp_dir (str): The temporary directory to store the custom database.
    protein (str): The name of the protein for which the database is created.
    templates (list): A list of template file paths.
    chains (list): A list of chain identifiers corresponding to the templates.

    Returns:
    Path: The path to the created custom template database.
    """
    threashold_clashes = FLAGS.threshold_clashes
    hb_allowance = FLAGS.hb_allowance
    plddt_threshold = FLAGS.plddt_threshold
    # local_path_to_custom_template_db = Path(".") / "custom_template_db" / protein # DEBUG
    local_path_to_custom_template_db = Path(temp_dir) / "custom_template_db" / protein
    logging.info(f"Path to local database: {local_path_to_custom_template_db}")
    create_db(local_path_to_custom_template_db, templates, chains, threashold_clashes, hb_allowance, plddt_threshold)

    return local_path_to_custom_template_db


def parse_csv_file(csv_path, fasta_paths, mmt_dir):
    """
    csv_path (str): Path to the text file with descriptions
        features.csv: A coma-separated file with three columns: PROTEIN name, PDB/CIF template, chain ID.
    fasta_paths (str): path to fasta file(s)
    mmt_dir (str): Path to directory with multimeric template mmCIF files

    Returns:
        a list of dictionaries with the following structure:
    [{"protein": protein_name, "sequence" :sequence", templates": [pdb_files], "chains": [chain_id]}, ...]}]
    """
    protein_names = {}
    for fasta_path in fasta_paths:
        if not os.path.isfile(fasta_path):
            logging.error(f"Fasta file {fasta_path} does not exist.")
            raise FileNotFoundError(f"Fasta file {fasta_path} does not exist.")
        for curr_seq, curr_desc in iter_seqs(fasta_paths):
            protein_names[curr_desc] = curr_seq

    parsed_dict = {}
    with open(csv_path, newline="") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if not row or len(row) != 3:
                logging.warning(f"Skipping invalid line in {csv_path}: {row}")
                continue
            protein, template, chain = map(str.strip, row)
            protein = convert_fasta_description_to_protein_name(protein)
            if protein not in protein_names:
                logging.error(f"Protein {protein} from description.csv is not found in the fasta files.")
                continue
            parsed_dict.setdefault(protein, {"protein": protein, "templates": [], "chains": [], "sequence": None})
            parsed_dict[protein]["sequence"] = protein_names[protein]
            parsed_dict[protein]["templates"].append(os.path.join(mmt_dir, template))
            parsed_dict[protein]["chains"].append(chain)

    return list(parsed_dict.values())


def create_pipeline():
    """
    Creates and returns a data pipeline for AlphaFold, configured with necessary binary paths and database paths.

    Returns:
    DataPipeline: An instance of the AlphaFold DataPipeline configured with necessary paths.
    """
    monomer_data_pipeline = DataPipeline(
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        hhblits_binary_path=FLAGS.hhblits_binary_path,
        uniref90_database_path=FLAGS.uniref90_database_path,
        mgnify_database_path=FLAGS.mgnify_database_path,
        bfd_database_path=FLAGS.bfd_database_path,
        uniref30_database_path=FLAGS.uniref30_database_path,
        small_bfd_database_path=FLAGS.small_bfd_database_path,
        use_small_bfd=use_small_bfd,
        use_precomputed_msas=FLAGS.use_precomputed_msas,
        template_searcher=hmmsearch.Hmmsearch(
            binary_path=FLAGS.hmmsearch_binary_path,
            hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
            database_path=FLAGS.pdb_seqres_database_path,
        ),
        template_featurizer=templates.HmmsearchHitFeaturizer(
            mmcif_dir=FLAGS.template_mmcif_dir,
            max_template_date=FLAGS.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=FLAGS.kalign_binary_path,
            obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
            release_dates_path=None,
        ),
    )
    return monomer_data_pipeline


def check_existing_objects(output_dir, pickle_name):
    return os.path.isfile(os.path.join(output_dir, pickle_name))


def create_and_save_monomer_objects(m, pipeline):
    """
    Processes a MonomericObject to create and save its features. If the skip_existing flag is set and the
    monomer object already exists as a pickle, the function skips processing.

    Args:
    m (MonomericObject): The monomeric object to be processed.
    pipeline (DataPipeline): The AlphaFold DataPipeline for processing.
    """
    use_mmseqs2 = FLAGS.use_mmseqs2
    if FLAGS.skip_existing and check_existing_objects(
            FLAGS.output_dir, f"{m.description}.pkl"
    ):
        logging.info(f"Already found {m.description}.pkl in {FLAGS.output_dir} Skipped")
        pass
    else:
        metadata_output_path = os.path.join(
            FLAGS.output_dir,
            f"{m.description}_feature_metadata_{datetime.date(datetime.now())}.json",
        )
        with output_meta_file(metadata_output_path) as meta_data_outfile:
            save_meta_data(flags_dict, meta_data_outfile)

        if not use_mmseqs2:
            m.make_features(
                pipeline,
                output_dir=FLAGS.output_dir,
                use_precomputed_msa=FLAGS.use_precomputed_msas,
                save_msa=FLAGS.save_msa_files,
            )
        else:
            logging.info("running mmseq now")
            m.make_mmseq_features(DEFAULT_API_SERVER=DEFAULT_API_SERVER,
                                  pipeline=pipeline, output_dir=FLAGS.output_dir
                                  )
        pickle.dump(m, open(f"{FLAGS.output_dir}/{m.description}.pkl", "wb"))
        del m


def iter_seqs(fasta_fns):
    """
    Generator that yields sequences and descriptions from multiple fasta files.

    Args:
    fasta_fns (list): A list of fasta file paths.

    Yields:
    tuple: A tuple containing a sequence and its corresponding description.
    """
    for fasta_path in fasta_fns:
        with open(fasta_path, "r") as f:
            sequences, descriptions = parse_fasta(f.read())
            for seq, desc in zip(sequences, descriptions):
                yield seq, desc


def main(argv):
    try:
        Path(FLAGS.output_dir).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        logging.error("Multiple processes are trying to create the same folder now.")
        pass
    if not FLAGS.use_mmseqs2:
        check_template_date_and_uniprot()

    if not FLAGS.path_to_mmt:
        process_sequences_individual_mode()
    else:
        process_sequences_multimeric_mode()


def check_template_date_and_uniprot():
    """
    Checks if the max_template_date is provided and updates the flags dictionary with the path to the Uniprot database.
    Exits the script if max_template_date is not provided or if the Uniprot database file is not found.
    """
    if not FLAGS.max_template_date:
        logging.info("You have not provided a max_template_date. Please specify a date and run again.")
        sys.exit()
    uniprot_database_path = os.path.join(FLAGS.data_dir, "uniprot/uniprot.fasta")
    flags_dict.update({"uniprot_database_path": uniprot_database_path})
    if not os.path.isfile(uniprot_database_path):
        logging.info(
            f"Failed to find uniprot.fasta under {uniprot_database_path}. Please make sure your data_dir has been configured correctly.")
        sys.exit()


def process_sequences_individual_mode():
    """
    Processes individual sequences specified in the fasta files. For each sequence, it creates a MonomericObject,
    processes it, and saves its features. Skips processing if the sequence index does not match the seq_index flag.

    """
    create_arguments()
    uniprot_runner = None if FLAGS.use_mmseqs2 else create_uniprot_runner(FLAGS.jackhmmer_binary_path,
                                                                          FLAGS.uniref90_database_path)
    pipeline = create_pipeline()
    seq_idx = 0
    for curr_seq, curr_desc in iter_seqs(FLAGS.fasta_paths):
        seq_idx += 1
        if FLAGS.seq_index is None or (FLAGS.seq_index == seq_idx):
            if curr_desc and not curr_desc.isspace():
                curr_monomer = MonomericObject(curr_desc, curr_seq)
                curr_monomer.uniprot_runner = uniprot_runner
                create_and_save_monomer_objects(curr_monomer, pipeline)


def process_sequences_multimeric_mode():
    """
    Processes sequences in multimeric mode using descriptions from a CSV file. For each entry in the CSV file,
    it processes the corresponding sequence if it matches the seq_index flag.
    """
    fasta_paths = flags_dict["fasta_paths"]
    feats = parse_csv_file(FLAGS.description_file, fasta_paths, FLAGS.path_to_mmt)
    logging.info(f"seq_index: {FLAGS.seq_index}, feats: {feats}")

    for idx, feat in enumerate(feats, 1):
        if FLAGS.seq_index is None or (FLAGS.seq_index == idx):
            process_multimeric_features(feat, idx)


def process_multimeric_features(feat, idx):
    """
    Processes a multimeric feature from a provided feature dictionary. It checks for the existence of template files
    and creates a custom database for the specified protein. It then processes the protein and saves its features.

    Args:
    feat (dict): A dictionary containing protein information and its corresponding templates and chains.
    idx (int): The index of the current protein being processed.
    """
    for temp_path in feat["templates"]:
        if not os.path.isfile(temp_path):
            logging.error(f"Template file {temp_path} does not exist.")
            raise FileNotFoundError(f"Template file {temp_path} does not exist.")

    protein = feat["protein"]
    chains = feat["chains"]
    templates = feat["templates"]
    logging.info(f"Processing {protein}: templates: {templates}, chains: {chains}")

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path_to_custom_db = create_custom_db(temp_dir, protein, templates, chains)
        create_arguments(local_path_to_custom_db)

        flags_dict.update({f"protein_{idx}": feat['protein'], f"multimeric_templates_{idx}": feat['templates'],
                           f"multimeric_chains_{idx}": feat['chains']})

        if not FLAGS.use_mmseqs2:
            uniprot_runner = create_uniprot_runner(FLAGS.jackhmmer_binary_path, FLAGS.uniref90_database_path)
        else:
            uniprot_runner = None
        pipeline = create_pipeline()
        curr_monomer = MonomericObject(feat['protein'], feat['sequence'])
        curr_monomer.uniprot_runner = uniprot_runner
        create_and_save_monomer_objects(curr_monomer, pipeline)


if __name__ == "__main__":
    flags.mark_flags_as_required(
        ["fasta_paths", "output_dir", "max_template_date", "data_dir"]
    )
    app.run(main)
