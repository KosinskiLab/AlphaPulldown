#!/usr/bin/env python3
# coding: utf-8
# Create features for AlphaFold from fasta file(s) or a csv file with descriptions for multimeric templates
# #

import os
import pickle
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from absl import logging, app
from alphafold.data import templates
from alphafold.data.pipeline import DataPipeline
from alphafold.data.tools import hmmsearch, hhsearch
from colabfold.utils import DEFAULT_API_SERVER

from alphapulldown.utils.create_custom_template_db import create_db
from alphapulldown.objects import MonomericObject
from alphapulldown.utils.file_handling import iter_seqs, parse_csv_file
from alphapulldown.utils.modelling_setup import get_run_alphafold, create_uniprot_runner
from alphapulldown.utils import save_meta_data

# Initialize and define flags
run_af = get_run_alphafold()
flags = run_af.flags
_check_flag = getattr(run_af, "_check_flag", None)

# All flags
flags.DEFINE_bool("use_mmseqs2", False,
                  "Use mmseqs2 remotely or not. 'true' or 'false', default is 'false'")
flags.DEFINE_bool("save_msa_files", False, "Save MSA output or not")
flags.DEFINE_bool("skip_existing", False,
                  "Skip existing monomer feature pickles or not")
flags.DEFINE_string("new_uniclust_dir", None,
                    "Directory where new version of uniclust is stored")
flags.DEFINE_integer(
    "seq_index", None, "Index of sequence in the fasta file, starting from 1")

flags.DEFINE_boolean("use_hhsearch", False,
                     "Use hhsearch instead of hmmsearch when looking for structure template. Default is False")

# Flags related to TrueMultimer
flags.DEFINE_string("path_to_mmt", None,
                    "Path to directory with multimeric template mmCIF files")
flags.DEFINE_string("description_file", None,
                    "Path to the text file with descriptions")
flags.DEFINE_float("threshold_clashes", 1000,
                   "Threshold for VDW overlap to identify clashes. The VDW overlap between two atoms is defined as "
                   "the sum of their VDW radii minus the distance between their centers."
                   "If the overlap exceeds this threshold, the two atoms are considered to be clashing."
                   "A positive threshold is how far the VDW surfaces are allowed to interpenetrate before considering "
                   "the atoms to be clashing."
                   "(default: 1000, i.e. no threshold, for thresholding, use 0.6-0.9)")
flags.DEFINE_float("hb_allowance", 0.4,
                   "Additional allowance for hydrogen bonding (default: 0.4)")
flags.DEFINE_float("plddt_threshold", 0,
                   "Threshold for pLDDT score (default: 0)")
flags.DEFINE_boolean("multiple_mmts", False,
                     "Use multiple mmts or not. 'true' or 'false', default is 'false'")

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20

flags_dict = FLAGS.flag_values_dict()


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

    FLAGS.uniref30_database_path = get_database_path(FLAGS.uniref30_database_path,
                                                     "uniref30/UniRef30_2023_02")
    FLAGS.uniref90_database_path = get_database_path(FLAGS.uniref90_database_path,
                                                     "uniref90/uniref90.fasta")
    FLAGS.mgnify_database_path = get_database_path(FLAGS.mgnify_database_path,
                                                   "mgnify/mgy_clusters_2022_05.fa")
    FLAGS.bfd_database_path = get_database_path(FLAGS.bfd_database_path,
                                                "bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt")
    FLAGS.small_bfd_database_path = get_database_path(FLAGS.small_bfd_database_path,
                                                      "small_bfd/bfd-first_non_consensus_sequences.fasta")
    FLAGS.pdb70_database_path = get_database_path(
        FLAGS.pdb70_database_path, "pdb70/pdb70")

    use_small_bfd = FLAGS.db_preset == "reduced_dbs"
    flags_dict.update({"use_small_bfd": use_small_bfd})

    # Update pdb related flags
    if local_path_to_custom_template_db:
        FLAGS.pdb_seqres_database_path = os.path.join(
            local_path_to_custom_template_db, "pdb_seqres", "pdb_seqres.txt")
        flags_dict.update(
            {"pdb_seqres_database_path": FLAGS.pdb_seqres_database_path})
        FLAGS.template_mmcif_dir = os.path.join(
            local_path_to_custom_template_db, "pdb_mmcif", "mmcif_files")
        flags_dict.update({"template_mmcif_dir": FLAGS.template_mmcif_dir})
        FLAGS.obsolete_pdbs_path = os.path.join(
            local_path_to_custom_template_db, "pdb_mmcif", "obsolete.dat")
        flags_dict.update({"obsolete_pdbs_path": FLAGS.obsolete_pdbs_path})
    else:
        FLAGS.pdb_seqres_database_path = get_database_path(
            FLAGS.pdb_seqres_database_path, "pdb_seqres/pdb_seqres.txt")
        flags_dict.update(
            {"pdb_seqres_database_path": FLAGS.pdb_seqres_database_path})
        FLAGS.template_mmcif_dir = get_database_path(
            FLAGS.template_mmcif_dir, "pdb_mmcif/mmcif_files")
        flags_dict.update({"template_mmcif_dir": FLAGS.template_mmcif_dir})
        FLAGS.obsolete_pdbs_path = get_database_path(
            FLAGS.obsolete_pdbs_path, "pdb_mmcif/obsolete.dat")
        flags_dict.update({"obsolete_pdbs_path": FLAGS.obsolete_pdbs_path})


def create_custom_db(temp_dir, protein, template_paths, chains):
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
    local_path_to_custom_template_db = Path(
        temp_dir) / "custom_template_db" / protein
    logging.info(f"Path to local database: {local_path_to_custom_template_db}")
    create_db(
        local_path_to_custom_template_db, template_paths, chains, threashold_clashes, hb_allowance, plddt_threshold
    )

    return local_path_to_custom_template_db


def create_pipeline():
    """
    Creates and returns a data pipeline for AlphaFold, configured with necessary binary paths and database paths.

    Returns:
    DataPipeline: An instance of the AlphaFold DataPipeline configured with necessary paths.
    """
    if FLAGS.use_hhsearch:
        logging.info("Will use hhsearch looking for templates")
        template_searcher = hhsearch.HHSearch(
            binary_path=FLAGS.hhsearch_binary_path,
            databases=[FLAGS.pdb70_database_path]
        )
        template_featuriser = templates.HhsearchHitFeaturizer(
            mmcif_dir=FLAGS.template_mmcif_dir,
            max_template_date=FLAGS.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=FLAGS.kalign_binary_path,
            release_dates_path=None,
            obsolete_pdbs_path=FLAGS.obsolete_pdbs_path
        )
    else:
        logging.info("Will use hmmsearch looking for templates")
        template_featuriser = templates.HmmsearchHitFeaturizer(
            mmcif_dir=FLAGS.template_mmcif_dir,
            max_template_date=FLAGS.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=FLAGS.kalign_binary_path,
            obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
            release_dates_path=None,
        )
        template_searcher = hmmsearch.Hmmsearch(
            binary_path=FLAGS.hmmsearch_binary_path,
            hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
            database_path=FLAGS.pdb_seqres_database_path,
        )
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
        template_searcher=template_searcher,
        template_featurizer=template_featuriser
    )
    return monomer_data_pipeline


def create_and_save_monomer_objects(monomer, pipeline):
    """
    Processes a MonomericObject to create and save its features. Skips processing if the feature file already exists
    and skipping is enabled.

    Args:
    monomer (MonomericObject): The monomeric object to process.
    pipeline (DataPipeline): The data pipeline object for feature creation.
    """
    pickle_path = os.path.join(FLAGS.output_dir, f"{monomer.description}.pkl")

    # Check if we should skip existing files
    if FLAGS.skip_existing and os.path.exists(pickle_path):
        logging.info(
            f"Feature file for {monomer.description} already exists. Skipping...")
        return

    # Save metadata
    metadata_output_path = os.path.join(FLAGS.output_dir,
                                        f"{monomer.description}_feature_metadata_{datetime.date(datetime.now())}.json")
    with save_meta_data.output_meta_file(metadata_output_path) as meta_data_outfile:
        save_meta_data.save_meta_data(flags_dict, meta_data_outfile)

    # Create features
    if FLAGS.use_mmseqs2:
        logging.info("Running MMseqs2 for feature generation...")
        monomer.make_mmseq_features(
            DEFAULT_API_SERVER=DEFAULT_API_SERVER,
            output_dir=FLAGS.output_dir
        )
    else:
        monomer.make_features(
            pipeline=pipeline,
            output_dir=FLAGS.output_dir,
            use_precomputed_msa=FLAGS.use_precomputed_msas,
            save_msa=FLAGS.save_msa_files,
        )

    # Save the processed monomer object
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(monomer, pickle_file)

    # Optional: Clear monomer from memory if necessary
    del monomer


def check_template_date_and_uniprot():
    """
    Checks if the max_template_date is provided and updates the flags dictionary with the path to the Uniprot database.
    Exits the script if max_template_date is not provided or if the Uniprot database file is not found.
    """
    if not FLAGS.max_template_date:
        logging.info(
            "You have not provided a max_template_date. Please specify a date and run again.")
        sys.exit()
    uniprot_database_path = os.path.join(
        FLAGS.data_dir, "uniprot/uniprot.fasta")
    flags_dict.update({"uniprot_database_path": uniprot_database_path})
    if not os.path.isfile(uniprot_database_path):
        logging.info(
            f"Failed to find uniprot.fasta under {uniprot_database_path}."
            f" Please make sure your data_dir has been configured correctly.")
        sys.exit()


def process_sequences_individual_mode():
    """
    Processes individual sequences specified in the fasta files. For each sequence, it creates a MonomericObject,
    processes it, and saves its features. Skips processing if the sequence index does not match the seq_index flag.

    """
    create_arguments()
    uniprot_runner = None if FLAGS.use_mmseqs2 else create_uniprot_runner(FLAGS.jackhmmer_binary_path,
                                                                          FLAGS.uniref90_database_path)
    pipeline = None if FLAGS.use_mmseqs2 else create_pipeline()
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
    fasta_paths = FLAGS.fasta_paths
    feats = parse_csv_file(FLAGS.description_file, fasta_paths, FLAGS.path_to_mmt, FLAGS.multiple_mmts)

    for idx, feat in enumerate(feats, 1):
        if FLAGS.seq_index is None or (FLAGS.seq_index == idx):
            logging.info(f"seq_index: {FLAGS.seq_index}, feats: {feat}")
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
            raise FileNotFoundError(
                f"Template file {temp_path} does not exist.")

    protein = feat["protein"]
    chains = feat["chains"]
    template_paths = feat["templates"]
    logging.info(
        f"Processing {protein}: templates: {templates}, chains: {chains}")

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path_to_custom_db = create_custom_db(
            temp_dir, protein, template_paths, chains)
        create_arguments(local_path_to_custom_db)

        flags_dict.update({f"protein_{idx}": protein, f"multimeric_templates_{idx}": template_paths,
                           f"multimeric_chains_{idx}": chains})

        if not FLAGS.use_mmseqs2:
            uniprot_runner = create_uniprot_runner(
                FLAGS.jackhmmer_binary_path, FLAGS.uniref90_database_path)
        else:
            uniprot_runner = None
        pipeline = create_pipeline()
        curr_monomer = MonomericObject(protein, feat['sequence'])
        curr_monomer.uniprot_runner = uniprot_runner
        create_and_save_monomer_objects(curr_monomer, pipeline)


def main(argv):
    del argv  # Unused.
    _check_flag('use_mmseqs2', 'path_to_mmt', False)  # Can't be both True.
    try:
        Path(FLAGS.output_dir).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        logging.error(
            "Multiple processes are trying to create the same folder now.")
        pass
    if not FLAGS.use_mmseqs2:
        check_template_date_and_uniprot()

    if not FLAGS.path_to_mmt:
        process_sequences_individual_mode()
    else:
        process_sequences_multimeric_mode()


if __name__ == "__main__":
    flags.mark_flags_as_required(
        ["fasta_paths", "output_dir", "max_template_date", "data_dir"]
    )
    app.run(main)
