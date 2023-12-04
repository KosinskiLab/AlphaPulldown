#!/usr/bin/env python3

# create_individual_features using custom multimeric templates
#

from alphapulldown.objects import MonomericObject
from alphapulldown.utils import create_uniprot_runner, get_run_alphafold, convert_fasta_description_to_protein_name
from alphapulldown.create_custom_template_db import create_db
from alphafold.data.pipeline import DataPipeline
from alphafold.data.tools import hmmsearch
from alphafold.data import templates
from absl import logging, app
import os
import sys
from pathlib import Path
import tempfile
import csv
from create_individual_features import create_and_save_monomer_objects, iter_seqs


run_af = get_run_alphafold()
flags = run_af.flags

flags.DEFINE_integer("job_index", None, "index of job in the description file, starting from 1")

flags.DEFINE_string("description_file", None, "Path to the text file with descriptions")

flags.DEFINE_string("path_to_mmt", None, "Path to directory with multimeric template mmCIF files")

flags.DEFINE_float("threshold_clashes", 1000, "Threshold for VDW overlap to identify clashes "
                                             "(default: 1000, i.e. no threshold, for thresholding, use 0.9)")

flags.DEFINE_float("hb_allowance", 0.4, "Allowance for hydrogen bonding (default: 0.4)")

flags.DEFINE_float("plddt_threshold", 0, "Threshold for pLDDT score (default: 0)")

FLAGS = flags.FLAGS
MAX_TEMPLATE_HITS = 1000000  # make it large enough to get all templates


def create_arguments(flags_dict, local_path_to_custom_template_db):
    """Create arguments for alphafold.run()"""
    global use_small_bfd

    # Path to the Uniref30 database for use by HHblits.
    FLAGS.uniref30_database_path = FLAGS.uniref30_database_path or os.path.join(
        FLAGS.data_dir, "uniref30", "UniRef30_2021_03"
    )
    flags_dict.update({"uniref30_database_path": FLAGS.uniref30_database_path})

    # Path to the Uniref90 database for use by JackHMMER.
    FLAGS.uniref90_database_path = FLAGS.uniref90_database_path or os.path.join(
        FLAGS.data_dir, "uniref90", "uniref90.fasta"
    )
    flags_dict.update({"uniref90_database_path": FLAGS.uniref90_database_path})

    # Path to the MGnify database for use by JackHMMER.
    FLAGS.mgnify_database_path = FLAGS.mgnify_database_path or os.path.join(
        FLAGS.data_dir, "mgnify", "mgy_clusters_2022_05.fa"
    )
    flags_dict.update({"mgnify_database_path": FLAGS.mgnify_database_path})

    # Path to the BFD database for use by HHblits.
    FLAGS.bfd_database_path = FLAGS.bfd_database_path or os.path.join(
        FLAGS.data_dir,
        "bfd",
        "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
    )
    flags_dict.update({"bfd_database_path": FLAGS.bfd_database_path})

    # Path to the Small BFD database for use by JackHMMER.
    FLAGS.small_bfd_database_path = FLAGS.small_bfd_database_path or os.path.join(
        FLAGS.data_dir, "small_bfd", "bfd-first_non_consensus_sequences.fasta"
    )
    flags_dict.update({"small_bfd_database_path": FLAGS.small_bfd_database_path})

    # Path to pdb70 database
    FLAGS.pdb70_database_path = FLAGS.pdb70_database_path or os.path.join(FLAGS.data_dir, "pdb70", "pdb70")
    flags_dict.update({"pdb70_database_path": FLAGS.pdb70_database_path})

    use_small_bfd= FLAGS.db_preset == "reduced_dbs"
    flags_dict.update({"use_small_bfd": use_small_bfd})

    # Update pdb related flags
    FLAGS.pdb_seqres_database_path = os.path.join(local_path_to_custom_template_db, "pdb_seqres", "pdb_seqres.txt")
    flags_dict.update({"pdb_seqres_database_path": FLAGS.pdb_seqres_database_path})
    FLAGS.template_mmcif_dir = os.path.join(local_path_to_custom_template_db, "pdb_mmcif", "mmcif_files")
    flags_dict.update({"template_mmcif_dir": FLAGS.template_mmcif_dir})
    FLAGS.obsolete_pdbs_path = os.path.join(local_path_to_custom_template_db, "pdb_mmcif", "obsolete.dat")
    flags_dict.update({"obsolete_pdbs_path": FLAGS.obsolete_pdbs_path})


def create_custom_db(temp_dir, protein, templates, chains):
    # Create custom template database
    threashold_clashes = FLAGS.threshold_clashes
    hb_allowance = FLAGS.hb_allowance
    plddt_threshold = FLAGS.plddt_threshold
    #local_path_to_custom_template_db = Path(".") / "custom_template_db" / protein # DEBUG
    local_path_to_custom_template_db = Path(temp_dir) / "custom_template_db" / protein
    logging.info(f"Path to local database: {local_path_to_custom_template_db}")
    create_db(local_path_to_custom_template_db, templates, chains, threashold_clashes, hb_allowance, plddt_threshold)

    return local_path_to_custom_template_db



def parse_csv_file(csv_path, fasta_paths, mmt_dir):
    """
    o csv_path: Path to the text file with descriptions
        features.csv: A coma-separated file with three columns: PROTEIN name, PDB/CIF template, chain ID.
    o fasta_paths: path to fasta file(s)
    o mmt_dir: Path to directory with multimeric template mmCIF files

    Returns:
        a list of dictionaries with the following structure:
    [{"protein": protein_name, "sequence" :sequence", templates": [pdb_files], "chains": [chain_id]}, ...]}]
    """
    protein_names = {}
    # Check that fasta files exist
    for fasta_path in fasta_paths:
        logging.info(f"Parsing {fasta_path}...")
        if not os.path.isfile(fasta_path):
            raise FileNotFoundError(f"Fasta file {fasta_path} does not exist. Please check your input file.")
    # Parse all protein names from fasta files
    for curr_seq, curr_desc in iter_seqs(fasta_paths):
        if curr_desc in protein_names.keys():
            continue
        protein_names[curr_desc] = curr_seq

    # Parse csv file
    parsed_dict = {}
    with open(csv_path, newline="") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # skip empty lines
            if not row:
                continue
            if len(row) == 3:
                protein, template, chain = [item.strip() for item in row]
                # Remove special symbols from protein name
                protein = convert_fasta_description_to_protein_name(protein)
                if protein not in protein_names:
                    raise Exception(f"Protein {protein} from description.csv is not found in the fasta file(s)."
                                    f"List of proteins in fasta file(s): {protein_names}")
                if protein not in parsed_dict:
                    parsed_dict[protein] = {
                        "protein": protein,
                        "templates": [],
                        "chains": [],
                        "sequence": None,
                    }
                parsed_dict[protein]["sequence"] = protein_names[protein]
                parsed_dict[protein]["templates"].append(os.path.join(mmt_dir, template))
                parsed_dict[protein]["chains"].append(chain)
            else:
                logging.error(f"Invalid line found in the file {csv_path}: {row}")
                sys.exit()

    return list(parsed_dict.values())


def create_pipeline():
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


def main(argv):
    try:
        Path(FLAGS.output_dir).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        logging.info("Multiple processes are trying to create the same folder now.")

    flags_dict = FLAGS.flag_values_dict()
    fasta_paths = flags_dict["fasta_paths"]
    feats = parse_csv_file(FLAGS.description_file, fasta_paths, FLAGS.path_to_mmt)
    logging.info(f"job_index: {FLAGS.job_index} feats: {feats}")
    for idx, feat in enumerate(feats, 1):
        if (FLAGS.job_index is None) or (FLAGS.job_index == idx):
            for temp in feat["templates"]:
                if not os.path.isfile:
                    logging.error(f"Template file {temp} does not exist. Please check your input file.")
                    sys.exit()
            protein = feat["protein"]
            chains = feat["chains"]
            templates = feat["templates"]
            logging.info(f"Processing {protein}: templates: {templates} chains: {chains}")
            # For each protein, create a temp dir
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path_to_custom_db = create_custom_db(temp_dir, protein, templates, chains)
                create_arguments(flags_dict, local_path_to_custom_db)

                # Update flags_dict to store data about templates
                flags_dict.update({f"protein_{idx}": feat['protein']})
                flags_dict.update({f"multimeric_templates_{idx}": feat['templates']})
                flags_dict.update({f"multimeric_chains_{idx}": feat['chains']})

                if not FLAGS.use_mmseqs2:
                    if not FLAGS.max_template_date:
                        logging.info("You have not provided a max_template_date. Please specify a date and run again.")
                        sys.exit()
                    else:
                        pipeline = create_pipeline()
                        uniprot_database_path = os.path.join(FLAGS.data_dir, "uniprot/uniprot.fasta")
                        flags_dict.update({"uniprot_database_path": uniprot_database_path})
                        if os.path.isfile(uniprot_database_path):
                            uniprot_runner = create_uniprot_runner(FLAGS.jackhmmer_binary_path, uniprot_database_path)
                        else:
                            logging.info(
                                f"Failed to find uniprot.fasta under {uniprot_database_path}."
                                "Please make sure your data_dir has been configured correctly."
                            )
                            sys.exit()
                else:
                    pipeline = create_pipeline()
                    uniprot_runner = None
                    flags_dict = FLAGS.flag_values_dict()
                curr_monomer = MonomericObject(feat['protein'], feat['sequence'])
                curr_monomer.uniprot_runner = uniprot_runner
                create_and_save_monomer_objects(
                    curr_monomer,
                    pipeline,
                    flags_dict,
                    use_mmseqs2=FLAGS.use_mmseqs2,
                )


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "description_file",
            "fasta_paths",
            "path_to_mmt",
            "output_dir",
            "max_template_date",
            "data_dir",
        ]
    )
    app.run(main)
