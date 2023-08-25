#!/usr/bin/env python3

"""
This script generates a fake template database for AlphaFold2 from PDB or mmCIF
template files.
Removes steric clashes and low pLDDT regions from the template files.
Can be used as a standalone script.

"""

import os
import sys
from pathlib import Path
from absl import logging, flags, app
from Bio import SeqIO
from Bio.PDB.Polypeptide import three_to_one
from alphapulldown.remove_clashes_low_plddt import remove_clashes, remove_low_plddt, to_bio
from colabfold.batch import validate_and_fix_mmcif
from alphafold.common.protein import Protein, _from_bio_structure, to_pdb, to_mmcif

FLAGS = flags.FLAGS



def extract_seqs_from_cif(file_path, chain_id):
    """
    Extract sequences from PDB/CIF file, if SEQRES records are not present,
    extract from atoms
    o file_path - path to CIF file
    o chain id - chain id
    Return:
        o list of tuples: (chain_id, sequence)
    """
    struct = to_bio(file_path)
    # Die if more than 1 model in the structure
    if len(struct.child_list) > 1:
        raise Exception(f'{len(struct.child_list)} models found in {cif_fn}!')
    model = struct[0]
    chain_ids = [chain.id for chain in model]
    if chain_id not in chain_ids:
        logging.error(f"No {chain_id} in {chain_ids} of {file_path}!")
    # Try to parse SEQRES records from mmCIF file
    seqs = []
    for record in SeqIO.parse(file_path, "cif-seqres"):
        if record.id != chain_id:
            logging.info("Parsing from seqres: Skipping chain %s", record.id)
            continue
        seqs.append((record.seq, record.id))
    if len(seqs) == 0:
        logging.info(f'No SEQRES records found in {file_path}! Parsing from atoms!')
        # Parse from atoms
        for chain in model:
            if chain.id != chain_id:
                logging.info("Parsing from atoms: Skipping chain %s", chain.id)
            seq_chain = ''
            for resi in chain:
                try:
                    one_letter = three_to_one(resi.resname)
                    seq_chain += one_letter
                except KeyError:
                    logging.warning(f'Skipping {resi.resname} with id {resi.id}')
                    continue
            seqs.append((seq_chain, chain.id))
    return seqs


def save_seqres(code, seqs, path):
    """
    o code - four letter PDB-like code
    o seqs - list of tuples: (chain_id, sequence)
    o path - path to the pdb_seqresi, unique for each chain
    Returns:
        o Path to the file
    """
    fn = path / 'pdb_seqres.txt'
    # Rewrite the file if it exists
    with open(fn, 'a') as f:
        for count, seq in enumerate(seqs):
            chain = seq[1]
            s = seq[0]
            lines = f">{code}_{chain} mol:protein length:{len(s)}\n{s}\n"
            logging.info(f'Saving SEQRES for chain {chain} to {fn}!')
            logging.debug(lines)
            f.write(lines)
    return fn


def parse_code(template):
    # Check that the code is 4 letters
    code = Path(template).stem
    with open(template, "r") as f:
        for line in f:
            if line.startswith("_entry.id"):
                code = line.split()[1]
                if len(code) != 4:
                    logging.error(f'Error for template {template}!\n'
                                  f'Code must have 4 characters but is {code}\n')
                    sys.exit(1)
    return code.lower()


def create_db(out_path, templates, chains, threshold_clashes, hb_allowance, plddt_threshold):
    """
    Main function that creates a fake template database for AlphaFold2
    from a PDB/CIF template files.
    o out_path - path to the output directory where the database will be created
    o templates - list of paths to the template files
    o chains - list of chain IDs of the multimeric templates
    o threshold_clashes - threshold for clashes removal
    o hb_allowance - hb_allowance for clashes removal
    o plddt_threshold - threshold for low pLDDT regions removal
    Returns:
        o None
    """

    # Create the database structure
    pdb_mmcif_dir = Path(out_path) / 'pdb_mmcif'
    mmcif_dir = pdb_mmcif_dir / 'mmcif_files'
    seqres_dir = Path(out_path) / 'pdb_seqres'
    try:
        Path(mmcif_dir).mkdir(parents=True)
        # Create empty obsolete.dat file
        open(pdb_mmcif_dir / 'obsolete.dat', 'a').close()
    except FileExistsError:
        logging.info("Output mmcif directory already exists!")
        logging.info("The existing database will be overwritten!")
        mmcif_files = os.listdir(mmcif_dir)
        if len(mmcif_files) > 0:
            logging.info("Removing existing mmcif files!")
            for f in mmcif_files:
                os.remove(mmcif_dir / Path(f))
    try:
        Path(seqres_dir).mkdir(parents=True)
    except FileExistsError:
        logging.info("Output mmcif directory already exists!")
        logging.info("The existing database will be overwritten!")
        if os.path.exists(seqres_dir / 'pdb_seqres.txt'):
            os.remove(seqres_dir / 'pdb_seqres.txt')

    for template, chain_id in zip(templates, chains):
        code = parse_code(template)
        logging.info(f"Processing template: {template}  Chain {chain_id} Code: {code}")
        structure = to_bio(template, chain_id)
        # Remove clashes and low pLDDT regions for each template
        structure = remove_clashes(structure, threshold_clashes, hb_allowance)
        structure = remove_low_plddt(structure, plddt_threshold)
        # Convert to Protein
        protein = _from_bio_structure(structure)
        # Convert to mmCIF
        mmcif_string = to_mmcif(protein, f"{code}_{chain_id}", "Monomer")
        # Save to file
        fn = mmcif_dir / f"{code}_{chain_id}.cif"
        with open(fn, 'w') as f:
            f.write(mmcif_string)
        # Fix and validate with ColabFold
        validate_and_fix_mmcif(fn)
        logging.info(f'{template} is done!')

        # Parse SEQRES from the original template file
        seqs = extract_seqs_from_cif(template, chain_id)
        sqrres_path = save_seqres(code, seqs, seqres_dir)
        logging.info(f"SEQRES saved to {sqrres_path}!")


def main(argv):
    flags.FLAGS(argv)
    create_db(flags.FLAGS.out_path, [flags.FLAGS.template], [flags.FLAGS.multimeric_chain])


if __name__ == '__main__':
    flags.DEFINE_string("out_path", None, "Path to the output directory")
    flags.DEFINE_string("template", None, "Path to the template mmCIF/PDB file")
    flags.DEFINE_string("multimeric_chain", None, "Chain ID of the multimeric template")
    flags.mark_flags_as_required(["out_path", "template", "multimeric_chain"])
    app.run(main)
