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
from Bio.PDB.Polypeptide import three_to_one, one_to_three
from alphapulldown.remove_clashes_low_plddt import remove_clashes, remove_low_plddt, to_bio
from colabfold.batch import validate_and_fix_mmcif
from alphafold.common.protein import _from_bio_structure, to_mmcif

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
    if len(struct.child_list) > 1:
        raise Exception(f'{len(struct.child_list)} models found in {file_path}!')

    model = struct[0]
    chain_ids = [chain.id for chain in model]

    if chain_id not in chain_ids:
        logging.error(f"No {chain_id} in {chain_ids} of {file_path}!")
        return []  # Exit if chain_id is not found

    seqs = []

    # Parsing SEQRES
    for record in SeqIO.parse(file_path, "cif-seqres"):
        if record.id == chain_id:
            seqs.append((chain_id, str(record.seq)))

    # Parsing from atoms if SEQRES records are not found
    if len(seqs) == 0:
        logging.info(f'No SEQRES records found in {file_path}! Parsing from atoms!')
        for chain in model:
            if chain.id == chain_id:
                seq_chain = ''
                for resi in chain:
                    try:
                        one_letter = three_to_one(resi.resname)
                        seq_chain += one_letter
                    except KeyError:
                        logging.warning(f'Skipping {resi.resname} with id {resi.id}')
                seqs.append((chain_id, seq_chain))

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


def replace_entity_poly_seq(mmcif_string, seqs, chain_id):
    new_mmcif_string = []
    saving = True
    start = -1
    # Remove old entity_poly_seq lines
    for index, line in enumerate(mmcif_string.splitlines()):
        if line.startswith("_entity_poly_seq.entity_id"):
            saving = False
            start = index
        if not saving and line.startswith("#"):
            saving = True
        if saving:
            new_mmcif_string.append(line)
    # Construct new entity_poly_seq lines
    new_entity_poly_seq = []
    for seq in seqs:
        if seq[1] == chain_id:
            new_entity_poly_seq.append("_entity_poly_seq.entity_id")
            new_entity_poly_seq.append("_entity_poly_seq.num")
            new_entity_poly_seq.append("_entity_poly_seq.mon_id")
            new_entity_poly_seq.append("_entity_poly_seq.hetero")
            entity_id = ord(chain_id.upper()) - 64
            for i, aa in enumerate(seq[0]):
                three_letter_aa = one_to_three(aa)
                # TODO: uncomment after chain ids are properly saved
                #new_entity_poly_seq.append(f"{entity_id}\t{i+1}\t{three_letter_aa}\tn")
                new_entity_poly_seq.append(f"0\t{i + 1}\t{three_letter_aa}\tn")
    # Insert new entity_poly_seq lines at the start index
    new_mmcif_string[start:start] = new_entity_poly_seq
    new_mmcif_string.append("\n")
    return '\n'.join(new_mmcif_string)



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
        # Parse SEQRES from the original template file
        seqs = extract_seqs_from_cif(template, chain_id)
        sqrres_path = save_seqres(code, seqs, seqres_dir)
        logging.info(f"SEQRES saved to {sqrres_path}!")
        # Prepare pdb_mmcif directory
        structure = to_bio(template, chain_id)
        # Remove clashes and low pLDDT regions for each template
        structure = remove_clashes(structure, threshold_clashes, hb_allowance)
        structure = remove_low_plddt(structure, plddt_threshold)
        # Convert to Protein
        protein = _from_bio_structure(structure)
        # Convert to mmCIF
        mmcif_string = to_mmcif(protein, f"{code}_{chain_id}", "Monomer")
        # Remove lines containing UNK
        mmcif_string = replace_entity_poly_seq(mmcif_string, seqs, chain_id)
        # Save to file
        fn = mmcif_dir / f"{code}.cif"
        with open(fn, 'w') as f:
            f.write(mmcif_string)
        # Fix and validate with ColabFold
        validate_and_fix_mmcif(fn)
        logging.info(f'{template} is done!')




def main(argv):
    flags.FLAGS(argv)
    create_db(flags.FLAGS.out_path, [flags.FLAGS.template], [flags.FLAGS.multimeric_chain])


if __name__ == '__main__':
    flags.DEFINE_string("out_path", None, "Path to the output directory")
    flags.DEFINE_string("template", None, "Path to the template mmCIF/PDB file")
    flags.DEFINE_string("multimeric_chain", None, "Chain ID of the multimeric template")
    flags.mark_flags_as_required(["out_path", "template", "multimeric_chain"])
    app.run(main)
