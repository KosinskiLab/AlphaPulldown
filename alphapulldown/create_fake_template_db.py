#!/usr/bin/env python3

'''
This script generates a fake template database for AlphaFold2 from a PDB or mmCIF
template file. Mainly based on ColabFold functions. Can be used as a standalone script.

'''

import os
import sys
from pathlib import Path

from Bio.PDB import MMCIFParser
from Bio import SeqIO
from absl import logging, flags, app
from Bio.PDB.Polypeptide import three_to_one

import colabfold.utils
from colabfold.batch import validate_and_fix_mmcif, convert_pdb_to_mmcif
import shutil

FLAGS = flags.FLAGS

extra_fields = '''
#
_chem_comp.id ala
_chem_comp.type other
_struct_asym.id
_struct_asym.entity_id
_entity_poly_seq.mon_id
'''


def save_cif(cif_fn, code, chain_id, path):
    """
    Read, validate and fix CIF file using ColabFold
    o cif_fn - path to the CIF file
    o code - four letter PDB-like code
    o chain_id - chain ID of the multimeric template
    o path - path where to save the CIF file
    """
    p = MMCIFParser()
    try:
        validate_and_fix_mmcif(cif_fn)
        logging.info(f'Validated and fixed {cif_fn}!')
    except Exception as e:
        logging.warning(f'Exception: {e}'
                        f'Cannot validate and fix {cif_fn}!')
    struct = p.get_structure(code, cif_fn)
    if len(struct.child_list) > 1:
        raise Exception(f'{len(struct.child_list)} models found in {cif_fn}!')
    # Check that it's only 1 model and chain_id is in the structure
    for model in struct:
        chain_ids = [chain.id for chain in model]
    if chain_id not in chain_ids:
        logging.warning(f"Warning! SEQRES chains may be different from ATOM chains!"
                        f"Chain {chain_id} not found in {cif_fn}!"
                        f"Found chains: {chain_ids}!")
    else:
        logging.info(f'Found chain {chain_id} in {cif_fn}!')
    # cif_io.save(path)
    # cif is corrupted due to Biopython bug, just copy template instead
    out_path = Path(path) / f'{code}.cif'
    shutil.copyfile(cif_fn, out_path)
    return out_path


def extract_seqs_from_cif(file_path, chain_id):
    """
    Extract sequences from PDB/CIF file, if SEQRES records are not present,
    extract from atoms
    o file_path - path to CIF file
    o chain id - chain id
    Return:
        o list of tuples: (chain_id, sequence)
    """
    seqs = []
    # Get the SEQRES records from the structure
    for record in SeqIO.parse(file_path, "cif-seqres"):
        if record.id != chain_id:
            logging.info("Skipping chain %s", record.id)
            continue
        seqs.append((record.seq, record.id))
    if len(seqs) == 0:
        logging.info(f'No SEQRES records found in {file_path}! Parsing from atoms!')
        # Get the SEQRES records from the structure
        cif_io = colabfold.utils.CFMMCIFIO()
        p = MMCIFParser()
        struct = p.get_structure('template', file_path)
        # Iterate through all chains in all models of the structure
        for model in struct.child_list:
            for chain in model:
                if chain.id != chain_id:
                    logging.info("Skipping chain %s", chain.id)
                    continue
                else:
                    seq_chain = ''
                    for resi in chain:
                        try:
                            one_letter = three_to_one(resi.resname)
                            seq_chain += one_letter
                        except KeyError:
                            logging.warning(f'Cannot convert {resi.resname} '
                                            f'to one letter code!')
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
            #logging.info(lines)
            f.write(lines)
    return fn


def parse_code(template):
    # Check that the code is 4 letters
    with open(template, "r") as f:
        for line in f:
            if line.startswith("_entry.id"):
                code = line.split()[1]
                if len(code) != 4:
                    logging.error(f'Error for template {template}!\n'
                                  f'Code must have 4 characters but is {code}\n')
                    sys.exit(1)
    return code.lower()

def create_db(out_path, templates, chains):
    """
    Main function that creates a fake template database for AlphaFold2
    from a PDB/CIF template files.
    o out_path - path to the output directory where the database will be created
    o templates - list of paths to the template files
    o chains - list of chain IDs of the multimeric templates
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

    # Convert PDB/MMCIF to the proper mmCIF files
    for template, chain_id in zip(templates, chains):
        code = parse_code(template)
        logging.info(f"Processing template: {template}  Code: {code}")
        if template.endswith('pdb'):
            logging.info(f"Converting {template} to CIF!")
            convert_pdb_to_mmcif(Path(template))
            cif = save_cif(template.replace('.pdb', '.cif'), code, chain_id, mmcif_dir)
        elif template.endswith('cif'):
            logging.info(f"Reading {template}!")
            cif = save_cif(template, code, chain_id, mmcif_dir)
        else:
            logging.error('Unknown format of ', template)
            sys.exit(1)

        # Save pdb_seqres.txt file to pdb_seqres
        seqs = extract_seqs_from_cif(cif, chain_id)
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
