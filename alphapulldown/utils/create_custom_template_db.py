#!/usr/bin/env python3

"""
This script generates a custom template database for AlphaFold2 from PDB or mmCIF
template files.
Removes steric clashes and low pLDDT regions from the template files.
Duplicates one template 4 times to increase impact on AF.
Can be used as a standalone script.

"""

import os
import hashlib
import base64
from pathlib import Path
from absl import logging, flags, app
from alphapulldown.utils.remove_clashes_low_plddt import MmcifChainFiltered
from colabfold.batch import validate_and_fix_mmcif
from alphafold.common.protein import _from_bio_structure, to_mmcif

FLAGS = flags.FLAGS


def save_seqres(code, chain, s, seqres_path, duplicate):
    """
    o code - four letter PDB-like code
    o chain - chain ID
    o s - sequence
    o seqres_path - path to the pdb_seqres.txt file
    Returns:
        o Path to the file
    """
    fn = seqres_path

    seqres_entries = []
    if duplicate:
        for i in range(1, 5):
            temp_code = f"{code[:-1]}{i}"
            seqres_entries.append(f">{temp_code}_{chain} mol:protein length:{len(s)}\n{s}\n")
    else:
        seqres_entries.append(f">{code}_{chain} mol:protein length:{len(s)}\n{s}\n")

    with open(fn, 'a') as f:
        for entry in seqres_entries:
            logging.info(f'Saving SEQRES for chain {chain} to {fn} with code {entry.split()[0][1:]}!')
            logging.debug(entry)
            f.write(entry)

    return fn


def generate_code(filename):
    # Create a hash of the filename
    hash_object = hashlib.sha256(filename.encode())
    # Convert the hash to a base64 encoded string
    base64_hash = base64.urlsafe_b64encode(hash_object.digest())
    # Take the first 4 characters of the base64 encoded hash
    code = base64_hash[:4].decode('utf-8')
    return code


def parse_code(template):
    # Check that the code is 4 letters
    code = Path(template).stem
    with open(template, "r") as f:
        for line in f:
            if line.startswith("_entry.id"):
                code = line.split()[1]

    # Generate a deterministic 4-character code if needed
    if len(code) != 4:
        code = generate_code(code)

    return code.lower()


def create_dir_and_remove_files(dir_path, files_to_remove=[]):
    try:
        Path(dir_path).mkdir(parents=True)
    except FileExistsError:
        logging.info(f"{dir_path} already exists!")
        logging.info("The existing database will be overwritten!")
        for f in files_to_remove:
            target_file = dir_path / Path(f)
            if target_file.exists():
                target_file.unlink()


def create_tree(pdb_mmcif_dir, mmcif_dir, seqres_path, templates_dir):
    """
    Create the db structure with empty directories
    o pdb_mmcif_dir - path to the output directory
    o mmcif_dir - path to the mmcif directory
    o seqres_path - path to the pdb_seqres.txt file (not a directory)
    o templates_dir - path to the directory with all-chain templates in mmcif format
    Returns:
        o None
    """
    if Path(pdb_mmcif_dir).exists():
        files_to_remove = os.listdir(pdb_mmcif_dir)
    else:
        files_to_remove = []
    create_dir_and_remove_files(mmcif_dir, files_to_remove)
    create_dir_and_remove_files(templates_dir)

    # Create empty obsolete.dat file
    with open(pdb_mmcif_dir / 'obsolete.dat', 'a'):
        pass

    # Create empty pdb_seqres.txt file at the correct location
    seqres_path.parent.mkdir(parents=True, exist_ok=True)
    with open(seqres_path, 'a'):
        pass


def copy_file_exclude_lines(starting_with, src, dst):
    """
    Copy contents from src to dst excluding lines that start with `starting_with`.

    o starting_with: A string that, if a line starts with it, the line will be excluded.
    o src: Source file path.
    o dst: Destination file path.
    """
    with open(src, 'r') as infile, open(dst, 'w') as outfile:
        for line in infile:
            if not line.startswith(starting_with):
                outfile.write(line)

def _prepare_template(template, code, chain_id, mmcif_dir, seqres_path, templates_dir,
                      threshold_clashes, hb_allowance, plddt_threshold, number_of_templates):
    """
    Process and prepare each template.
    """
    duplicate = number_of_templates == 1
    new_template = templates_dir / Path(code + Path(template).suffix)
    copy_file_exclude_lines('HETATM', template, new_template)
    logging.info(f"Processing template: {new_template}  Chain {chain_id}")

    # Convert to (our) mmcif object
    mmcif_obj = MmcifChainFiltered(new_template, code, chain_id)
    # Determine the full sequence
    seqres = mmcif_obj.sequence_seqres if mmcif_obj.sequence_seqres else mmcif_obj.sequence_atom
    sqrres_path = save_seqres(code, chain_id, seqres, seqres_path, duplicate)
    logging.info(f"SEQRES saved to {sqrres_path}!")

    # Remove clashes and low pLDDT regions for each template
    mmcif_obj.remove_clashes(threshold_clashes, hb_allowance)
    mmcif_obj.remove_low_plddt(plddt_threshold)

    # Convert to Protein and mmCIF format
    protein = _from_bio_structure(mmcif_obj.structure)
    sequence_ids = mmcif_obj.atom_site_label_seq_ids

    # Save to file and validate
    codes_to_process = [f"{code[:-1]}{i}" for i in range(1, 5)] if duplicate else [code]
    for temp_code in codes_to_process:
        mmcif_string = to_mmcif(protein, f"{temp_code}_{chain_id}", "Monomer", chain_id, seqres, sequence_ids)
        fn = mmcif_dir / f"{temp_code}.cif"
        with open(fn, 'w') as f:
            f.write(mmcif_string)
        validate_and_fix_mmcif(fn)
        logging.info(f'{new_template} processed with code {temp_code}!')


def create_db(out_path, templates, chains, threshold_clashes, hb_allowance, plddt_threshold):
    """
    Main function that creates a custom template database for AlphaFold2
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
    out_path = Path(out_path)
    # Create the database structure
    pdb_mmcif_dir = out_path / 'pdb_mmcif'
    mmcif_dir = pdb_mmcif_dir / 'mmcif_files'
    seqres_path = out_path / 'pdb_seqres.txt'
    templates_dir = out_path / 'templates'

    create_tree(pdb_mmcif_dir, mmcif_dir, seqres_path, templates_dir)

    # Process each template/chain pair
    for template, chain_id in zip(templates, chains):
        template=Path(template)
        code = parse_code(template)
        logging.info(f"Template code: {code}")
        assert len(code) == 4
        _prepare_template(
            template, code, chain_id, mmcif_dir, seqres_path, templates_dir,
            threshold_clashes, hb_allowance, plddt_threshold, len(templates)
        )


def main(argv):
    flags.FLAGS(argv)
    create_db(flags.FLAGS.out_path, [flags.FLAGS.template], [flags.FLAGS.multimeric_chain],
              flags.FLAGS.threshold_clashes, flags.FLAGS.hb_allowance, flags.FLAGS.plddt_threshold)


if __name__ == '__main__':
    flags.DEFINE_string("out_path", None, "Path to the output directory")
    flags.DEFINE_string("template", None, "Path to the template mmCIF/PDB file")
    flags.DEFINE_string("multimeric_chain", None, "Chain ID of the multimeric template")
    flags.DEFINE_float("threshold_clashes", 1000, "Threshold for VDW overlap to identify clashes "
                                                  "(default: 1000, i.e. no threshold, for thresholding, use 0.9)")
    flags.DEFINE_float("hb_allowance", 0.4, "Allowance for hydrogen bonding (default: 0.4)")
    flags.DEFINE_float("plddt_threshold", 0, "Threshold for pLDDT score (default: 0)")
    flags.mark_flags_as_required(["out_path", "template", "multimeric_chain"])
    app.run(main)
