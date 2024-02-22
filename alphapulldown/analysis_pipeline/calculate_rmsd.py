#!/usr/bin/env python3

import logging
from Bio.PDB import PDBParser, Superimposer, PDBIO
from Bio.Align import PairwiseAligner
from Bio.PDB.Polypeptide import three_to_one
from absl import app, flags
from pathlib import Path

flags.DEFINE_string('reference_pdb', None, 'Path to the reference PDB file')
flags.DEFINE_string('target_pdb', None, 'Path to the target PDB file')

FLAGS = flags.FLAGS


def setup_logging():
    """Set up the logging format and level."""
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)


def extract_ca_sequence(structure):
    """Extracts the CA (alpha carbon) sequence from a PDB structure."""
    sequence = ''
    for res in structure.get_residues():
        if 'CA' in res:
            try:
                sequence += three_to_one(res.get_resname())
            except KeyError:
                sequence += '-'
    return sequence


def align_sequences(seq1, seq2):
    """Aligns two sequences and returns the best alignment."""
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -1
    aligner.mismatch_score = -1
    aligner.match_score = 2
    alignment = aligner.align(seq1, seq2)[0]
    return alignment


def get_common_atoms(ref_res, target_res):
    """Returns atoms common to both reference and target residues."""
    common_atoms = []
    for atom in ref_res:
        if atom.get_id() in target_res:
            common_atoms.append((atom, target_res[atom.get_id()]))
    return common_atoms


def process_chain(chain_id, ref_structure, target_structure, alignment):
    """Processes a single chain and extracts atoms for superposition."""
    ref_chain = ref_structure[0][chain_id]
    target_chain = target_structure[0][chain_id]
    ref_atoms, target_atoms = [], []

    ref_residues = [res for res in ref_chain.get_residues() if 'CA' in res]
    target_residues = [res for res in target_chain.get_residues() if 'CA' in res]

    for ref_res, target_res in zip(ref_residues, target_residues):
        common = get_common_atoms(ref_res, target_res)
        for ref_atom, target_atom in common:
            ref_atoms.append(ref_atom)
            target_atoms.append(target_atom)

    return ref_atoms, target_atoms


def calculate_rmsd_and_superpose(reference_pdb, target_pdb, temp_dir=None):
    """Calculates RMSD and superposes the target structure onto the reference."""
    parser = PDBParser(QUIET=True)
    ref_structure = parser.get_structure('reference', reference_pdb)
    target_structure = parser.get_structure('target', target_pdb)

    ref_sequence = extract_ca_sequence(ref_structure)
    target_sequence = extract_ca_sequence(target_structure)
    alignment = align_sequences(ref_sequence, target_sequence)

    combined_ref_atoms, combined_target_atoms = [], []
    for chain in ref_structure.get_chains():
        if chain.id in target_structure[0]:
            ref_atoms, target_atoms = process_chain(chain.id, ref_structure, target_structure, alignment)
            combined_ref_atoms.extend(ref_atoms)
            combined_target_atoms.extend(target_atoms)

    if not combined_ref_atoms or not combined_target_atoms:
        logging.error("No suitable atoms found for RMSD calculation.")
        return

    superimposer = Superimposer()
    superimposer.set_atoms(combined_ref_atoms, combined_target_atoms)
    superimposer.apply(target_structure.get_atoms())

    # Save the superimposed structures
    io = PDBIO()
    ref_structure_id = Path(reference_pdb).stem
    target_structure_id = Path(target_pdb).stem

    io.set_structure(ref_structure)
    io.set_structure(target_structure)

    if temp_dir:
        io.save(f"{temp_dir}/superposed_{ref_structure_id}.pdb")
        io.save(f"{temp_dir}/superposed_{target_structure_id}.pdb")
    else:
        io.save(f"superposed_{ref_structure_id}.pdb")
        io.save(f"superposed_{target_structure_id}.pdb")

    logging.info(f"RMSD between {reference_pdb} and {target_pdb}: {superimposer.rms:.4f}")
    return superimposer.rms


def main(argv):
    del argv  # Unused
    setup_logging()

    if not FLAGS.reference_pdb or not FLAGS.target_pdb:
        logging.error("Both reference and target PDB paths must be provided.")
        return

    calculate_rmsd_and_superpose(FLAGS.reference_pdb, FLAGS.target_pdb)


if __name__ == '__main__':
    app.run(main)
