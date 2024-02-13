#!/usr/bin/env python3

import logging
from Bio.PDB import PDBParser, Superimposer, PDBIO
from Bio.Align import PairwiseAligner
from Bio.PDB.Polypeptide import three_to_one
from concurrent.futures import ThreadPoolExecutor
from absl import app, flags
from pathlib import Path

FLAGS = flags.FLAGS

flags.DEFINE_string('reference_pdb', None, 'Path to the reference PDB file')
flags.DEFINE_string('target_pdb', None, 'Path to the target PDB file')

logging.basicConfig(level=logging.DEBUG)

def extract_ca_sequence_from_pdb(structure):
    sequence = ''
    for res in structure.get_residues():
        if res.id[0] == ' ' and 'CA' in res:
            try:
                sequence += three_to_one(res.get_resname())
            except KeyError:
                sequence += '-'
    return sequence

def align_sequences(seq1, seq2):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -1
    aligner.mismatch_score = -1
    aligner.match_score = 2
    alignment = aligner.align(seq1, seq2)[0]
    return alignment

def get_common_atoms(ref_res, target_res):
    common_atoms = []
    ref_atoms = {atom.get_name(): atom for atom in ref_res}
    target_atoms = {atom.get_name(): atom for atom in target_res}
    for atom_name in ref_atoms:
        if atom_name in target_atoms:
            common_atoms.append((ref_atoms[atom_name], target_atoms[atom_name]))
    return common_atoms

def process_chain(chain_id, ref_structure, target_structure, alignment):
    ref_chain = ref_structure[0][chain_id]
    target_chain = target_structure[0][chain_id]
    ref_atoms = []
    target_atoms = []
    aligned_seq = str(alignment.aligned[0][0])
    ref_residues = list(ref_chain.get_residues())
    target_residues = list(target_chain.get_residues())
    ref_index, target_index = 0, 0
    for seq_char in aligned_seq:
        if seq_char != '-':
            ref_res, target_res = ref_residues[ref_index], target_residues[target_index]
            common = get_common_atoms(ref_res, target_res)
            for ref_atom, target_atom in common:
                ref_atoms.append(ref_atom)
                target_atoms.append(target_atom)
            target_index += 1
        ref_index += 1
    return ref_atoms, target_atoms

def calculate_rmsd_and_superpose(reference_pdb, target_pdb):
    parser = PDBParser(QUIET=True)
    ref_structure = parser.get_structure('reference', reference_pdb)
    target_structure = parser.get_structure('target', target_pdb)

    ref_sequence = extract_ca_sequence_from_pdb(ref_structure[0])
    target_sequence = extract_ca_sequence_from_pdb(target_structure[0])
    alignment = align_sequences(ref_sequence, target_sequence)
    logging.debug(f"Alignment: {alignment.aligned[0][0]}\n{alignment.aligned[0][1]}")

    combined_ref_atoms, combined_target_atoms = [], []
    for chain in ref_structure[0].get_chains():
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

    io = PDBIO()
    ref_structure_id = Path(reference_pdb).stem
    target_structure_id = Path(target_pdb).stem

    io.set_structure(ref_structure)
    io.save(f"superposed_{ref_structure_id}.pdb")

    io.set_structure(target_structure)
    io.save(f"superposed_{target_structure_id}.pdb")

    logging.info(f"RMSD between {reference_pdb} and {target_pdb}: {superimposer.rms}")

def main(argv):
    del argv  # Unused.
    if FLAGS.reference_pdb and FLAGS.target_pdb:
        calculate_rmsd_and_superpose(FLAGS.reference_pdb, FLAGS.target_pdb)
    else:
        logging.error("Reference and Target PDB paths must be provided.")

if __name__ == '__main__':
    app.run(main)
