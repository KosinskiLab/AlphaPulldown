#!/usr/bin/env python3

import logging
from Bio.PDB import PDBParser, Superimposer
from Bio.Align import PairwiseAligner
from Bio.PDB.Polypeptide import three_to_one
from concurrent.futures import ThreadPoolExecutor
from absl import app, flags

flags.DEFINE_string('reference_pdb', None, 'Path to the reference PDB file')
flags.DEFINE_string('target_pdb', None, 'Path to the target PDB file')

FLAGS = flags.FLAGS

logging.basicConfig(level=logging.INFO)

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

def get_ca_atoms_based_on_alignment(chain, aligned_sequence):
    ca_atoms = []
    chain_seq_index = 0
    aligned_seq_index = 0  # Index to track position in aligned_sequence

    for res in chain.get_residues():
        if res.id[0] == ' ' and 'CA' in res:
            if aligned_seq_index < len(aligned_sequence) and aligned_sequence[aligned_seq_index] != '-':
                ca_atoms.append(res['CA'])
            chain_seq_index += 1
            if aligned_seq_index < len(aligned_sequence):
                aligned_seq_index += 1

    return ca_atoms


def process_chain(chain_id, ref_structure, target_structure):
    logging.info(f"Processing chain {chain_id}")
    ref_chain = ref_structure[0][chain_id]
    target_chain = target_structure[0][chain_id]

    ref_sequence = extract_ca_sequence_from_pdb(ref_chain)
    target_sequence = extract_ca_sequence_from_pdb(target_chain)

    alignment = align_sequences(ref_sequence, target_sequence)
    logging.debug(f"Alignment: \n{alignment}")

    ref_ca_atoms = get_ca_atoms_based_on_alignment(ref_chain, str(alignment.aligned[0][0]))
    target_ca_atoms = get_ca_atoms_based_on_alignment(target_chain, str(alignment.aligned[1][0]))

    return ref_ca_atoms, target_ca_atoms

def calculate_rmsd(reference_pdb, target_pdb):
    parser = PDBParser(QUIET=True)
    ref_structure = parser.get_structure('reference', reference_pdb)
    target_structure = parser.get_structure('target', target_pdb)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chain, chain.id, ref_structure, target_structure) for chain in ref_structure[0].get_chains()]
        results = [future.result() for future in futures]

    combined_ref_ca_atoms = [atom for ref_atoms, _ in results for atom in ref_atoms]
    combined_target_ca_atoms = [atom for _, target_atoms in results for atom in target_atoms]

    if not combined_ref_ca_atoms or not combined_target_ca_atoms:
        logging.error("No suitable CA atoms found for RMSD calculation.")
        return None

    superimposer = Superimposer()
    superimposer.set_atoms(combined_ref_ca_atoms, combined_target_ca_atoms)
    superimposer.apply(combined_target_ca_atoms)

    return superimposer.rms

def main(argv):
    logging.info(f"Reference PDB: {FLAGS.reference_pdb}")
    logging.info(f"Target PDB: {FLAGS.target_pdb}")

    if FLAGS.reference_pdb and FLAGS.target_pdb:
        rmsd = calculate_rmsd(FLAGS.reference_pdb, FLAGS.target_pdb)
        if rmsd is not None:
            print(f"Overall RMSD using CA atoms: {rmsd}")
        else:
            logging.error("Failed to calculate RMSD using CA atoms.")
    else:
        logging.error("Reference and Target PDB paths must be provided.")

if __name__ == '__main__':
    app.run(main)
