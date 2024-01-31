#!/usr/bin/env python3

from absl import app, flags, logging
from Bio.PDB import PDBParser, Superimposer
from Bio.Align import PairwiseAligner
from concurrent.futures import ThreadPoolExecutor
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('reference_pdb', None, 'Path to the reference PDB file')
flags.DEFINE_string('target_pdb', None, 'Path to the target PDB file')

three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def extract_sequence_from_atoms(atom_lines):
    sequence = []
    current_residue = None
    for line in atom_lines:
        res_name = line[17:20].strip()
        res_seq = int(line[22:26].strip())
        if current_residue != res_seq:
            sequence.append(three_to_one.get(res_name, 'X'))
            current_residue = res_seq
    return ''.join(sequence)

def align_sequences(seq1, seq2):
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5
    aligner.mismatch_score = -0.5
    aligner.match_score = 1
    alignment = aligner.align(seq1, seq2)[0]
    return alignment

def map_alignment_to_atoms(chain, alignment, seq_idx_offset):
    mapped_atoms = []
    seq_idx = seq_idx_offset
    for res in chain:
        if res.id[0] == ' ' and seq_idx < len(alignment) and alignment[seq_idx] != '-':
            atoms = [atom for atom in res if atom.get_name() in ['N', 'CA', 'C', 'O']]
            if not atoms:
                logging.warning(f"No backbone atoms found for residue {res.get_resname()} {res.id} in chain {chain.id}")
            else:
                logging.info(f"Adding {len(atoms)} atoms for residue {res.get_resname()} {res.id} in chain {chain.id}")
                mapped_atoms.extend(atoms)
        else:
            if res.id[0] == ' ':
                logging.debug(f"Skipping residue {res.get_resname()} {res.id} in chain {chain.id} due to alignment gap or sequence mismatch")
        seq_idx += 1
    return mapped_atoms


def process_chain(chain_id, ref_structure, target_structure, ref_atoms, target_atoms):
    logging.info(f"Processing chain {chain_id.id}")
    try:
        ref_chain = ref_structure[0][chain_id.id]
        target_chain = target_structure[0][chain_id.id]
    except KeyError:
        logging.warning(f"Chain {chain_id.id} not found in one of the structures.")
        return [], []

    ref_seq = extract_sequence_from_atoms(ref_atoms)
    target_seq = extract_sequence_from_atoms(target_atoms)
    logging.info(f"Reference sequence: {ref_seq}")
    logging.info(f"Target sequence: {target_seq}")

    alignment = align_sequences(ref_seq, target_seq)
    logging.info(f"Alignment: {alignment}")

    coordinates = np.array(alignment.coordinates).transpose()
    start_ref, end_ref = coordinates[0]
    start_target, end_target = coordinates[-1]

    ref_atoms = map_alignment_to_atoms(ref_chain, str(alignment.aligned[0][0]), start_ref)
    target_atoms = map_alignment_to_atoms(target_chain, str(alignment.aligned[1][0]), start_target)

    if len(ref_atoms) != len(target_atoms):
        logging.warning(f"Number of atoms in chain {chain_id.id} differ between reference and target. Ref: {len(ref_atoms)}, Target: {len(target_atoms)}")
        return ref_atoms, target_atoms  # Returning the atoms regardless of the count mismatch for further analysis

    return ref_atoms, target_atoms

def calculate_rmsd(reference_pdb, target_pdb):
    parser = PDBParser(QUIET=True)
    ref_structure = parser.get_structure('reference', reference_pdb)
    target_structure = parser.get_structure('target', target_pdb)

    with open(reference_pdb, 'r') as file:
        ref_atoms = [line for line in file if line.startswith("ATOM")]
    with open(target_pdb, 'r') as file:
        target_atoms = [line for line in file if line.startswith("ATOM")]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chain, chain_id, ref_structure, target_structure, ref_atoms, target_atoms) for chain_id in ref_structure[0]]
        results = [future.result() for future in futures]

    combined_ref_atoms = [atom for ref_atoms, _ in results for atom in ref_atoms]
    combined_target_atoms = [atom for _, target_atoms in results for atom in target_atoms]

    if not combined_ref_atoms or not combined_target_atoms:
        logging.error("No suitable atoms found for RMSD calculation.")
        return None

    superimposer = Superimposer()
    superimposer.set_atoms(combined_ref_atoms, combined_target_atoms)
    superimposer.apply(combined_target_atoms)

    return superimposer.rms

def main(argv):
    logging.info(f"Reference PDB: {FLAGS.reference_pdb}")
    logging.info(f"Target PDB: {FLAGS.target_pdb}")

    rmsd = calculate_rmsd(FLAGS.reference_pdb, FLAGS.target_pdb)
    if rmsd is not None:
        print(f"Overall RMSD: {rmsd}")
    else:
        logging.error("Failed to calculate RMSD.")

if __name__ == '__main__':
    app.run(main)
