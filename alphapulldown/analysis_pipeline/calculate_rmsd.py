#!/usr/bin/env python3

from absl import app, flags, logging
from Bio.PDB import PDBParser, Superimposer


"""
Superimposes and calculates RMSD for two PDB files with identical number of atoms.
"""

FLAGS = flags.FLAGS

flags.DEFINE_string('reference_pdb', None, 'Path to the reference PDB file')
flags.DEFINE_string('target_pdb', None, 'Path to the target PDB file')
flags.DEFINE_string('chain_id', None, 'Chain ID to superimpose; if not provided, will try all chains')

def calculate_rmsd(reference_pdb, target_pdb, chain_id=None):
    parser = PDBParser()
    ref_structure = parser.get_structure('reference', reference_pdb)
    target_structure = parser.get_structure('target', target_pdb)

    if chain_id:
        chains_to_try = [chain_id]
    else:
        chains_to_try = [chain.id for chain in ref_structure[0]]

    rmsds = []
    for chain_id in chains_to_try:
        ref_chain = ref_structure[0][chain_id]
        target_chain = target_structure[0][chain_id]

        ref_residues = [res for i, res in enumerate(ref_chain)]
        target_residues = [res for i, res in enumerate(target_chain)]

        ref_atoms_aligned = [atom for res in ref_residues for atom in res]
        target_atoms_aligned = [atom for res in target_residues for atom in res]

        superimposer = Superimposer()
        superimposer.set_atoms(ref_atoms_aligned, target_atoms_aligned)

        superimposer.apply(target_atoms_aligned)

        logging.info(f'RMSD for chain {chain_id}: {superimposer.rms}')
        rmsds.append(superimposer.rms)
    return rmsds

def main(argv):
    rmsds = calculate_rmsd(FLAGS.reference_pdb, FLAGS.target_pdb, FLAGS.chain_id)
    print("RMSDs:", rmsds)

if __name__ == '__main__':
    flags.mark_flag_as_required('reference_pdb')
    flags.mark_flag_as_required('target_pdb')
    app.run(main)
