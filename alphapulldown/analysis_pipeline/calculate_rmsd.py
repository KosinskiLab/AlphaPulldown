from absl import app, flags, logging
from Bio.PDB import PDBParser, Superimposer

FLAGS = flags.FLAGS

flags.DEFINE_string('reference_pdb', None, 'Path to the reference PDB file')
flags.DEFINE_string('target_pdb', None, 'Path to the target PDB file')
flags.DEFINE_string('chain_id', None, 'Chain ID to superimpose; if not provided, will try all chains')


def main(argv):
    # Initialize PDB parser and read files
    parser = PDBParser()
    ref_structure = parser.get_structure('reference', FLAGS.ref_pdb)
    target_structure = parser.get_structure('target', FLAGS.target_pdb)

    use_str = '-----------RTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDI---------------'

    if FLAGS.chain_id:
        chains_to_try = [FLAGS.chain_id]
    else:
        chains_to_try = [chain.id for chain in ref_structure[0]]

    for chain_id in chains_to_try:
        ref_chain = ref_structure[0][chain_id]
        target_chain = target_structure[0][chain_id]

        # Extract aligned residues
        ref_residues_aligned = [res for i, res in enumerate(ref_chain) if use_str[i] != '-']
        target_residues_aligned = [res for i, res in enumerate(target_chain) if use_str[i] != '-']

        # Extract atoms from those residues (you can customize which atoms e.g. CA, backbone atoms, etc.)
        ref_atoms_aligned = [atom for res in ref_residues_aligned for atom in res]
        target_atoms_aligned = [atom for res in target_residues_aligned for atom in res]

        # Superimpose
        superimposer = Superimposer()
        superimposer.set_atoms(ref_atoms_aligned, target_atoms_aligned)

        # Apply the transformation to target atoms
        superimposer.apply(target_atoms_aligned)

        # Print RMSD
        logging.info(f'RMSD for chain {chain_id}: {superimposer.rms}')


if __name__ == '__main__':
    flags.mark_flag_as_required('reference_pdb')
    flags.mark_flag_as_required('target_pdb')
    app.run(main)
