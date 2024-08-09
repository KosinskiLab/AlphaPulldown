import os
import alphapulldown
from alphapulldown.utils.remove_clashes_low_plddt import MmcifChainFiltered
from pathlib import Path

alphapulldown_dir = os.path.dirname(alphapulldown.__file__)
alphapulldown_dir = Path(alphapulldown_dir)
pdb_file = alphapulldown_dir / '..' / 'test' / 'test_data' / 'templates' / 'ranked_0.pdb'


def test_init():
    for chain in ['B', 'C']:
        mmcif_chain = MmcifChainFiltered(pdb_file, "TEST", chain)
        assert mmcif_chain.input_file_path == pdb_file.with_suffix('.cif')
        assert mmcif_chain.chain_id == chain


def test_eq():
    mmcif_chain1 = MmcifChainFiltered(pdb_file, "TEST", "B")
    mmcif_chain2 = MmcifChainFiltered(pdb_file, "TEST", "B")
    assert mmcif_chain1 == mmcif_chain2


def test_remove_clashes():
    for chain in ['B', 'C']:
        mmcif_chain = MmcifChainFiltered(pdb_file, "TEST", chain)
        initial_atoms = list(mmcif_chain.structure.get_atoms())

        mmcif_chain.remove_clashes()
        mmcif_chain.remove_low_plddt()
        final_atoms = list(mmcif_chain.structure.get_atoms())

        assert len(final_atoms) < len(initial_atoms)


def test_remove_low_plddt():
    for chain in ['B', 'C']:
        mmcif_chain = MmcifChainFiltered(pdb_file, "TEST", chain)
        initial_atoms = list(mmcif_chain.structure.get_atoms())

        mmcif_chain.remove_low_plddt()
        final_atoms = list(mmcif_chain.structure.get_atoms())

        assert len(final_atoms) < len(initial_atoms)

#TODO: Add more tests
