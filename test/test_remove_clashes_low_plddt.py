import pytest
import os
import alphapulldown
from alphapulldown.remove_clashes_low_plddt import MmcifChainFiltered
from pathlib import Path

alphapulldown_dir = os.path.dirname(alphapulldown.__file__)
alphapulldown_dir = Path(alphapulldown_dir)
cif_file = alphapulldown_dir / '..' / 'test' / 'test_data' / 'true_multimer' / 'cage_BC_AF.cif'


def test_init():
    for chain in ['B', 'C']:
        mmcif_chain = MmcifChainFiltered(cif_file, "TEST", chain)
        assert mmcif_chain.input_file_path == cif_file
        assert mmcif_chain.chain_id == chain


def test_eq():
    mmcif_chain1 = MmcifChainFiltered(cif_file, "TEST", "B")
    mmcif_chain2 = MmcifChainFiltered(cif_file, "TEST", "B")
    assert mmcif_chain1 == mmcif_chain2


def test_remove_clashes():
    for chain in ['B', 'C']:
        mmcif_chain = MmcifChainFiltered(cif_file, "TEST", chain)
        initial_atoms = list(mmcif_chain.structure.get_atoms())

        mmcif_chain.remove_clashes()
        final_atoms = list(mmcif_chain.structure.get_atoms())

        assert len(final_atoms) < len(initial_atoms)


def test_remove_low_plddt():
    for chain in ['B', 'C']:
        mmcif_chain = MmcifChainFiltered(cif_file, "TEST", chain)
        initial_atoms = list(mmcif_chain.structure.get_atoms())

        mmcif_chain.remove_low_plddt()
        final_atoms = list(mmcif_chain.structure.get_atoms())

        assert len(final_atoms) < len(initial_atoms)

#TODO: Add more tests
