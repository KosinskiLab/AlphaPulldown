import pytest
from alphapulldown.create_custom_template_db import create_db, parse_code
import tempfile
import os
from alphafold.data import mmcif_parsing
from pathlib import Path
from Bio.PDB import MMCIF2Dict
from alphafold.data.mmcif_parsing import _get_atom_site_list, _get_protein_chains


def run_test(pdb_templates, chains):
    threshold_clashes = 1000
    hb_allowance = 0.4
    plddt_threshold = 0

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname) / "test_custom_db"
        create_db(
            tmpdirname, pdb_templates, chains,
            threshold_clashes, hb_allowance, plddt_threshold
        )

        assert os.path.exists(f"{tmpdirname}/pdb_mmcif/obsolete.dat")
        assert os.path.exists(f"{tmpdirname}/pdb_seqres/pdb_seqres.txt")
        # check that there are mmcif files
        mmcif_files = [f for f in os.listdir(f"{tmpdirname}/pdb_mmcif/mmcif_files") if f.endswith(".cif")]
        assert len(mmcif_files) > 0
        path_to_mmcif = f"{tmpdirname}/pdb_mmcif/mmcif_files/{mmcif_files[0]}"

        mmcif_dict = MMCIF2Dict.MMCIF2Dict(path_to_mmcif)
        valid_chains = _get_protein_chains(parsed_info= mmcif_dict)
        assert (chains[0] in valid_chains)

        with open(path_to_mmcif, "r") as f:
            mmcif_string = f.read()
        parse_result = mmcif_parsing.parse(
                file_id="TEST",
                mmcif_string=mmcif_string,
                catch_all_errors=True)

        assert not parse_result.errors
        mmcif_object = parse_result.mmcif_object
        model = mmcif_object.structure
        # check the chain
        assert len(model.child_dict) == 1
        assert chains[0] in model.child_dict
        assert chains[0] in mmcif_object.chain_to_seqres
        # check that the sequence is the same as the one in the pdb_seqres.txt
        with open(f"{tmpdirname}/pdb_seqres/pdb_seqres.txt", "r") as f:
            seqres_seq = f.readlines()[-1]
        assert mmcif_object.chain_to_seqres[chains[0]]+'\n' == seqres_seq
        # check there are atoms in the model
        atoms = list(model.child_dict[chains[0]].get_atoms())
        assert len(atoms) > 0
        # check seqres and atom label_id count are the same
        seqres_ids = [int(x+1) for x in mmcif_object.seqres_to_structure[chains[0]].keys()]
        atoms = _get_atom_site_list(mmcif_dict)
        for atom in atoms:
            if atom.mmcif_chain_id == chains[0] or atom.hetatm_atom:
                #print(f"Debug: atom.mmci_seq_num: {atom.mmcif_seq_num}")
                #print(f"Debug: atom.author_seq_num: {atom.author_seq_num}")
                assert int(atom.mmcif_seq_num) in seqres_ids


def test_from_pdb(capfd):
    run_test(["./test/test_data/true_multimer/3L4Q.pdb"], ["C"])

def test_from_cif(capfd):
    run_test(["./test/test_data/true_multimer/3L4Q.cif"], ["A"])

def test_from_af_output_pdb(capfd):
    run_test(["./test/test_data/true_multimer/cage_BC_AF.pdb"], ["B"])

def test_from_minimal_pdb(capfd):
    run_test(["./test/test_data/true_multimer/RANdom_name1_.7-1_0.pdb"], ["B"])