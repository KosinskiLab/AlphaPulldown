import pytest
from alphapulldown.create_fake_template_db import create_db
import tempfile
import os
from alphafold.data import mmcif_parsing
from pathlib import Path

def run_test(pdb_templates, chains):
    threshold_clashes = 1000
    hb_allowance = 0.4
    plddt_threshold = 0

    with tempfile.TemporaryDirectory() as tmpdirname:
        create_db(
            Path(tmpdirname), pdb_templates, chains,
            threshold_clashes, hb_allowance, plddt_threshold
        )

        assert os.path.exists(f"{tmpdirname}/pdb_mmcif/obsolete.dat")
        assert os.path.exists(f"{tmpdirname}/pdb_seqres/pdb_seqres.txt")

        path_to_mmcif = Path(tmpdirname) / f"pdb_mmcif/mmcif_files/3l4q.cif"

        assert os.path.exists(path_to_mmcif)

        with open(path_to_mmcif, "r") as f:
            mmcif_string = f.read()
        parse_result = mmcif_parsing.parse(
                file_id="TEST",
                mmcif_string=mmcif_string,
                catch_all_errors=True)

        assert not parse_result.errors
        mmcif_object = parse_result.mmcif_object
        model = mmcif_object.structure
        assert len(model.child_dict) == 1
        assert chains[0] in model.child_dict
        assert chains[0] in mmcif_object.chain_to_seqres

def test_from_pdb():
    run_test(["./test/test_data/true_multimer/3L4Q.pdb"], ["C"])

def test_from_cif():
    run_test(["./test/test_data/true_multimer/3L4Q.cif"], ["A"])
