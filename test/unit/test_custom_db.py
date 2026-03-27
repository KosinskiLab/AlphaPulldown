import pytest
import logging
from alphapulldown.utils.create_custom_template_db import create_db
import tempfile
import os
from alphafold.data import mmcif_parsing
from pathlib import Path
from Bio.PDB import MMCIF2Dict
from alphafold.data.mmcif_parsing import _get_atom_site_list, _get_protein_chains

logger = logging.getLogger(__name__)

def run_test(pdb_templates, chains):
    threshold_clashes = 1000
    hb_allowance = 0.4
    plddt_threshold = 0

    logger.info(f"Testing custom DB creation with templates: {pdb_templates}, chains: {chains}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname) / "test_custom_db"
        logger.info(f"Creating custom DB in: {tmpdirname}")
        
        create_db(
            tmpdirname, pdb_templates, chains,
            threshold_clashes, hb_allowance, plddt_threshold
        )

        # Verify required files exist
        obsolete_dat_path = f"{tmpdirname}/pdb_mmcif/obsolete.dat"
        pdb_seqres_path = f"{tmpdirname}/pdb_seqres.txt"
        
        logger.info(f"Checking if {obsolete_dat_path} exists")
        assert os.path.exists(obsolete_dat_path), f"obsolete.dat not found at {obsolete_dat_path}"
        
        logger.info(f"Checking if {pdb_seqres_path} exists")
        assert os.path.exists(pdb_seqres_path), f"pdb_seqres.txt not found at {pdb_seqres_path}"
        
        # check that there are mmcif files
        mmcif_dir = f"{tmpdirname}/pdb_mmcif/mmcif_files"
        mmcif_files = [f for f in os.listdir(mmcif_dir) if f.endswith(".cif")]
        logger.info(f"Found {len(mmcif_files)} mmCIF files: {mmcif_files}")
        assert len(mmcif_files) > 0, f"No mmCIF files found in {mmcif_dir}"
        
        path_to_mmcif = f"{mmcif_dir}/{mmcif_files[0]}"
        logger.info(f"Testing mmCIF file: {path_to_mmcif}")

        mmcif_dict = MMCIF2Dict.MMCIF2Dict(path_to_mmcif)
        valid_chains = _get_protein_chains(parsed_info=mmcif_dict)
        logger.info(f"Valid chains in mmCIF: {valid_chains}")
        assert chains[0] in valid_chains, f"Chain {chains[0]} not found in valid chains {valid_chains}"

        with open(path_to_mmcif, "r") as f:
            mmcif_string = f.read()
        parse_result = mmcif_parsing.parse(
                file_id="TEST",
                mmcif_string=mmcif_string,
                catch_all_errors=True)

        if parse_result.errors:
            logger.error(f"mmCIF parsing errors: {parse_result.errors}")
        assert not parse_result.errors, f"mmCIF parsing failed: {parse_result.errors}"
        
        mmcif_object = parse_result.mmcif_object
        model = mmcif_object.structure
        
        # check the chain
        logger.info(f"Model has {len(model.child_dict)} chains: {list(model.child_dict.keys())}")
        assert len(model.child_dict) == 1, f"Expected 1 chain, found {len(model.child_dict)}"
        assert chains[0] in model.child_dict, f"Chain {chains[0]} not in model chains {list(model.child_dict.keys())}"
        assert chains[0] in mmcif_object.chain_to_seqres, f"Chain {chains[0]} not in chain_to_seqres"
        
        # check that the sequence is the same as the one in the pdb_seqres.txt
        with open(pdb_seqres_path, "r") as f:
            seqres_seq = f.readlines()[-1]
        expected_seq = mmcif_object.chain_to_seqres[chains[0]]+'\n'
        logger.info(f"Comparing sequences - mmCIF: {mmcif_object.chain_to_seqres[chains[0]]}, pdb_seqres: {seqres_seq.strip()}")
        assert expected_seq == seqres_seq, f"Sequence mismatch: expected {expected_seq}, got {seqres_seq}"
        
        # check there are atoms in the model
        atoms = list(model.child_dict[chains[0]].get_atoms())
        logger.info(f"Found {len(atoms)} atoms in chain {chains[0]}")
        assert len(atoms) > 0, f"No atoms found in chain {chains[0]}"
        
        # check seqres and atom label_id count are the same
        seqres_ids = [int(x+1) for x in mmcif_object.seqres_to_structure[chains[0]].keys()]
        atoms = _get_atom_site_list(mmcif_dict)
        logger.info(f"Checking {len(atoms)} atoms against {len(seqres_ids)} seqres IDs")
        
        for atom in atoms:
            if atom.mmcif_chain_id == chains[0] or atom.hetatm_atom:
                assert int(atom.mmcif_seq_num) in seqres_ids, f"Atom seq_num {atom.mmcif_seq_num} not in seqres_ids {seqres_ids}"
        
        logger.info("Custom DB test completed successfully")

def test_from_pdb():
    """Test custom DB creation from PDB file"""
    run_test(["./test/test_data/templates/3L4Q.pdb"], ["C"])

def test_from_cif():
    """Test custom DB creation from CIF file"""
    run_test(["./test/test_data/templates/3L4Q.cif"], ["A"])

def test_from_af_output_pdb():
    """Test custom DB creation from AlphaFold output PDB"""
    run_test(["./test/test_data/templates/ranked_0.pdb"], ["B"])

def test_from_minimal_pdb():
    """Test custom DB creation from minimal PDB file"""
    run_test(["./test/test_data/templates/RANdom_name1_.7-1_0.pdb"], ["B"])

def test_long_filename_generates_valid_code():
    """Test that generate_code produces only lowercase alphanumeric characters that can be parsed by AlphaFold"""
    from alphapulldown.utils.create_custom_template_db import generate_code, parse_code
    from alphafold.data.parsers import _parse_hmmsearch_description
    import tempfile
    from pathlib import Path
    
    # Test with a long filename that will trigger hash generation
    long_filename = "ranbp5_pb1_181_216_noxl_af3_dertemp_model.cif"
    template_path = Path("./test/test_data/templates") / long_filename
    
    if not template_path.exists():
        pytest.skip(f"Template file {template_path} not found")
    
    # Generate code from the long filename
    code = parse_code(template_path)
    logger.info(f"Generated code for long filename: {code}")
    
    # Verify code is 4 characters
    assert len(code) == 4, f"Code should be 4 characters, got {len(code)}"
    
    # Verify code contains only lowercase alphanumeric characters
    assert code.isalnum() and code.islower(), f"Code '{code}' should be lowercase alphanumeric"
    
    # Test that the code can be used in a pdb_seqres entry that AlphaFold can parse
    # The format is: >{code}_{chain} mol:protein length:{len}
    # But AlphaFold parser expects: >{code}_{chain}/{start}-{end} [subseq from] mol:protein length:{len}
    # However, we can at least verify the code part matches the regex pattern
    import re
    pdb_id_pattern = r'^[a-z0-9]+$'
    assert re.match(pdb_id_pattern, code), f"Code '{code}' does not match AlphaFold PDB ID pattern {pdb_id_pattern}"
    
    # Create a test entry and verify it can be parsed (with minimal required format)
    # We'll create a minimal valid entry that matches the parser's expectations
    test_chain = "A"
    test_length = 100
    test_start = 1
    test_end = 100
    test_entry = f">{code}_{test_chain}/{test_start}-{test_end} [subseq from] mol:protein length:{test_length}"
    
    try:
        metadata = _parse_hmmsearch_description(test_entry)
        logger.info(f"Successfully parsed entry: {test_entry}")
        assert metadata.pdb_id == code, f"Parsed PDB ID {metadata.pdb_id} does not match generated code {code}"
    except ValueError as e:
        pytest.fail(f"Generated code '{code}' cannot be parsed by AlphaFold parser: {e}")