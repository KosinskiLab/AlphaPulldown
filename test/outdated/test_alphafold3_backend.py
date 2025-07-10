import os
import json
import tempfile
import shutil
import pickle
from pathlib import Path
from absl.testing import parameterized, absltest
import logging
import numpy as np

from alphapulldown.folding_backend.alphafold3_backend import AlphaFold3Backend
from alphapulldown.objects import MonomericObject, MultimericObject
from alphafold3.common import folding_input

class TestAlphaFold3Backend(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        # Set up test directories
        this_dir = Path(__file__).resolve().parent
        self.test_data_dir = this_dir / "test_data"
        self.features_dir = self.test_data_dir / "features"
        
        # Create temporary directory for test output
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_output_dir = self.temp_dir / "test_output"
        self.test_output_dir.mkdir()
        
        # Create test input.json files
        self.af3_input_json = self.test_output_dir / "test_input.json"
        self.af3_server_json = self.test_output_dir / "test_server_input.json"
        
        # Load monomeric objects from pickle files
        with open(self.features_dir / "TEST.pkl", 'rb') as f:
            self.monomer1 = pickle.load(f)
            
        with open(self.features_dir / "A0A024R1R8.pkl", 'rb') as f:
            self.monomer2 = pickle.load(f)
            
        # Create multimer from the two monomers
        self.multimer = MultimericObject(
            interactors=[self.monomer1, self.monomer2],
            pair_msa=True,  # Enable MSA pairing for multimer
            multimeric_template=False  # No multimeric templates for this test
        )
            
        # Verify loaded objects
        self.assertIsInstance(self.monomer1, MonomericObject)
        self.assertIsInstance(self.monomer2, MonomericObject)
        self.assertIsInstance(self.multimer, MultimericObject)
        logging.info(f"Loaded monomer1: {self.monomer1.description}")
        logging.info(f"Loaded monomer2: {self.monomer2.description}")
        logging.info(f"Created multimer with {len(self.multimer.interactors)} interactors")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_test_input_json(self, dialect="alphafold3"):
        """Create test input.json files in different formats"""
        if dialect == "alphafold3":
            # Create AlphaFold3 format input.json
            input_data = {
                "dialect": "alphafold3",
                "version": 1,
                "name": "test_input",
                "modelSeeds": [42],
                "sequences": [{
                    "protein": {
                        "id": "C",  # Changed to "C" to avoid conflicts with multimer chains (A and B)
                        "sequence": self.monomer1.sequence,
                        "modifications": [],
                        "unpairedMsa": None,
                        "pairedMsa": None,
                        "templates": []
                    }
                }],
                "bondedAtomPairs": [],
                "userCCD": None
            }
            self.af3_input_json = self.test_output_dir / "test_input.json"
        else:
            # Create AlphaFold Server format input.json
            input_data = [{
                "name": "test_server_input",
                "modelSeeds": [42],
                "sequences": [{
                    "proteinChain": {
                        "sequence": self.monomer1.sequence,
                        "count": 1,
                        "modifications": []
                        # Removed glycans field as it's not supported
                    }
                }]
            }]
            self.af3_input_json = self.test_output_dir / "test_server_input.json"

        # Save to test_data/features directory
        features_dir = self.test_data_dir / "features"
        features_dir.mkdir(exist_ok=True)
        with open(features_dir / self.af3_input_json.name, 'w') as f:
            json.dump(input_data, f)

        # Also save to test output directory
        with open(self.af3_input_json, 'w') as f:
            json.dump(input_data, f)

    @parameterized.named_parameters(
        {
            "testcase_name": "alphafold3_format",
            "dialect": "alphafold3",
            "input_file": "test_input.json"
        },
        {
            "testcase_name": "alphafold_server_format",
            "dialect": "alphafold_server",
            "input_file": "test_server_input.json"
        }
    )
    def test_input_json_parsing(self, dialect, input_file):
        """Test parsing of different input.json formats"""
        self._create_test_input_json(dialect)
        
        # Test parsing
        prepared_inputs = AlphaFold3Backend.prepare_input(
            objects_to_model=[],
            random_seed=42,
            af3_input_json=[str(self.test_output_dir / input_file)]
        )
        
        self.assertEqual(len(prepared_inputs), 1)
        fold_input, output_dir = next(iter(prepared_inputs[0].items()))
        
        # Verify parsed content
        self.assertIsInstance(fold_input, folding_input.Input)
        self.assertEqual(len(fold_input.chains), 1)
        self.assertEqual(fold_input.chains[0].sequence, self.monomer1.sequence)

    def test_invalid_input_json(self):
        """Test handling of invalid input.json files"""
        # Test missing required fields
        invalid_json = {
            "dialect": "alphafold3",
            "version": "1.0",
            # Missing required fields
        }
        with open(self.af3_input_json, 'w') as f:
            json.dump(invalid_json, f)
            
        with self.assertRaises(ValueError) as cm:
            AlphaFold3Backend.prepare_input(
                objects_to_model=[],
                random_seed=42,
                af3_input_json=[str(self.af3_input_json)]
            )
        self.assertIn("Missing required fields", str(cm.exception))

        # Test invalid dialect
        invalid_json["dialect"] = "invalid"
        invalid_json["version"] = "1.0"
        invalid_json["name"] = "test"
        invalid_json["modelSeeds"] = [42]
        invalid_json["sequences"] = []
        invalid_json["bondedAtomPairs"] = []
        invalid_json["userCCD"] = None
        
        with open(self.af3_input_json, 'w') as f:
            json.dump(invalid_json, f)
            
        with self.assertRaises(ValueError) as cm:
            AlphaFold3Backend.prepare_input(
                objects_to_model=[],
                random_seed=42,
                af3_input_json=[str(self.af3_input_json)]
            )
        self.assertIn("Unsupported dialect", str(cm.exception))

    @parameterized.named_parameters(
        {
            "testcase_name": "monomer_with_json",
            "object_type": "monomer",
            "has_json": True
        },
        {
            "testcase_name": "multimer_with_json",
            "object_type": "multimer",
            "has_json": True
        },
        {
            "testcase_name": "monomer_only",
            "object_type": "monomer",
            "has_json": False
        },
        {
            "testcase_name": "multimer_only",
            "object_type": "multimer",
            "has_json": False
        }
    )
    def test_merge_objects(self, object_type, has_json):
        """Test merging AlphaPulldown objects with input.json"""
        # Create test objects
        if object_type == "monomer":
            objects_to_model = [{self.monomer1: str(self.test_output_dir / "monomer_output")}]
        else:
            objects_to_model = [{self.multimer: str(self.test_output_dir / "multimer_output")}]
            
        # Create input.json if needed
        if has_json:
            self._create_test_input_json()
            af3_input_json = [str(self.af3_input_json)]
        else:
            af3_input_json = None
            
        # Test merging
        prepared_inputs = AlphaFold3Backend.prepare_input(
            objects_to_model=objects_to_model,
            random_seed=42,
            af3_input_json=af3_input_json
        )
        
        self.assertEqual(len(prepared_inputs), 1)
        fold_input, output_dir = next(iter(prepared_inputs[0].items()))
        
        # Verify merged content
        expected_chains = 1 if object_type == "monomer" else len(self.multimer.interactors)
        if has_json:
            expected_chains += 1  # Add one chain from input.json
            
        self.assertEqual(len(fold_input.chains), expected_chains)
        
        # Verify chain IDs are unique
        chain_ids = {chain.id for chain in fold_input.chains}
        self.assertEqual(len(chain_ids), expected_chains)
        
        # Verify sequences match
        if object_type == "monomer":
            self.assertEqual(fold_input.chains[0].sequence, self.monomer1.sequence)
        else:
            for i, chain in enumerate(fold_input.chains[:len(self.multimer.interactors)]):
                self.assertEqual(chain.sequence, self.multimer.interactors[i].sequence)

    def test_empty_input(self):
        """Test handling of empty input"""
        prepared_inputs = AlphaFold3Backend.prepare_input(
            objects_to_model=[],
            random_seed=42
        )
        self.assertEqual(prepared_inputs, [])

if __name__ == '__main__':
    absltest.main() 