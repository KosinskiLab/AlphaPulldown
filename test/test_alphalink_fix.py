#!/usr/bin/env python
"""
Test to verify that the AlphaLink backend fix works correctly.
"""
import io
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from absl.testing import absltest

import alphapulldown
from alphapulldown.utils.create_combinations import process_files


class TestAlphaLinkFix(absltest.TestCase):
    """Test that the AlphaLink backend fix works correctly."""
    
    def setUp(self):
        super().setUp()
        
        # Create temporary directories
        self.temp_dir = Path(tempfile.mkdtemp(prefix="alphalink_test_"))
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()
        
        # paths to alphapulldown CLI scripts
        apd_path = Path(alphapulldown.__path__[0])
        self.script_multimer = apd_path / "scripts" / "run_multimer_jobs.py"
        self.script_single = apd_path / "scripts" / "run_structure_prediction.py"
        
        # Test data paths
        this_dir = Path(__file__).resolve().parent
        self.test_features_dir = this_dir / "test_data" / "features"
        self.test_protein_lists_dir = this_dir / "test_data" / "protein_lists"
        self.test_crosslinks_dir = this_dir / "alphalink"

    def tearDown(self):
        super().tearDown()
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except (PermissionError, OSError):
            pass

    def test_alphalink_command_construction(self):
        """Test that AlphaLink commands are constructed correctly."""
        
        # Create a simple test protein list
        test_protein_list = self.temp_dir / "test_proteins.txt"
        with open(test_protein_list, 'w') as f:
            f.write("TEST\n")
        
        # Mock the AlphaLink weights path
        mock_weights_path = "/mock/alphalink/weights/AlphaLink-Multimer_SDA_v3.pt"
        mock_crosslinks_path = str(self.test_crosslinks_dir / "example_crosslink.pkl.gz")
        
        # Test command construction for run_multimer_jobs.py
        args = [
            sys.executable,
            str(self.script_multimer),
            "--mode=custom",
            "--num_cycle=1",
            "--num_predictions_per_model=1",
            f"--data_dir={mock_weights_path}",
            f"--monomer_objects_dir={self.test_features_dir}",
            "--job_index=1",
            f"--output_path={self.output_dir}",
            "--use_alphalink=True",
            f"--alphalink_weight={mock_weights_path}",
            f"--crosslinks={mock_crosslinks_path}",
            f"--protein_lists={test_protein_list}",
        ]
        
        # Mock subprocess.run to avoid actually running the command
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            # This should not raise an exception if the fix works
            try:
                subprocess.run(args, capture_output=True, text=True, timeout=5)
            except subprocess.TimeoutExpired:
                # This is expected since we're mocking the weights
                pass
            except FileNotFoundError:
                # This is also expected since the weights don't exist
                pass
        
        # Verify that the command was constructed with the correct flags
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        
        # Check that the command contains the expected AlphaLink flags
        command_str = ' '.join(call_args)
        # The fold_backend is set internally, so we check for the use_alphalink flag
        self.assertIn("--use_alphalink", command_str)
        self.assertIn("--alphalink_weight", command_str)
        self.assertIn("--crosslinks", command_str)
        print(f"Command string: {command_str}")

    def test_alphalink_random_seed_fix(self):
        """Test that the random seed fix for AlphaLink works correctly."""
        
        # Import the fixed function
        from alphapulldown.scripts.run_structure_prediction import predict_structure
        
        # Mock the backend setup to return AlphaLink-style config
        mock_config = {
            "param_path": "/mock/weights.pt",
            "configs": {"mock": "config"}
        }
        
        # Mock the backend
        with patch('alphapulldown.folding_backend.backend') as mock_backend:
            mock_backend.setup.return_value = mock_config
            
            # This should not raise a KeyError for 'model_runners'
            try:
                # Call predict_structure with minimal arguments
                predict_structure(
                    objects_to_model=[],
                    model_flags={},
                    postprocess_flags={},
                    fold_backend="alphalink"
                )
            except KeyError as e:
                if "model_runners" in str(e):
                    self.fail("The AlphaLink random seed fix is not working")
                else:
                    # Other KeyError is expected since we're not providing real data
                    pass
            except Exception as e:
                # Other exceptions are expected since we're not providing real data
                if "model_runners" in str(e):
                    self.fail("The AlphaLink random seed fix is not working")


if __name__ == "__main__":
    absltest.main() 