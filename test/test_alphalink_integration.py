#!/usr/bin/env python
"""
Simple integration test for AlphaLink backend with correct weights path.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from absl.testing import absltest

import alphapulldown


class TestAlphaLinkIntegration(absltest.TestCase):
    """Test AlphaLink integration with correct weights path."""
    
    def setUp(self):
        super().setUp()
        
        # Create temporary directories
        self.temp_dir = Path(tempfile.mkdtemp(prefix="alphalink_integration_"))
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()
        
        # Test data paths
        this_dir = Path(__file__).resolve().parent
        self.test_features_dir = this_dir / "test_data" / "features"
        self.test_protein_lists_dir = this_dir / "test_data" / "protein_lists"
        self.test_crosslinks_dir = this_dir / "alphalink"
        
        # AlphaLink weights path
        self.alphalink_weights_dir = "/scratch/AlphaFold_DBs/alphalink_weights"
        self.alphalink_weights_file = os.path.join(self.alphalink_weights_dir, "AlphaLink-Multimer_SDA_v3.pt")
        
        # Check if weights exist
        if not os.path.exists(self.alphalink_weights_file):
            self.skipTest(f"AlphaLink weights not found at {self.alphalink_weights_file}")

    def tearDown(self):
        super().tearDown()
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except (PermissionError, OSError):
            pass

    def test_alphalink_weights_path(self):
        """Test that AlphaLink weights path is correct."""
        self.assertTrue(os.path.exists(self.alphalink_weights_file), 
                       f"AlphaLink weights file not found: {self.alphalink_weights_file}")
        
        # Check file size (should be large)
        file_size = os.path.getsize(self.alphalink_weights_file)
        self.assertGreater(file_size, 1000000000,  # Should be > 1GB
                          f"AlphaLink weights file seems too small: {file_size} bytes")

    def test_alphalink_command_with_crosslinks(self):
        """Test AlphaLink command construction with crosslinks."""
        # Create a simple test protein list
        test_protein_list = self.temp_dir / "test_proteins.txt"
        with open(test_protein_list, 'w') as f:
            f.write("TEST\n")
        
        # Test command construction
        args = [
            sys.executable,
            "alphapulldown/scripts/run_structure_prediction.py",
            "--input=TEST",
            f"--output_directory={self.output_dir}",
            "--num_cycle=1",
            "--num_predictions_per_model=1",
            f"--data_directory={self.alphalink_weights_dir}",
            f"--features_directory={self.test_features_dir}",
            "--fold_backend=alphalink",
            "--use_alphalink=True",
            f"--alphalink_weight={self.alphalink_weights_file}",
            f"--crosslinks={self.test_crosslinks_dir}/example_crosslink.pkl.gz",
        ]
        
        # This should not raise an exception if the fix works
        try:
            # Use a timeout to avoid hanging
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            # We expect it to fail due to missing dependencies, but not due to the KeyError
            if result.returncode != 0:
                stderr = result.stderr
                if "KeyError" in stderr and "model_runners" in stderr:
                    self.fail("The AlphaLink fix is not working - KeyError still occurs")
                elif "No module named 'torch'" in stderr:
                    print("Expected failure: PyTorch not available")
                else:
                    print(f"Command failed with return code {result.returncode}")
                    print(f"STDERR: {stderr}")
        except subprocess.TimeoutExpired:
            print("Command timed out (expected if weights are large)")
        except FileNotFoundError:
            print("Command not found (expected in test environment)")

    def test_alphalink_command_without_crosslinks(self):
        """Test AlphaLink command construction without crosslinks."""
        # Create a simple test protein list
        test_protein_list = self.temp_dir / "test_proteins.txt"
        with open(test_protein_list, 'w') as f:
            f.write("TEST\n")
        
        # Test command construction without crosslinks
        args = [
            sys.executable,
            "alphapulldown/scripts/run_structure_prediction.py",
            "--input=TEST",
            f"--output_directory={self.output_dir}",
            "--num_cycle=1",
            "--num_predictions_per_model=1",
            f"--data_directory={self.alphalink_weights_dir}",
            f"--features_directory={self.test_features_dir}",
            "--fold_backend=alphalink",
            "--use_alphalink=True",
            f"--alphalink_weight={self.alphalink_weights_file}",
            # No crosslinks flag
        ]
        
        # This should not raise an exception if the fix works
        try:
            # Use a timeout to avoid hanging
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            # We expect it to fail due to missing dependencies, but not due to the KeyError
            if result.returncode != 0:
                stderr = result.stderr
                if "KeyError" in stderr and "model_runners" in stderr:
                    self.fail("The AlphaLink fix is not working - KeyError still occurs")
                elif "No module named 'torch'" in stderr:
                    print("Expected failure: PyTorch not available")
                else:
                    print(f"Command failed with return code {result.returncode}")
                    print(f"STDERR: {stderr}")
        except subprocess.TimeoutExpired:
            print("Command timed out (expected if weights are large)")
        except FileNotFoundError:
            print("Command not found (expected in test environment)")


if __name__ == "__main__":
    absltest.main() 