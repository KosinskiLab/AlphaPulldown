#!/usr/bin/env python
"""
Functional Alphapulldown tests for AlphaFold3 (parameterised).

The script is identical for Slurm and workstation users – only the
wrapper decides *how* each case is executed.
"""
from __future__ import annotations
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import shutil
import argparse

from absl.testing import absltest, parameterized

import alphapulldown
from alphapulldown.utils.create_combinations import process_files


# --------------------------------------------------------------------------- #
#                       configuration / environment guards                    #
# --------------------------------------------------------------------------- #
# Point to the full Alphafold database once, via env-var.
DATA_DIR = os.getenv(
    "ALPHAFOLD_DATA_DIR",
    "/g/alphafold/AlphaFold_DBs/3.0.0"   #  default for EMBL cluster
)
if not os.path.exists(DATA_DIR):
    absltest.skip("set $ALPHAFOLD_DATA_DIR to run Alphafold functional tests")


# --------------------------------------------------------------------------- #
#                       common helper mix-in / assertions                     #
# --------------------------------------------------------------------------- #
class _TestBase(parameterized.TestCase):
    use_temp_dir = False  # Class variable to control directory behavior

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Create a base directory for all test outputs
        if cls.use_temp_dir:
            cls.base_output_dir = Path(tempfile.mkdtemp(prefix="af3_test_"))
        else:
            cls.base_output_dir = Path("test/test_data/predictions/af3_backend")
            if cls.base_output_dir.exists():
                shutil.rmtree(cls.base_output_dir)
            cls.base_output_dir.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        super().setUp()

        # directories inside the repo (relative to this file)
        this_dir = Path(__file__).resolve().parent
        self.test_data_dir = this_dir / "test_data"
        self.test_fastas_dir = self.test_data_dir / "fastas"
        self.test_features_dir = self.test_data_dir / "features"
        self.test_protein_lists_dir = this_dir / "test_data" / "protein_lists"
        self.test_templates_dir = this_dir / "test_data" / "templates"
        self.test_modelling_dir = this_dir / "test_data" / "predictions"

        # Create a unique output directory for this test
        test_name = self._testMethodName
        self.output_dir = self.base_output_dir / test_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # paths to alphapulldown CLI scripts
        apd_path = Path(alphapulldown.__path__[0])
        self.script_multimer = apd_path / "scripts" / "run_multimer_jobs.py"
        self.script_single = apd_path / "scripts" / "run_structure_prediction.py"

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        # Clean up all test outputs after all tests are done
        if cls.use_temp_dir and cls.base_output_dir.exists():
            shutil.rmtree(cls.base_output_dir)

    # ---------------- assertions reused by all subclasses ----------------- #
    def _runCommonTests(self, res: subprocess.CompletedProcess):
        print(res.stdout)
        print(res.stderr)
        self.assertEqual(res.returncode, 0, "sub-process failed")

        # Look in the parent directory for output files
        files = list(self.output_dir.iterdir())
        print(f"contents of {self.output_dir}: {[f.name for f in files]}")

        # Check for AlphaFold3 output files
        # 1. Main output files
        self.assertIn("TERMS_OF_USE.md", {f.name for f in files})
        self.assertIn("ranking_scores.csv", {f.name for f in files})
        
        # 2. Data and confidence files
        data_files = [f for f in files if f.name.endswith("_data.json")]
        conf_files = [f for f in files if f.name.endswith("_confidences.json")]
        summary_conf_files = [f for f in files if f.name.endswith("_summary_confidences.json")]
        model_files = [f for f in files if f.name.endswith("_model.cif")]
        
        self.assertTrue(len(data_files) > 0, "No data.json files found")
        self.assertTrue(len(conf_files) > 0, "No confidences.json files found")
        self.assertTrue(len(summary_conf_files) > 0, "No summary_confidences.json files found")
        self.assertTrue(len(model_files) > 0, "No model.cif files found")

        # 3. Check sample directories
        sample_dirs = [f for f in files if f.is_dir() and f.name.startswith("seed-")]
        self.assertEqual(len(sample_dirs), 5, "Expected 5 sample directories")

        for sample_dir in sample_dirs:
            sample_files = list(sample_dir.iterdir())
            self.assertIn("confidences.json", {f.name for f in sample_files})
            self.assertIn("model.cif", {f.name for f in sample_files})
            self.assertIn("summary_confidences.json", {f.name for f in sample_files})

        # 4. Verify ranking scores
        with open(self.output_dir / "ranking_scores.csv") as f:
            lines = f.readlines()
            self.assertTrue(len(lines) > 1, "ranking_scores.csv should have header and data")
            self.assertEqual(len(lines[0].strip().split(",")), 3, "ranking_scores.csv should have 3 columns")

    # convenience builder
    def _args(self, *, plist, mode, script):
        if script == "run_structure_prediction.py":
            # Format from run_multimer_jobs.py input to run_structure_prediction.py input
            buffer = io.StringIO()
            _ = process_files(
                input_files=[str(self.test_protein_lists_dir / plist)],
                output_path=buffer,
                exclude_permutations = True
            )
            buffer.seek(0)
            formatted_input_lines = [x.strip().replace(",", ":").replace(";", "+") for x in buffer.readlines() if x.strip()]
            # Use the first non-empty line as the input string
            formatted_input = formatted_input_lines[0] if formatted_input_lines else ""
            args = [
                sys.executable,
                str(self.script_single),
                f"--input={formatted_input}",
                f"--output_directory={self.output_dir}",
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={self.test_features_dir}",
                "--fold_backend=alphafold3",
                "--flash_attention_implementation=xla",
            ]
            return args
        elif script == "run_multimer_jobs.py":
            args = [
                sys.executable,
                str(self.script_multimer),
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_dir={DATA_DIR}",
                f"--monomer_objects_dir={self.test_features_dir}",
                "--job_index=1",
                f"--output_path={self.output_dir}",
                f"--mode={mode}",
                "--oligomer_state_file"
                if mode == "homo-oligomer"
                else "--protein_lists"
                + f"={self.test_protein_lists_dir / plist}",
            ]
            return args


# --------------------------------------------------------------------------- #
#                        parameterised "run mode" tests                       #
# --------------------------------------------------------------------------- #
class TestAlphaFold3RunModes(_TestBase):
    @parameterized.named_parameters(
        dict(testcase_name="monomer", protein_list="test_monomer.txt", mode="custom", script="run_structure_prediction.py"),
        dict(testcase_name="dimer", protein_list="test_dimer.txt", mode="custom", script="run_structure_prediction.py"),
        dict(testcase_name="trimer", protein_list="test_trimer.txt", mode="custom", script="run_structure_prediction.py"),
        dict(testcase_name="homo_oligomer", protein_list="test_homooligomer.txt", mode="homo-oligomer", script="run_structure_prediction.py"),
        dict(testcase_name="chopped_dimer", protein_list="test_dimer_chopped.txt", mode="custom", script="run_structure_prediction.py"),
        dict(testcase_name="long_name", protein_list="test_long_name.txt", mode="custom", script="run_structure_prediction.py"),
        # Test cases for combining AlphaPulldown monomer with different JSON inputs
        dict(
            testcase_name="monomer_with_dna", 
            protein_list="test_monomer.txt", 
            mode="custom", 
            script="run_structure_prediction.py"
        ),
        dict(
            testcase_name="monomer_with_rna", 
            protein_list="test_monomer_with_rna.txt", 
            mode="custom", 
            script="run_structure_prediction.py"
        ),
        dict(
            testcase_name="monomer_with_ligand", 
            protein_list="test_monomer.txt", 
            mode="custom", 
            script="run_structure_prediction.py"
        ),
    )
    def test_(self, protein_list, mode, script):
        # Create environment with GPU settings
        env = os.environ.copy()
        env["XLA_FLAGS"] = "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_force_compilation_parallelism=0"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        env["XLA_CLIENT_MEM_FRACTION"] = "0.95"
        env["JAX_FLASH_ATTENTION_IMPL"] = "xla"
        # Remove deprecated variable if present
        if "XLA_PYTHON_CLIENT_MEM_FRACTION" in env:
            del env["XLA_PYTHON_CLIENT_MEM_FRACTION"]
        
        # Debug output
        print("\nEnvironment variables:")
        print(f"XLA_FLAGS: {env.get('XLA_FLAGS')}")
        print(f"XLA_PYTHON_CLIENT_PREALLOCATE: {env.get('XLA_PYTHON_CLIENT_PREALLOCATE')}")
        print(f"XLA_CLIENT_MEM_FRACTION: {env.get('XLA_CLIENT_MEM_FRACTION')}")
        print(f"JAX_FLASH_ATTENTION_IMPL: {env.get('JAX_FLASH_ATTENTION_IMPL')}")
        
        # Check GPU availability
        try:
            import jax
            print("\nJAX GPU devices:")
            print(jax.devices())
            print("JAX GPU local devices:")
            print(jax.local_devices(backend='gpu'))
        except Exception as e:
            print(f"\nError checking JAX GPU: {e}")
        
        res = subprocess.run(
            self._args(plist=protein_list, mode=mode, script=script),
            capture_output=True,
            text=True,
            env=env
        )
        self._runCommonTests(res)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AlphaFold3 tests')
    parser.add_argument('--use-temp-dir', action='store_true',
                      help='Use temporary directory for test outputs instead of test/test_data/predictions/af3_backend')
    args, remaining = parser.parse_known_args()
    
    # Set the use_temp_dir flag on the test class
    _TestBase.use_temp_dir = args.use_temp_dir
    
    # Remove the --use-temp-dir argument from sys.argv so it doesn't interfere with pytest
    if '--use-temp-dir' in sys.argv:
        sys.argv.remove('--use-temp-dir')
    
    absltest.main() 