#!/usr/bin/env python
"""
Functional Alphapulldown tests for AlphaFold3 (parameterised).

The script is identical for Slurm and workstation users – only the
wrapper decides *how* each case is executed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from absl.testing import absltest, parameterized

import alphapulldown


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
    def setUp(self):
        super().setUp()

        # directories inside the repo (relative to this file)
        this_dir = Path(__file__).resolve().parent
        self.test_data_dir = this_dir / "test_data"
        self.test_fastas_dir = self.test_data_dir / "fastas"
        self.test_features_dir = self.test_data_dir / "features"
        self.test_protein_lists_dir = self.test_data_dir / "protein_lists"
        self.test_templates_dir = self.test_data_dir / "templates"
        self.test_modelling_dir = self.test_data_dir / "predictions"

        # output dir – ephemeral
        self.output_dir = Path("test/test_data/predictions/af3_backend/")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # paths to alphapulldown CLI scripts
        apd_path = Path(alphapulldown.__path__[0])
        self.script_multimer = apd_path / "scripts" / "run_multimer_jobs.py"
        self.script_single = apd_path / "scripts" / "run_structure_prediction.py"

    # ---------------- assertions reused by all subclasses ----------------- #
    def _runCommonTests(self, res: subprocess.CompletedProcess, multimer: bool, dirname: str | None = None):
        print(res.stdout)
        print(res.stderr)
        self.assertEqual(res.returncode, 0, "sub-process failed")

        dirs = [dirname] if dirname else [
            d for d in os.listdir(self.output_dir) if Path(self.output_dir, d).is_dir()
        ]

        for d in dirs:
            folder = self.output_dir / d
            files = list(folder.iterdir())
            print(f"contents of {folder}: {[f.name for f in files]}")

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
            with open(folder / "ranking_scores.csv") as f:
                lines = f.readlines()
                self.assertTrue(len(lines) > 1, "ranking_scores.csv should have header and data")
                self.assertEqual(len(lines[0].strip().split(",")), 3, "ranking_scores.csv should have 3 columns")

    # convenience builder
    def _args(self, *, plist, mode, script, af3_input_json=None):
        if script == "run_multimer_jobs.py":
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
                "--fold_backend=alphafold3",
                (
                    "--oligomer_state_file"
                    if mode == "homo-oligomer"
                    else "--protein_lists"
                )
                + f"={self.test_protein_lists_dir / plist}",
            ]
            if af3_input_json:
                args.append(f"--af3_input_json={af3_input_json}")
            return args
        else:  # run_structure_prediction.py
            args = [
                sys.executable,
                str(self.script_single),
                "--input=A0A075B6L2:10:1-3:4-5:6-7:7-8",
                f"--output_directory={self.output_dir}",
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={self.test_features_dir}",
                "--fold_backend=alphafold3",
            ]
            if af3_input_json:
                args.append(f"--af3_input_json={af3_input_json}")
            return args


# --------------------------------------------------------------------------- #
#                        parameterised "run mode" tests                       #
# --------------------------------------------------------------------------- #
class TestAlphaFold3RunModes(_TestBase):
    @parameterized.named_parameters(
        dict(testcase_name="monomer", protein_list="test_monomer.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="monomer_with_rna", protein_list="test_monomer_with_rna.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="dimer", protein_list="test_dimer.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="dimer_with_rna", protein_list="test_dimer_with_rna.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="trimer", protein_list="test_trimer.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="homo_oligomer", protein_list="test_homooligomer.txt", mode="homo-oligomer", script="run_multimer_jobs.py"),
        dict(testcase_name="chopped_dimer", protein_list="test_dimer_chopped.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="long_name", protein_list="test_long_name.txt", mode="custom", script="run_structure_prediction.py"),
        # New test cases for mixing with test_input.json
        dict(
            testcase_name="monomer_with_json", 
            protein_list="test_monomer.txt", 
            mode="custom", 
            script="run_multimer_jobs.py",
            af3_input_json=str(Path(__file__).parent / "test_data/features/test_input.json")
        ),
        dict(
            testcase_name="dimer_with_json", 
            protein_list="test_dimer.txt", 
            mode="custom", 
            script="run_multimer_jobs.py",
            af3_input_json=str(Path(__file__).parent / "test_data/features/test_input.json")
        ),
        dict(
            testcase_name="protein_rna_complex", 
            protein_list="test_monomer.txt", 
            mode="custom", 
            script="run_multimer_jobs.py",
            af3_input_json=str(Path(__file__).parent / "test_data/features/test_protein_rna.json")
        ),
    )
    def test_(self, protein_list, mode, script, af3_input_json=None):
        multimer = "monomer" not in protein_list
        res = subprocess.run(
            self._args(plist=protein_list, mode=mode, script=script, af3_input_json=af3_input_json), 
            capture_output=True, 
            text=True
        )
        self._runCommonTests(res, multimer)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    absltest.main() 