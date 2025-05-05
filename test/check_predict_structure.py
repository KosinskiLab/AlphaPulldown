#!/usr/bin/env python
"""
Functional Alphapulldown tests (parameterised).

The script is identical for Slurm and workstation users – only the
wrapper decides *how* each case is executed.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
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
    "/g/alphafold/AlphaFold_DBs/2.3.0"   #  default for EMBL cluster
)
if not os.path.exists(DATA_DIR):
    absltest.skip("set $ALPHAFOLD_DATA_DIR to run Alphafold functional tests")

# Toggle fast (1 Evoformer block) vs full inference with an env flag.
FAST = os.getenv("ALPHAFOLD_FAST", "1") != "0"
if FAST:
    from alphafold.model import config

    # shave the Evoformer
    config.CONFIG_MULTIMER.model.embeddings_and_evoformer.evoformer_num_block = 1


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
        self.tempdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tempdir.name)

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

            self.assertEqual(len([f for f in files if f.name.startswith("ranked") and f.suffix == ".pdb"]), 5)
            pkls = [f for f in files if f.name.startswith("result") and f.suffix == ".pkl"]
            self.assertEqual(len(pkls), 5)

            example = pickle.load(pkls[0].open("rb"))
            keys_multimer = {
                "experimentally_resolved",
                "predicted_aligned_error",
                "predicted_lddt",
                "structure_module",
                "plddt",
                "max_predicted_aligned_error",
                "seqs",
                "iptm",
                "ptm",
                "ranking_confidence",
            }
            keys_monomer = keys_multimer - {"iptm"}
            expected_keys = keys_multimer if multimer else keys_monomer
            self.assertTrue(expected_keys <= example.keys())

            # pae, png, timings, ranking_debug
            self.assertEqual(len([f for f in files if f.name.startswith("pae") and f.suffix == ".json"]), 5)
            self.assertEqual(len([f for f in files if f.suffix == ".png"]), 5)
            self.assertIn("ranking_debug.json", {f.name for f in files})
            self.assertIn("timings.json", {f.name for f in files})

            ranking = json.loads((folder / "ranking_debug.json").read_text())
            self.assertEqual(len(ranking["order"]), 5)

    # convenience builder
    def _args(self, *, plist, mode, script):
        if script == "run_multimer_jobs.py":
            return [
                sys.executable,
                str(self.script_multimer),
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_dir={DATA_DIR}",
                f"--monomer_objects_dir={self.test_features_dir}",
                "--job_index=1",
                f"--output_path={self.output_dir}",
                f"--mode={mode}",
                (
                    "--oligomer_state_file"
                    if mode == "homo-oligomer"
                    else "--protein_lists"
                )
                + f"={self.test_protein_lists_dir / plist}",
            ]
        else:  # run_structure_prediction.py
            return [
                sys.executable,
                str(self.script_single),
                "--input=A0A075B6L2:10:1-3:4-5:6-7:7-8",
                f"--output_directory={self.output_dir}",
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={self.test_features_dir}",
            ]


# --------------------------------------------------------------------------- #
#                        parameterised “run mode” tests                       #
# --------------------------------------------------------------------------- #
class TestRunModes(_TestBase):
    @parameterized.named_parameters(
        dict(testcase_name="monomer", protein_list="test_monomer.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="dimer", protein_list="test_dimer.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="trimer", protein_list="test_trimer.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="homo_oligomer", protein_list="test_homooligomer.txt", mode="homo-oligomer", script="run_multimer_jobs.py"),
        dict(testcase_name="chopped_dimer", protein_list="test_dimer_chopped.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="long_name", protein_list="test_long_name.txt", mode="custom", script="run_structure_prediction.py"),
    )
    def test_(self, protein_list, mode, script):
        multimer = "monomer" not in protein_list
        res = subprocess.run(self._args(plist=protein_list, mode=mode, script=script), capture_output=True, text=True)
        self._runCommonTests(res, multimer)


# --------------------------------------------------------------------------- #
#                    parameterised “resume” / relaxation tests                #
# --------------------------------------------------------------------------- #
class TestResume(_TestBase):
    def setUp(self):
        super().setUp()
        self.protein_lists = self.test_protein_lists_dir / "test_dimer.txt"

        if not self.test_modelling_dir.exists():
            raise FileNotFoundError(self.test_modelling_dir)
        shutil.copytree(self.test_modelling_dir, self.output_dir, dirs_exist_ok=True)

        self.base_args = [
            sys.executable,
            str(self.script_multimer),
            "--mode=custom",
            "--num_cycle=1",
            "--num_predictions_per_model=1",
            f"--data_dir={DATA_DIR}",
            f"--protein_lists={self.protein_lists}",
            f"--monomer_objects_dir={self.test_features_dir}",
            "--job_index=1",
            f"--output_path={self.output_dir}",
        ]

    # ------------ helper --------------------------------------------------- #
    def _runAfterRelaxTests(self, relax_mode="All"):
        expected = {"None": 0, "Best": 1, "All": 5}[relax_mode]
        d = self.output_dir / "TEST_and_TEST"
        got = len([f for f in d.iterdir() if f.name.startswith("relaxed") and f.suffix == ".pdb"])
        self.assertEqual(got, expected)

    # ------------ the parameterised resume table                            #
    @parameterized.named_parameters(
        dict(
            testcase_name="no_relax",
            relax_mode="None",
            remove=[
                "relaxed_model_1_multimer_v3_pred_0.pdb",
                "relaxed_model_2_multimer_v3_pred_0.pdb",
                "relaxed_model_3_multimer_v3_pred_0.pdb",
                "relaxed_model_4_multimer_v3_pred_0.pdb",
                "relaxed_model_5_multimer_v3_pred_0.pdb",
            ],
        ),
        dict(
            testcase_name="relax_all",
            relax_mode="All",
            remove=[
                "relaxed_model_1_multimer_v3_pred_0.pdb",
                "relaxed_model_2_multimer_v3_pred_0.pdb",
                "relaxed_model_3_multimer_v3_pred_0.pdb",
                "relaxed_model_4_multimer_v3_pred_0.pdb",
                "relaxed_model_5_multimer_v3_pred_0.pdb",
            ],
        ),
        dict(
            testcase_name="continue_relax",
            relax_mode="All",
            remove=["relaxed_model_5_multimer_v3_pred_0.pdb"],
        ),
        dict(
            testcase_name="continue_prediction",
            relax_mode="Best",
            remove=[
                "unrelaxed_model_5_multimer_v3_pred_0.pdb",
                "relaxed_model_1_multimer_v3_pred_0.pdb",
                "relaxed_model_2_multimer_v3_pred_0.pdb",
                "relaxed_model_3_multimer_v3_pred_0.pdb",
                "relaxed_model_4_multimer_v3_pred_0.pdb",
                "relaxed_model_5_multimer_v3_pred_0.pdb",
            ],
        ),
    )
    def test_(self, relax_mode, remove):
        args = self.base_args + [f"--models_to_relax={relax_mode}"]

        for f in remove:
            try:
                os.remove(self.output_dir / "TEST_and_TEST" / f)
            except FileNotFoundError:
                pass

        res = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(res, multimer=True, dirname="TEST_and_TEST")
        self._runAfterRelaxTests(relax_mode)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    absltest.main()
