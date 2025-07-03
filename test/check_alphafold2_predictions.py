#!/usr/bin/env python
"""
Functional Alphapulldown alphafold2 backend tests.
Needs GPU(s) to run.
"""
from __future__ import annotations

import io
import os
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import logging
from pathlib import Path

from absl.testing import absltest, parameterized

import alphapulldown
from alphapulldown.utils.create_combinations import process_files

# --------------------------------------------------------------------------- #
#                         configuration / logging                             #
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("ALPHAFOLD_DATA_DIR", "/scratch/AlphaFold_DBs/2.3.0"))

FAST = __import__("os").getenv("ALPHAFOLD_FAST", "1") != "0"
if FAST:
    from alphafold.model import config
    config.CONFIG_MULTIMER.model.embeddings_and_evoformer.evoformer_num_block = 1

# --------------------------------------------------------------------------- #
#                       common helper mix-in / assertions                     #
# --------------------------------------------------------------------------- #
class _TestBase(parameterized.TestCase):
    use_temp_dir = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # do the skip here so import-time doesn't abort discovery
        if not DATA_DIR.is_dir():
            cls.skipTest(f"set $ALPHAFOLD_DATA_DIR to run Alphafold functional tests (tried {DATA_DIR!r})")

        # Create base output dir
        if cls.use_temp_dir:
            cls.base_output_dir = Path(tempfile.mkdtemp(prefix="af2_test_"))
        else:
            cls.base_output_dir = Path("test/test_data/predictions/af2_backend")
            if cls.base_output_dir.exists():
                try:
                    shutil.rmtree(cls.base_output_dir)
                except (PermissionError, OSError) as e:
                    logger.warning("Could not remove %s: %s", cls.base_output_dir, e)
            cls.base_output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if cls.use_temp_dir and cls.base_output_dir.exists():
            try:
                shutil.rmtree(cls.base_output_dir)
            except (PermissionError, OSError) as e:
                logger.warning("Could not remove temporary directory %s: %s", cls.base_output_dir, e)

    def setUp(self):
        super().setUp()

        this_dir = Path(__file__).resolve().parent
        self.test_data_dir = this_dir / "test_data"
        self.test_features_dir = self.test_data_dir / "features"
        self.test_protein_lists_dir = self.test_data_dir / "protein_lists"
        self.test_modelling_dir = self.test_data_dir / "predictions"
        self.af2_backend_dir = self.test_modelling_dir / "af2_backend"

        test_name = self._testMethodName
        self.output_dir = self.af2_backend_dir / test_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        apd_path = Path(alphapulldown.__path__[0])
        self.script_multimer = apd_path / "scripts" / "run_multimer_jobs.py"
        self.script_single = apd_path / "scripts" / "run_structure_prediction.py"

    # ---------------- assertions reused by all subclasses ----------------- #
    def _runCommonTests(self, res: subprocess.CompletedProcess, multimer: bool, dirname: str | None = None):
        if res.returncode != 0:
            self.fail(
                f"Subprocess failed (code {res.returncode})\n"
                f"STDOUT:\n{res.stdout}\n"
                f"STDERR:\n{res.stderr}"
            )

        dirs = [dirname] if dirname else [
            d for d in self.output_dir.iterdir() if d.is_dir()
        ]

        for d in dirs:
            folder = self.output_dir / d
            files = list(folder.iterdir())

            self.assertEqual(
                len([f for f in files if f.name.startswith("ranked") and f.suffix == ".pdb"]),
                5
            )

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

            self.assertEqual(len([f for f in files if f.name.startswith("pae") and f.suffix == ".json"]), 5)
            self.assertEqual(len([f for f in files if f.suffix == ".png"]), 5)
            names = {f.name for f in files}
            self.assertIn("ranking_debug.json", names)
            self.assertIn("timings.json", names)

            ranking = json.loads((folder / "ranking_debug.json").read_text())
            self.assertEqual(len(ranking["order"]), 5)

    def _args(self, *, plist, mode, script):
        if script.endswith("run_multimer_jobs.py"):
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
                ) + f"={self.test_protein_lists_dir / plist}",
            ]
        else:
            buffer = io.StringIO()
            _ = process_files(
                input_files=[str(self.test_protein_lists_dir / plist)],
                output_path=buffer,
                exclude_permutations=True
            )
            buffer.seek(0)
            lines = [
                x.strip().replace(",", ":").replace(";", "+")
                for x in buffer.readlines() if x.strip()
            ]
            formatted_input = lines[0] if lines else ""
            return [
                sys.executable,
                str(self.script_single),
                f"--input={formatted_input}",
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
        res = subprocess.run(
            self._args(plist=protein_list, mode=mode, script=script),
            capture_output=True, text=True
        )
        self._runCommonTests(res, multimer)


# --------------------------------------------------------------------------- #
#                    parameterised “resume” / relaxation tests                #
# --------------------------------------------------------------------------- #
class TestResume(_TestBase):
    def setUp(self):
        super().setUp()
        self.protein_lists = self.test_protein_lists_dir / "test_dimer.txt"
        self.af2_backend_dir.mkdir(parents=True, exist_ok=True)

        source = self.test_modelling_dir / "TEST_and_TEST"
        target = self.af2_backend_dir / "TEST_and_TEST"
        shutil.copytree(source, target, dirs_exist_ok=True)

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

    def _runAfterRelaxTests(self, relax_mode="All"):
        expected = {"None": 0, "Best": 1, "All": 5}[relax_mode]
        d = self.output_dir / "TEST_homo_2er"
        got = len([f for f in d.iterdir() if f.name.startswith("relaxed") and f.suffix == ".pdb"])
        self.assertEqual(got, expected)

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
        for fname in remove:
            try:
                (self.output_dir / "TEST_homo_2er" / fname).unlink()
            except FileNotFoundError:
                pass
        res = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(res, multimer=True, dirname="TEST_homo_2er")
        self._runAfterRelaxTests(relax_mode)


def _parse_test_args():
    use_temp = '--use-temp-dir' in sys.argv or __import__("os").getenv('USE_TEMP_DIR', '').lower() in ('1','true','yes')
    while '--use-temp-dir' in sys.argv:
        sys.argv.remove('--use-temp-dir')
    return use_temp

_TestBase.use_temp_dir = _parse_test_args()

if __name__ == "__main__":
    absltest.main()