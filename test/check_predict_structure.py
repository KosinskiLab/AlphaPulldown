"""
Running the test script:
1. Batch job on gpu-el8
sbatch --array={test_number} test_predict_structure.sh {conda_env}

2. Interactive session on gpu-el8
salloc -p gpu-el8 --ntasks 1 --cpus-per-task 8 --qos=highest --mem=16000 -C gaming -N 1 --gres=gpu:1 -t 05:00:00
srun python test/check_predict_structure.py # slower due to the slow compilation error
"""
import shutil
import tempfile
import sys
import pickle
import os
import subprocess
import json
#import alphapulldown.utils.calculate_rmsd as ...
import alphapulldown
from absl.testing import absltest
from absl.testing import parameterized

FAST = True
if FAST:
    from alphafold.model import config
    config.CONFIG_MULTIMER.model.embeddings_and_evoformer.evoformer_num_block = 1
    # TODO: can it be done faster? most of the time is taken by jax model compilation.


class _TestBase(parameterized.TestCase):
    def setUp(self) -> None:
        self.data_dir = "/scratch/AlphaFold_DBs/2.3.2/"
        # Paths to test directories
        this_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_data_dir = os.path.join(this_dir, "test_data")
        self.test_fastas_dir = os.path.join(self.test_data_dir, "fastas")
        self.test_features_dir = os.path.join(self.test_data_dir, "features")
        self.test_protein_lists_dir = os.path.join(self.test_data_dir, "protein_lists")
        self.test_templates_dir = os.path.join(self.test_data_dir, "templates")
        self.test_modelling_dir = os.path.join(self.test_data_dir, "predictions")

        self.tempdir = tempfile.TemporaryDirectory()
        self.output_dir = self.tempdir.name

        # Path to scripts
        alphapulldown_path = alphapulldown.__path__[0]
        self.script_path1 = os.path.join(alphapulldown_path, "scripts", "run_multimer_jobs.py")
        self.script_path2 = os.path.join(alphapulldown_path, "scripts", "run_structure_prediction.py")

    def _build_args(self, protein_list, mode, script, fold_backend="alphafold2"):
        """Helper to build command-line arguments for each test scenario."""
        if script == 'run_multimer_jobs.py':
            args = [
                sys.executable,
                self.script_path1,
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_dir={self.data_dir}",
                f"--monomer_objects_dir={self.test_features_dir}",
                "--job_index=1",
                f"--output_path={self.output_dir}",
                f"--fold_backend={fold_backend}",
            ]
            # If "homo-oligomer", we pass --oligomer_state_file, else --protein_lists
            flag = "--oligomer_state_file" if mode == "homo-oligomer" else "--protein_lists"
            pl_path = os.path.join(self.test_protein_lists_dir, protein_list)
            args.extend([f"--mode={mode}", f"{flag}={pl_path}"])

        elif script == 'run_structure_prediction.py':
            args = [
                sys.executable,
                self.script_path2,
                "--input=A0A075B6L2:10:1-3:4-5:6-7:7-8",
                f"--output_directory={self.output_dir}",
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_directory={self.data_dir}",
                f"--features_directory={self.test_features_dir}",
                f"--fold_backend={fold_backend}",
            ]

        return args

    def _runCommonTests(self, result, multimer_mode, fold_backend="alphafold2", dirname=None):
        """Checks that the output is correct given the fold_backend."""
        print(result.stdout)
        print(result.stderr)
        self.assertEqual(
            result.returncode, 0,
            f"Script failed with output:\n{result.stdout}\n{result.stderr}"
        )

        # If no specific directory name is given, check all subdirs
        if dirname is None:
            directories = [
                d for d in os.listdir(self.output_dir)
                if os.path.isdir(os.path.join(self.output_dir, d))
            ]
        else:
            directories = [dirname]

        for d in directories:
            dir_path = os.path.join(self.output_dir, d)
            files_in_dir = os.listdir(dir_path)
            print(f"Contents of {dir_path}:", files_in_dir)

            if fold_backend == "alphafold2":
                # Original checks for alphafold2 output
                self._checkAF2Outputs(dir_path, files_in_dir, multimer_mode)
            else:
                # New checks for alphafold3
                self._checkAF3Outputs(dir_path, files_in_dir, multimer_mode)

    def _checkAF2Outputs(self, dir_path, files_in_dir, multimer_mode):
        """Check the old AlphaFold2 style outputs: ranked_*.pdb, result_*.pkl, pae_*.json, etc."""
        # 5 ranked PDB
        self.assertEqual(
            len([f for f in files_in_dir if f.startswith("ranked") and f.endswith(".pdb")]),
            5
        )
        # 5 result pickles
        pickles = [f for f in files_in_dir if f.startswith("result") and f.endswith(".pkl")]
        self.assertEqual(len(pickles), 5)

        # 5 PAE files
        pae_files = [f for f in files_in_dir if f.startswith("pae") and f.endswith(".json")]
        self.assertEqual(len(pae_files), 5)

        # 5 PNG (plddt) images
        png_files = [f for f in files_in_dir if f.endswith(".png")]
        self.assertEqual(len(png_files), 5)

        # ranking_debug.json & timings.json present
        self.assertIn("ranking_debug.json", files_in_dir)
        self.assertIn("timings.json", files_in_dir)

        # All must be non-empty
        for f in files_in_dir:
            path = os.path.join(dir_path, f)
            self.assertGreater(os.path.getsize(path), 0)

        # Check ranking_debug
        with open(os.path.join(dir_path, "ranking_debug.json"), "r") as f:
            ranking_debug = json.load(f)
        self.assertEqual(len(ranking_debug["order"]), 5)

        expected_set_multimer = {
            "model_1_multimer_v3_pred_0",
            "model_2_multimer_v3_pred_0",
            "model_3_multimer_v3_pred_0",
            "model_4_multimer_v3_pred_0",
            "model_5_multimer_v3_pred_0"
        }
        expected_set_monomer = {
            "model_1_pred_0",
            "model_2_pred_0",
            "model_3_pred_0",
            "model_4_pred_0",
            "model_5_pred_0"
        }
        expected_set = expected_set_multimer if multimer_mode else expected_set_monomer
        if "iptm+ptm" in ranking_debug:
            self.assertEqual(len(ranking_debug["iptm+ptm"]), 5)
            self.assertSetEqual(set(ranking_debug["order"]), expected_set)
            self.assertSetEqual(set(ranking_debug["iptm+ptm"].keys()), expected_set)
        elif "plddt" in ranking_debug:
            self.assertEqual(len(ranking_debug["plddt"]), 5)
            self.assertSetEqual(set(ranking_debug["order"]), expected_set)
            self.assertSetEqual(set(ranking_debug["plddt"].keys()), expected_set)

    def _checkAF3Outputs(self, dir_path, files_in_dir, multimer_mode):
        """
        Checks the new AlphaFold3 style outputs, e.g.:
        - ranking_scores.csv
        - pXXXX_model.cif (or pXXXX_and_YYYY_model.cif for multi)
        - pXXXX_confidences.json, pXXXX_data.json, pXXXX_summary_confidences.json
        - seed-42_sample-# subdirectories each containing model.cif, confidences.json, summary_confidences.json
        - TERMS_OF_USE.md
        """
        # Check presence of "ranking_scores.csv"
        self.assertIn("ranking_scores.csv", files_in_dir)
        # Check presence of TERMS_OF_USE.md
        self.assertIn("TERMS_OF_USE.md", files_in_dir)

        # Check we have 5 seeds
        seed_dirs = [f for f in files_in_dir if f.startswith("seed-42_sample-")]
        self.assertEqual(len(seed_dirs), 5)

        # For the top-level model.cif, it might be "<monomer>_model.cif" or "<monomer>_and_<monomer>_model.cif"
        # We'll just search for something that ends in "_model.cif"
        top_level_cif = [f for f in files_in_dir if f.endswith("_model.cif")]
        self.assertEqual(len(top_level_cif), 1, "Should have exactly one top-level .cif file")

        # We also expect top-level confidences.json, data.json, summary_confidences.json
        # For a monomer: "p01308_confidences.json", "p01308_data.json", "p01308_summary_confidences.json"
        # For a dimer: "p01308_and_p61626_confidences.json", etc.
        # So let's just check that there is exactly 1 file that ends in "_confidences.json",
        # 1 that ends in "_data.json", 1 that ends in "_summary_confidences.json"
        conf_json = [f for f in files_in_dir if f.endswith("_confidences.json")]
        data_json = [f for f in files_in_dir if f.endswith("_data.json")]
        sumconf_json = [f for f in files_in_dir if f.endswith("_summary_confidences.json")]
        self.assertEqual(len(conf_json), 1)
        self.assertEqual(len(data_json), 1)
        self.assertEqual(len(sumconf_json), 1)

        # Check seed sub-directories have the correct files
        for sd in seed_dirs:
            sd_path = os.path.join(dir_path, sd)
            contents = os.listdir(sd_path)
            # We expect: model.cif, confidences.json, summary_confidences.json
            self.assertIn("model.cif", contents)
            self.assertIn("confidences.json", contents)
            self.assertIn("summary_confidences.json", contents)
            # each must be non-empty
            for f in contents:
                path = os.path.join(sd_path, f)
                self.assertGreater(os.path.getsize(path), 0)

        # Also check the top-level files aren't empty
        for f in [ "ranking_scores.csv", "TERMS_OF_USE.md", top_level_cif[0],
                   conf_json[0], data_json[0], sumconf_json[0] ]:
            path = os.path.join(dir_path, f)
            self.assertGreater(os.path.getsize(path), 0)


class TestRunModes(_TestBase):
    #
    # ============= AlphaFold2 tests =============
    #
    def test__monomer(self):
        """AF2 monomer test."""
        multimer_mode = False
        args = self._build_args("test_monomer.txt", "custom", "run_multimer_jobs.py", "alphafold2")
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode, fold_backend="alphafold2")

    def test__dimer(self):
        """AF2 dimer test."""
        multimer_mode = True
        args = self._build_args("test_dimer.txt", "custom", "run_multimer_jobs.py", "alphafold2")
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode, fold_backend="alphafold2")

    def test__trimer(self):
        """AF2 trimer test."""
        multimer_mode = True
        args = self._build_args("test_trimer.txt", "custom", "run_multimer_jobs.py", "alphafold2")
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode, fold_backend="alphafold2")

    def test__homo_oligomer(self):
        """AF2 homo-oligomer test."""
        multimer_mode = True
        args = self._build_args("test_homooligomer.txt", "homo-oligomer", "run_multimer_jobs.py", "alphafold2")
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode, fold_backend="alphafold2")

    def test__chopped_dimer(self):
        """AF2 chopped dimer test."""
        multimer_mode = True
        args = self._build_args("test_dimer_chopped.txt", "custom", "run_multimer_jobs.py", "alphafold2")
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode, fold_backend="alphafold2")

    def test__long_name(self):
        """AF2 test with run_structure_prediction.py for a 'long name' example."""
        multimer_mode = True  # or false depending on the input
        args = self._build_args("test_long_name.txt", "custom", "run_structure_prediction.py", "alphafold2")
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode, fold_backend="alphafold2")

    #
    # ============= AlphaFold3 tests =============
    #
    def test__monomer_af3(self):
        """AF3 monomer test."""
        multimer_mode = False
        args = self._build_args("test_monomer.txt", "custom", "run_multimer_jobs.py", "alphafold3")
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode, fold_backend="alphafold3")

    def test__dimer_af3(self):
        """AF3 dimer test."""
        multimer_mode = True
        args = self._build_args("test_dimer.txt", "custom", "run_multimer_jobs.py", "alphafold3")
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode, fold_backend="alphafold3")

    def test__trimer_af3(self):
        """AF3 trimer test."""
        multimer_mode = True
        args = self._build_args("test_trimer.txt", "custom", "run_multimer_jobs.py", "alphafold3")
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode, fold_backend="alphafold3")

    def test__homo_oligomer_af3(self):
        """AF3 homo-oligomer test."""
        multimer_mode = True
        args = self._build_args("test_homooligomer.txt", "homo-oligomer", "run_multimer_jobs.py", "alphafold3")
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode, fold_backend="alphafold3")

    def test__chopped_dimer_af3(self):
        """AF3 chopped dimer test."""
        multimer_mode = True
        args = self._build_args("test_dimer_chopped.txt", "custom", "run_multimer_jobs.py", "alphafold3")
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode, fold_backend="alphafold3")

    def test__long_name_af3(self):
        """AF3 test with run_structure_prediction.py for a 'long name' example."""
        multimer_mode = True
        args = self._build_args("test_long_name.txt", "custom", "run_structure_prediction.py", "alphafold3")
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode, fold_backend="alphafold3")


class TestResume(_TestBase):
    """
    Tests re-running or resuming partial outputs. We do these with both
    alphafold2 and alphafold3, checking different 'models_to_relax' scenarios.
    """
    def setUp(self) -> None:
        super().setUp()
        self.protein_lists = os.path.join(self.test_protein_lists_dir, "test_dimer.txt")
        if os.path.exists(self.test_modelling_dir):
            shutil.copytree(self.test_modelling_dir, self.output_dir, dirs_exist_ok=True)
        else:
            raise FileNotFoundError(f"Directory not found: {self.test_modelling_dir}")

        # Base arguments for run_multimer_jobs
        self.args_base = [
            sys.executable,
            self.script_path1,
            "--mode=custom",
            "--num_cycle=1",
            "--num_predictions_per_model=1",
            f"--data_dir={self.data_dir}",
            f"--protein_lists={self.protein_lists}",
            f"--monomer_objects_dir={self.test_features_dir}",
            "--job_index=1",
            f"--output_path={self.output_dir}",
        ]

    def _runAfterRelaxTests(self, relax_mode='All', dir_suffix='TEST_and_TEST'):
        """Simple check for # of relaxed files in old AF2 style. For AF3 we skip."""
        # For the original pipeline with alphafold2, check # of 'relaxed_*.pdb'
        # For alphafold3, there's no "relaxed_*.pdb" so this check won't find anything.
        if relax_mode == 'None':
            expected_count = 0
        elif relax_mode == 'Best':
            expected_count = 1
        else:
            expected_count = 5

        dirname = os.path.join(self.output_dir, f"{dir_suffix}/")
        if not os.path.exists(dirname):
            # For AF3 style, we might not even have a subfolder named "TEST_and_TEST"
            return

        relaxed_files = [
            f for f in os.listdir(dirname)
            if f.startswith("relaxed") and f.endswith(".pdb")
        ]
        self.assertEqual(len(relaxed_files), expected_count,
            f"Expected {expected_count} relaxed files, found {len(relaxed_files)}")

    #
    # ----------- AF2 resume tests ----------
    #
    def test__no_relax(self):
        """Resume test with no_relax on alphafold2."""
        args = self.args_base + ["--models_to_relax=None"]
        # AF2 default
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode=True, fold_backend="alphafold2", dirname='TEST_and_TEST')
        self._runAfterRelaxTests(relax_mode='None', dir_suffix='TEST_and_TEST')

    def test__relax_all(self):
        """Resume test with relax_all on alphafold2."""
        args = self.args_base + ["--models_to_relax=All"]
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode=True, fold_backend="alphafold2", dirname='TEST_and_TEST')
        self._runAfterRelaxTests(relax_mode='All', dir_suffix='TEST_and_TEST')

    def test__continue_relax(self):
        """
        Suppose we only have 4 relaxed files. Removing one, so the pipeline
        will only relax the missing one.
        """
        args = self.args_base + ["--models_to_relax=All"]
        remove_file = os.path.join(self.output_dir, 'TEST_and_TEST/', "relaxed_model_5_multimer_v3_pred_0.pdb")
        if os.path.exists(remove_file):
            os.remove(remove_file)

        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode=True, fold_backend="alphafold2", dirname='TEST_and_TEST')
        self._runAfterRelaxTests(relax_mode='All', dir_suffix='TEST_and_TEST')

    def test__continue_prediction(self):
        """
        Suppose we remove the unrelaxed model_5 and all relaxed. Then it has to re-run
        the last model inference + relax (mode=Best).
        """
        args = self.args_base + ["--models_to_relax=Best"]
        remove_list = [
            "unrelaxed_model_5_multimer_v3_pred_0.pdb",
            "relaxed_model_1_multimer_v3_pred_0.pdb",
            "relaxed_model_2_multimer_v3_pred_0.pdb",
            "relaxed_model_3_multimer_v3_pred_0.pdb",
            "relaxed_model_4_multimer_v3_pred_0.pdb",
            "relaxed_model_5_multimer_v3_pred_0.pdb"
        ]
        for f in remove_list:
            path = os.path.join(self.output_dir, 'TEST_and_TEST/', f)
            if os.path.exists(path):
                os.remove(path)

        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode=True, fold_backend="alphafold2", dirname='TEST_and_TEST')
        self._runAfterRelaxTests(relax_mode='Best', dir_suffix='TEST_and_TEST')

    #
    # ----------- AF3 resume tests ----------
    #
    def test__no_relax_af3(self):
        """Resume test with no_relax on alphafold3."""
        args = self.args_base + ["--models_to_relax=None", "--fold_backend=alphafold3"]
        result = subprocess.run(args, capture_output=True, text=True)
        # We'll still store output in "TEST_and_TEST" or similar
        self._runCommonTests(result, multimer_mode=True, fold_backend="alphafold3", dirname='TEST_and_TEST')
        # _runAfterRelaxTests won't see any relaxed_*.pdb in AF3

    def test__relax_all_af3(self):
        """Resume test with relax_all on alphafold3."""
        args = self.args_base + ["--models_to_relax=All", "--fold_backend=alphafold3"]
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode=True, fold_backend="alphafold3", dirname='TEST_and_TEST')

    def test__continue_relax_af3(self):
        """
        For AF3, there's no 'relaxed_*.pdb', but this test ensures we can safely skip that step
        without error. We just remove some seed-42 dirs if needed, but not strictly required.
        """
        args = self.args_base + ["--models_to_relax=All", "--fold_backend=alphafold3"]
        # If we wanted to simulate partial results, we might remove one seed subdir, e.g.:
        # remove_dir = os.path.join(self.output_dir, 'TEST_and_TEST/', "seed-42_sample-4")
        # if os.path.isdir(remove_dir):
        #     shutil.rmtree(remove_dir)
        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode=True, fold_backend="alphafold3", dirname='TEST_and_TEST')

    def test__continue_prediction_af3(self):
        """Simulate partial inference with AF3, e.g., remove seed-42_sample-4 folder."""
        args = self.args_base + ["--models_to_relax=Best", "--fold_backend=alphafold3"]

        # For demonstration, remove the top-level model.cif or the sample-4 directory
        path_to_remove = os.path.join(self.output_dir, 'TEST_and_TEST/', "seed-42_sample-4")
        if os.path.isdir(path_to_remove):
            shutil.rmtree(path_to_remove)

        result = subprocess.run(args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode=True, fold_backend="alphafold3", dirname='TEST_and_TEST')
        # No relaxed_*.pdb for AF3


if __name__ == '__main__':
    absltest.main()
