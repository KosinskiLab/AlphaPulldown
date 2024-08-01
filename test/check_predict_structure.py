"""
Running the test script:
1. Batch job on gpu-el8
sbatch --array={test_number} test_predict_structure.sh {conda_env}

2. Interactive session on gpu-el8
salloc -p gpu-el8 --ntasks 1 --cpus-per-task 8 --qos=highest --mem=16000 -C gaming -N 1 --gres=gpu:1 -t 05:00:00
srun python test/check_predict_structure.py # this will be slower due to the slow compilation error
"""
import shutil
import tempfile
import sys
import pickle
import os
import subprocess
import json
from alphapulldown.utils.calculate_rmsd import calculate_rmsd_and_superpose
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
        # Get test_data directory as relative path to this script
        self.test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        self.test_fastas_dir = os.path.join(self.test_data_dir, "fastas")
        self.test_features_dir = os.path.join(self.test_data_dir, "features")
        self.test_protein_lists_dir = os.path.join(self.test_data_dir, "protein_lists")
        self.test_templates_dir = os.path.join(self.test_data_dir, "templates")
        self.test_modelling_dir = os.path.join(self.test_data_dir, "predictions")
        self.tempdir = tempfile.TemporaryDirectory()
        self.output_dir = self.tempdir.name
        # Get path of the alphapulldown module
        alphapulldown_path = alphapulldown.__path__[0]
        # Join the path with the script name
        self.script_path = os.path.join(alphapulldown_path, "scripts/run_multimer_jobs.py")

    def _runCommonTests(self, result, multimer_mode):
        print(result.stdout)
        print(result.stderr)
        self.assertEqual(result.returncode, 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}")
        dirname = next(
            subdir for subdir in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, subdir)))
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if
                              f.startswith("ranked") and f.endswith(".pdb")]), 5)
        pickles = [f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("result") and f.endswith(".pkl")]
        self.assertEqual(len(pickles), 5)
        example_pickle = pickles[0]
        example_pickle = pickle.load(open(os.path.join(self.output_dir, dirname, example_pickle), 'rb'))

        required_keys_multimer = ['distogram', 'experimentally_resolved', 'masked_msa', 'predicted_aligned_error',
                                  'predicted_lddt', 'structure_module', 'plddt', 'aligned_confidence_probs',
                                  'max_predicted_aligned_error', 'seqs', 'iptm', 'ptm', 'ranking_confidence']
        required_keys_monomer = ['distogram', 'experimentally_resolved', 'masked_msa', 'predicted_aligned_error',
                                 'predicted_lddt', 'structure_module', 'plddt', 'aligned_confidence_probs',
                                 'max_predicted_aligned_error', 'seqs', 'ptm', 'ranking_confidence']

        required_keys = required_keys_multimer if multimer_mode else required_keys_monomer
        self.assertContainsSubset(required_keys, list(example_pickle.keys()))

        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if
                              f.startswith("pae") and f.endswith(".json")]), 5)
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.endswith(".png")]), 5)
        self.assertTrue("ranking_debug.json" in os.listdir(os.path.join(self.output_dir, dirname)))
        self.assertTrue("timings.json" in os.listdir(os.path.join(self.output_dir, dirname)))

        for f in os.listdir(os.path.join(self.output_dir, dirname)):
            self.assertGreater(os.path.getsize(os.path.join(self.output_dir, dirname, f)), 0)

        with open(os.path.join(self.output_dir, dirname, "ranking_debug.json"), "r") as f:
            ranking_debug = json.load(f)
            self.assertEqual(len(ranking_debug["order"]), 5)
            expected_set_multimer = set(
                ["model_1_multimer_v3_pred_0", "model_2_multimer_v3_pred_0", "model_3_multimer_v3_pred_0",
                 "model_4_multimer_v3_pred_0", "model_5_multimer_v3_pred_0"])
            expected_set_monomer = set(
                ["model_1_pred_0", "model_2_pred_0", "model_3_pred_0", "model_4_pred_0", "model_5_pred_0"])
            expected_set = expected_set_multimer if multimer_mode else expected_set_monomer

            if "iptm+ptm" in ranking_debug:
                self.assertEqual(len(ranking_debug["iptm+ptm"]), 5)
                self.assertSetEqual(set(ranking_debug["order"]), expected_set)
                self.assertSetEqual(set(ranking_debug["iptm+ptm"].keys()), expected_set)
            elif "plddt" in ranking_debug:
                self.assertEqual(len(ranking_debug["plddt"]), 5)
                self.assertSetEqual(set(ranking_debug["order"]), expected_set)
                self.assertSetEqual(set(ranking_debug["plddt"].keys()), expected_set)


class TestRunModes(_TestBase):
    def setUp(self) -> None:
        super().setUp()

        self.args = [
            sys.executable,
            self.script_path,
            "--num_cycle=1",
            "--num_predictions_per_model=1",
            f"--data_dir={self.data_dir}",
            f"--monomer_objects_dir={self.test_features_dir}",
            "--job_index=1"
        ]

    def tearDown(self) -> None:
        # Remove the temporary directory
        # shutil.rmtree(self.output_dir)
        pass


    @parameterized.named_parameters(
        {'testcase_name': 'monomer', 'protein_list': 'test_monomer.txt', 'mode': 'custom'},
        {'testcase_name': 'dimer', 'protein_list': 'test_dimer.txt', 'mode': 'custom'},
        {'testcase_name': 'homo_oligomer', 'protein_list': "test_homooligomer.txt", 'mode': 'homo-oligomer'},
        {'testcase_name': 'chopped_dimer', 'protein_list': 'test_dimer_chopped.txt', 'mode': 'custom'}
    )
    def test_(self, protein_list, mode):
        """Test run monomer structure prediction"""
        #self.output_dir = f"{self.test_modelling_dir}/" #Debug
        self.args.append(f"--output_path={self.output_dir}")
        flag = "--protein_lists"
        if mode == "homo-oligomer":
            flag = "--oligomer_state_file"
        if "monomer" in protein_list:
            multimer_mode = False
        else:
            multimer_mode = True
        protein_list = os.path.join(self.test_protein_lists_dir, protein_list)
        self.args.extend([
            f"--mode={mode}",
            f"{flag}={protein_list}",
        ])
        result = subprocess.run(self.args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode)


class TestResume(_TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.protein_lists = os.path.join(self.test_protein_lists_dir, "test_dimer.txt")
        dimer_dir = os.path.join(self.test_modelling_dir, 'TEST_and_TEST/')
        if not os.path.exists(dimer_dir):
            shutil.copytree(dimer_dir, self.output_dir)
        self.args = [
            sys.executable,
            self.script_path,
            "--mode=custom",
            "--num_cycle=1",
            "--num_predictions_per_model=1",
            f"--data_dir={self.data_dir}",
            f"--protein_lists={self.protein_lists}",
            f"--monomer_objects_dir={self.test_features_dir}",
            "--job_index=1",
            f"--output_path={self.output_dir}"
        ]

    def _runAfterRelaxTests(self):
        dirname = next(
            subdir for subdir in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, subdir)))
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname))
                              if f.startswith("relaxed") and f.endswith(".pdb")]), 5)

    @parameterized.named_parameters(
        {'testcase_name': 'no_relax', 'relax_mode': 'None', 'expected_relaxed_files': 0,
         'should_run_after_relax': False},
        {'testcase_name': 'relax_all', 'relax_mode': 'All', 'expected_relaxed_files': 5,
         'should_run_after_relax': True},
        {'testcase_name': 'continue_relax', 'relax_mode': 'All', 'expected_relaxed_files': 5,
         'should_run_after_relax': True,
         'continue_mode': True, 'test_dir': "TEST_and_TEST", 'test_file': "relaxed_model_5_multimer_v3_pred_0.pdb"},
        {'testcase_name': 'continue_prediction', 'relax_mode': 'Best', 'expected_relaxed_files': 1,
         'should_run_after_relax': False,
         'continue_mode': True, 'test_dir': "TEST_and_TEST", 'test_file': "unrelaxed_model_5_multimer_v3_pred_0.pdb"}
    )
    def test_(self, relax_mode, expected_relaxed_files, should_run_after_relax, continue_mode=False,
                                test_dir=None, test_file=None):
        """Test run with various relaxation modes and continuation scenarios"""
        if relax_mode != 'None':
            self.args.append(f"--models_to_relax={relax_mode}")

        if continue_mode and test_dir and test_file:
            os.remove(os.path.join(self.output_dir, test_dir, test_file))

        result = subprocess.run(self.args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode=True)

        if should_run_after_relax:
            self._runAfterRelaxTests()

        dirname = next(
            subdir for subdir in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, subdir)))
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname))
                              if f.startswith("relaxed_") and f.endswith(".pdb")]), expected_relaxed_files)

        self.assertIn("model_1_multimer_v3_pred_0", result.stdout + result.stderr)

        if continue_mode:
            self.assertIn("All predictions for", result.stdout + result.stderr)
        else:
            self.assertIn("using model_1_multimer_v3_pred_0", result.stdout + result.stderr)
            self.assertIn("using model_2_multimer_v3_pred_0", result.stdout + result.stderr)

    @absltest.skip("Not implemented yet")
    def testRunWithTemplate(self):
        """
        Test running structure prediction with --multimeric_template=True
        Checks that the output model follows provided template (RMSD < 3 A)
        """
        config.CONFIG_MULTIMER.model.embeddings_and_evoformer.evoformer_num_block = 48
        self.assertTrue(os.path.exists(os.path.join(self.test_features_dir, "3L4Q_A.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.test_features_dir, "3L4Q_C.pkl")))
        self.args = [
            sys.executable,
            self.script_path,
            "--mode=custom",
            "--num_cycle=48",
            "--num_predictions_per_model=5",
            "--multimeric_template=True",
            "--model_names=model_2_multimer_v3",
            "--msa_depth=30",
            f"--output_path={self.output_dir}",
            f"--data_dir={self.data_dir}",
            f"--protein_lists={self.test_protein_lists_dir}/test_truemultimer.txt",
            f"--monomer_objects_dir={self.test_features_dir}",
            "--job_index=1"
        ]
        result = subprocess.run(self.args, capture_output=True, text=True)
        print(self.args)
        print(result.stdout)
        print(result.stderr)
        reference = os.path.join(self.test_modelling_dir, "3L4Q_A_and_3L4Q_C", "ranked_0.pdb")
        for i in range(5):
            target = os.path.join(self.output_dir, "3L4Q_A_and_3L4Q_C", f"ranked_{i}.pdb")
            assert os.path.exists(target)
            with tempfile.TemporaryDirectory() as temp_dir:
                rmsds = calculate_rmsd_and_superpose(reference, target, temp_dir=temp_dir)
                print(f"Model {i} RMSD {rmsds}")
        # Best RMSD is high because of FAST=True
        # TODO: assert min(rmsd_chain_b) < ??

    @absltest.skip("Not implemented yet")
    def testRun_7(self):
        """Test multimeric template modelling without creating fake dbs and features"""
        self.assertTrue(os.path.exists(os.path.join(
            self.test_features_dir, "3L4Q_A.pkl")))
        self.assertTrue(os.path.exists(os.path.join(
            self.test_features_dir, "3L4Q_C.pkl")))
        with tempfile.TemporaryDirectory() as tmpdir:
            self.args = [
                sys.executable,
                self.script_path,
                "--mode=custom",
                "--num_cycle=3",
                "--num_predictions_per_model=1",
                "--multimeric_template=True",
                "--model_names=model_2_multimer_v3",
                "--msa_depth=16",
                f"--path_to_mmt={self.test_templates_dir}",
                f"--description_file={self.test_protein_lists_dir}/description_file.csv",
                f"--output_path={tmpdir}",
                f"--data_dir={self.data_dir}",
                f"--protein_lists={self.test_protein_lists_dir}/test_truemultimer.txt",
                f"--monomer_objects_dir={self.test_features_dir}",
                "--job_index=1"
            ]
            result = subprocess.run(self.args, capture_output=True, text=True)
            print(f"{result.stderr}")
            self.assertTrue("ranking_debug.json" in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C")))
            self.assertEqual(len([f for f in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C")) if f.startswith("pae") and f.endswith(".json")]), 1)
            self.assertEqual(len([f for f in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C")) if f.startswith("result") and f.endswith(".pkl")]), 1)
            # then test resume 
            self.args = [
                sys.executable,
                self.script_path,
                "--mode=custom",
                "--num_cycle=3",
                "--num_predictions_per_model=1",
                "--multimeric_template=True",
                "--msa_depth=16",
                f"--path_to_mmt={self.test_data_dir}/true_multimer",
                f"--description_file={self.test_protein_lists_dir}/description_file.csv",
                f"--output_path={tmpdir}",
                f"--data_dir={self.data_dir}",
                f"--protein_lists={self.test_protein_lists_dir}/custom.txt",
                f"--monomer_objects_dir={self.test_features_dir}",
                f"--remove_result_pickles",
                "--job_index=1"
            ]
            result = subprocess.run(self.args, capture_output=True, text=True)
            print(f"{result.stderr}")
            self.assertTrue("ranking_debug.json" in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C")))
            self.assertEqual(len([f for f in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C"))
                                  if f.startswith("result") and f.endswith(".pkl")]), 1)

    @absltest.skip("Not implemented yet")
    def testRun_8(self):
        """Test modelling with padding"""
        self.assertTrue(os.path.exists(os.path.join(
            self.test_data_dir, "true_multimer", "features", "3L4Q_A.pkl")))
        self.assertTrue(os.path.exists(os.path.join(
            self.test_data_dir, "true_multimer", "features", "3L4Q_C.pkl")))
        with tempfile.TemporaryDirectory() as tmpdir:
            # Firstly test running with padding AND multimeric template modelling
            self.args = [
                sys.executable,
                self.script_path,
                "--mode=custom",
                "--num_cycle=3",
                "--num_predictions_per_model=1",
                "--multimeric_template=True",
                "--model_names=model_2_multimer_v3",
                f"--path_to_mmt={self.test_data_dir}/true_multimer",
                f"--description_file={self.test_data_dir}/true_multimer/description_file.csv",
                f"--output_path={tmpdir}",
                f"--data_dir={self.data_dir}",
                f"--protein_lists={self.test_data_dir}/true_multimer/custom.txt",
                f"--monomer_objects_dir={self.test_data_dir}/true_multimer/features",
                "--desired_num_res=500",
                "--desired_num_msa=2000"
            ]
            result = subprocess.run(self.args, capture_output=True, text=True)
            print(f"{result.stderr}")
            self.assertTrue("ranking_debug.json" in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C")))
            self.assertEqual(len([f for f in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C")) if f.startswith("pae") and f.endswith(".json")]), 1)
            self.assertEqual(len([f for f in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C")) if f.startswith("result") and f.endswith(".pkl")]), 1)
        pass

    def testRun_9(self):
        """Test modelling with padding"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Then test running with padding WITHOUT multimeric template modelling
            self.args = [
                sys.executable,
                self.script_path,
                "--mode=custom",
                "--num_cycle=3",
                "--num_predictions_per_model=1",
                f"--output_path={tmpdir}",
                f"--data_dir={self.data_dir}",
                f"--protein_lists={self.test_protein_lists_dir}/custom.txt",
                f"--monomer_objects_dir={self.test_features_dir}/features",
                f"--noremove_result_pickles",
                "--desired_num_res=500",
                "--desired_num_msa=2000"
            ]
            result = subprocess.run(self.args, capture_output=True, text=True)
            print(f"{result.stderr}")
            self.assertTrue("ranking_debug.json" in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C")))
            self.assertEqual(len([f for f in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C"))
                                  if f.startswith("result") and f.endswith(".pkl")]), 5)
    
    def testRun_10(self):
        """
        Test no_pair_msa flag and the shape of the msa matrix
        msa shape = (2048, 121) when msa is paired
        msa shape = (2048, 121) when msa is not paired
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # First check the msa matrix shape with msa pairing
            
            self.args = [
                "run_multimer_jobs.py",
                "--mode=custom",
                "--num_cycle=3",
                "--num_predictions_per_model=1",
                f"--output_path={tmpdir}",
                f"--data_dir={self.data_dir}",
                f"--protein_lists={self.test_protein_lists_dir}/A0A075B6L2_P0DPR3.txt",
                f"--monomer_objects_dir={self.test_features_dir}",
                f"--noremove_result_pickles",
            ]
            result = subprocess.run(self.args, capture_output=True, text=True)
            print(f"{result.stderr}")
            self.assertIn("(2048, 121)", result.stdout + result.stderr) 
            self.assertNotIn("(2049, 121)", result.stdout + result.stderr)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # First check the msa matrix shape with msa pairing, msa should be paired even though pair_msa is not added to the command
            self.args = [
                "run_multimer_jobs.py",
                "--mode=custom",
                "--num_cycle=3",
                "--num_predictions_per_model=1",
                f"--output_path={tmpdir}",
                f"--data_dir={self.data_dir}",
                f"--protein_lists={self.test_protein_lists_dir}/A0A075B6L2_P0DPR3.txt",
                f"--monomer_objects_dir={self.test_data_dir}",
                f"--noremove_result_pickles",
                "--nopair_msa"
            ]
            result = subprocess.run(self.args, capture_output=True, text=True)
            print(f"{result.stderr}")
            self.assertNotIn("(2048, 121)", result.stdout + result.stderr)
            self.assertIn("(2049, 121)", result.stdout + result.stderr)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # First check the msa matrix shape with msa pairing, msa should be paired even though pair_msa is not added to the command
            self.args = [
                "run_multimer_jobs.py",
                "--mode=custom",
                "--num_cycle=3",
                "--num_predictions_per_model=1",
                f"--output_path={tmpdir}",
                f"--data_dir={self.data_dir}",
                f"--protein_lists={self.test_protein_lists_dir}/A0A075B6L2_P0DPR3.txt",
                f"--monomer_objects_dir={self.test_features_dir}",
                f"--noremove_result_pickles",
                f"--pair_msa"
            ]
            result = subprocess.run(self.args, capture_output=True, text=True)
            print(f"{result.stderr}")
            self.assertIn("(2048, 121)", result.stdout + result.stderr) 
            self.assertNotIn("(2049, 121)", result.stdout + result.stderr)

#TODO: Add tests for other modeling examples subclassing the class above
#TODO: Add tests that assess that the modeling results are as expected from native AlphaFold2
#TODO: Add tests that assess that the ranking is correct
#TODO: Add tests for features with and without templates
#TODO: Add tests for the different modeling modes (pulldown, homo-oligomeric, all-against-all, custom)
#TODO: Add tests for monomeric modeling done


if __name__ == '__main__':
    absltest.main()
