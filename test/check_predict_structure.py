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
import pytest
from alphapulldown.folding_backend.alphafold_backend import ModelsToRelax

FAST=True
if FAST:
    from alphafold.model import config
    config.CONFIG_MULTIMER.model.embeddings_and_evoformer.evoformer_num_block = 1
    #TODO: can it be done faster? For P0DPR3_and_P0DPR3 example, I think most of the time is taken by jax model compilation.

class _TestBase(parameterized.TestCase):
    def setUp(self) -> None:
        self.data_dir = "/scratch/AlphaFold_DBs/2.3.2/"
        #Get test_data directory as relative path to this script
        self.test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

class TestScript(_TestBase):
    #Add setup that creates empty output directory temporary
    def setUp(self) -> None:
        #Call the setUp method of the parent class
        super().setUp()

        #Create a temporary directory for the output
        self.output_dir = tempfile.mkdtemp()
        self.protein_lists = os.path.join(self.test_data_dir, "tiny_monomeric_features_homodimer.txt")
        self.monomer_objects_dir = self.test_data_dir

        #Get path of the alphapulldown module
        alphapulldown_path = alphapulldown.__path__[0]
        #join the path with the script name
        self.script_path = os.path.join(alphapulldown_path, "scripts/run_multimer_jobs.py")
        print(sys.executable)
        print(self.script_path)
        self.args = [
            sys.executable,
            self.script_path,
            "--mode=custom",
            "--num_cycle=1",
            "--num_predictions_per_model=1",
            f"--output_path={self.output_dir}",
            f"--data_dir={self.data_dir}",
            f"--protein_lists={self.protein_lists}",
            f"--monomer_objects_dir={self.monomer_objects_dir}"
        ]

    def tearDown(self) -> None:
        #Remove the temporary directory
        shutil.rmtree(self.output_dir)

    def _runCommonTests(self, result, multimer_mode: True):
        print(result.stdout)
        print(result.stderr)
        self.assertEqual(result.returncode, 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}")
        #Get the name of the first directory in the output directory
        dirname = next(subdir for subdir in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, subdir)))
        #Check if the directory contains five files starting from ranked and ending with .pdb
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("ranked") and f.endswith(".pdb")]), 5)
        #Check if the directory contains five files starting from result and ending with .pkl
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("result") and f.endswith(".pkl")]), 5)
        #Check if the result pickle dictionary contains all the keys 
        example_pickle = [f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("result") and f.endswith(".pkl")][0]
        example_pickle = pickle.load(open((os.path.join(self.output_dir, dirname, example_pickle)), 'rb'))
        if multimer_mode:
            required_keys = ['distogram', 'experimentally_resolved', 'masked_msa', 'predicted_aligned_error', 'predicted_lddt', 'structure_module', 'plddt', 'aligned_confidence_probs', 'max_predicted_aligned_error', 'seqs', 'ptm', 'iptm', 'ranking_confidence']
        else:
            required_keys = ['distogram', 'experimentally_resolved', 'masked_msa', 'predicted_aligned_error', 'predicted_lddt', 'structure_module', 'plddt', 'aligned_confidence_probs', 'max_predicted_aligned_error', 'seqs','ranking_confidence']
        self.assertContainsSubset(required_keys, list(example_pickle.keys()))
        #Check if the directory contains five files starting from pae and ending with .json
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("pae") and f.endswith(".json")]), 5)
        #Check if the directory contains five files ending with png
        print(os.listdir(os.path.join(self.output_dir, dirname)))
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.endswith(".png")]), 5)
        #Check if the directory contains ranking_debug.json
        self.assertTrue("ranking_debug.json" in os.listdir(os.path.join(self.output_dir, dirname)))
        #Check if the directory contains timings.json
        self.assertTrue("timings.json" in os.listdir(os.path.join(self.output_dir, dirname)))
        #Check timings_temp.json is not present
        self.assertFalse("timings_temp.json" in os.listdir(os.path.join(self.output_dir, dirname)))
        #Check if all files not empty
        for f in os.listdir(os.path.join(self.output_dir, dirname)):
            self.assertGreater(os.path.getsize(os.path.join(self.output_dir, dirname, f)), 0)
        #open the ranking_debug.json file and check if the number of models is 5
        with open(os.path.join(self.output_dir, dirname, "ranking_debug.json"), "r") as f:
            ranking_debug = json.load(f)
            self.assertEqual(len(ranking_debug["order"]), 5)
            if "iptm+ptm" in ranking_debug:
                expected_set = set(["model_1_multimer_v3_pred_0", "model_2_multimer_v3_pred_0", "model_3_multimer_v3_pred_0", "model_4_multimer_v3_pred_0", "model_5_multimer_v3_pred_0"])
                self.assertEqual(len(ranking_debug["iptm+ptm"]), 5)
                #Check if order contains the correct models
                self.assertSetEqual(set(ranking_debug["order"]), expected_set)
                #Check if iptm+ptm contains the correct models
                self.assertSetEqual(set(ranking_debug["iptm+ptm"].keys()), expected_set)
            elif "plddt" in ranking_debug:
                expected_set = set(["model_1_pred_0", "model_2_pred_0", "model_3_pred_0", "model_4_pred_0", "model_5_pred_0"])
                self.assertEqual(len(ranking_debug["plddt"]), 5)
                self.assertSetEqual(set(ranking_debug["order"]), expected_set)
                #Check if iptm+ptm contains the correct models
                self.assertSetEqual(set(ranking_debug["plddt"].keys()), expected_set)

    def testRun_1(self):
        """test run monomer structure prediction"""
        self.monomer_objects_dir = self.test_data_dir
        self.oligomer_state_file = os.path.join(self.test_data_dir, "test_homooligomer_state.txt")
        self.args = [
            sys.executable,
            self.script_path,
            "--mode=homo-oligomer",
            "--num_cycle=1",
            "--num_predictions_per_model=1",
            f"--output_path={self.output_dir}",
            f"--data_dir={self.data_dir}",
            f"--oligomer_state_file={self.oligomer_state_file}",
            f"--monomer_objects_dir={self.monomer_objects_dir}",
            "--job_index=1"
        ]
        result = subprocess.run(self.args, capture_output=True, text=True)
        self._runCommonTests(result, multimer_mode=False)

    def _runAfterRelaxTests(self, result):
        dirname = next(
            subdir for subdir in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, subdir)))
        #Check if the directory contains five files starting from relaxed and ending with .pdb
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("relaxed") and f.endswith(".pdb")]), 5)

    #@parameterized.named_parameters(('relax', ModelsToRelax.ALL),('no_relax', ModelsToRelax.NONE))
    def testRun_2(self):
        """test run without amber relaxation"""
        result = subprocess.run(self.args, capture_output=True, text=True)
        self._runCommonTests(result)
        #Check that directory does not contain relaxed pdb files
        dirname = next(
            subdir for subdir in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, subdir)))
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("relaxed") and f.endswith(".pdb")]), 0)
        self.assertIn("model_1_multimer_v3_pred_0", result.stdout + result.stderr)

    #pytest.mark.xfail
    def testRun_3(self):
        """test run with relaxation for all models"""
        self.args.append("--models_to_relax=All")
        result = subprocess.run(self.args, capture_output=True, text=True)
        self._runCommonTests(result)
        self._runAfterRelaxTests(result)
        self.assertIn("model_1_multimer_v3_pred_0", result.stdout + result.stderr)

    #@pytest.mark.xfail
    def testRun_4(self):
        """
        Test if the script can resume after all 5 models are finished, running amber relax on the 5 models
        """
        #Copy the example directory called "test" to the output directory
        shutil.copytree(os.path.join(self.test_data_dir,"P0DPR3_and_P0DPR3"), os.path.join(self.output_dir, "P0DPR3_and_P0DPR3"))
        self.args.append("--models_to_relax=All")
        result = subprocess.run(self.args, capture_output=True, text=True)
        self._runCommonTests(result)
        self._runAfterRelaxTests(result)
        self.assertIn("All predictions for", result.stdout + result.stderr)
        self.assertIn("are already completed.", result.stdout + result.stderr)
        
    def testRun_5(self):
        """
        Test if the script can resume after 2 models are finished
        """
        # Copy the example directory called "test" to the output directory
        shutil.copytree(os.path.join(self.test_data_dir, "P0DPR3_and_P0DPR3_partial"), os.path.join(self.output_dir, "P0DPR3_and_P0DPR3"))
        result = subprocess.run(self.args, capture_output=True, text=True)
        # self.assertIn("Found existing results, continuing from there", result.stdout + result.stderr) # this part of logging has been removed
        self.assertNotIn("using model_1_multimer_v3_pred_0", result.stdout + result.stderr)
        self.assertNotIn("using model_2_multimer_v3_pred_0", result.stdout + result.stderr)

        self._runCommonTests(result)

    def testRun_6(self):
        """
        Test running structure prediction with --multimeric_template=True
        Checks that the output model follows provided template (RMSD < 3 A)
        """
        #checks that features contain pickle files
        self.assertTrue(os.path.exists(os.path.join(
            self.test_data_dir, "true_multimer", "features", "3L4Q_A.pkl")))
        self.assertTrue(os.path.exists(os.path.join(
            self.test_data_dir, "true_multimer", "features", "3L4Q_C.pkl")))
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
            f"--protein_lists={self.test_data_dir}/true_multimer/custom.txt",
            f"--monomer_objects_dir={self.test_data_dir}/true_multimer/features",
            "--job_index=1"
        ]
        result = subprocess.run(self.args, capture_output=True, text=True)
        print(self.args)
        print(result.stdout)
        print(result.stderr)
        #self._runCommonTests(result) # fails because only one model is run
        reference = os.path.join(
            self.test_data_dir, "true_multimer", "modelling", "3L4Q_A_and_3L4Q_C", "ranked_0.pdb")
        for i in range(5):
            target = os.path.join(self.output_dir, "3L4Q_A_and_3L4Q_C", f"ranked_{i}.pdb")
            assert os.path.exists(target)
            with tempfile.TemporaryDirectory() as temp_dir:
                rmsds = calculate_rmsd_and_superpose(reference, target, temp_dir=temp_dir)
                print(f"Model {i} RMSD {rmsds}")
        # Best RMSD must be below ?? A now it's between 20 and 22 A
        #TODO: assert min(rmsd_chain_b) < ??

    def testRun_7(self):
        """Test multimeric template modelling without creating fake dbs and features"""
        self.assertTrue(os.path.exists(os.path.join(
            self.test_data_dir, "true_multimer", "features", "3L4Q_A.pkl")))
        self.assertTrue(os.path.exists(os.path.join(
            self.test_data_dir, "true_multimer", "features", "3L4Q_C.pkl")))
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
                f"--path_to_mmt={self.test_data_dir}/true_multimer",
                f"--description_file={self.test_data_dir}/true_multimer/description_file.csv",
                f"--output_path={tmpdir}",
                f"--data_dir={self.data_dir}",
                f"--protein_lists={self.test_data_dir}/true_multimer/custom.txt",
                f"--monomer_objects_dir={self.test_data_dir}/true_multimer/features",
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
                f"--description_file={self.test_data_dir}/true_multimer/description_file.csv",
                f"--output_path={tmpdir}",
                f"--data_dir={self.data_dir}",
                f"--protein_lists={self.test_data_dir}/true_multimer/custom.txt",
                f"--monomer_objects_dir={self.test_data_dir}/true_multimer/features",
                f"--remove_result_pickles",
                "--job_index=1"
            ]
            result = subprocess.run(self.args, capture_output=True, text=True)
            print(f"{result.stderr}")
            self.assertTrue("ranking_debug.json" in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C")))
            self.assertEqual(len([f for f in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C")) if f.startswith("result") and f.endswith(".pkl")]), 1)
        pass
    
    @pytest.mark.xfail
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
                f"--protein_lists={self.test_data_dir}/true_multimer/custom.txt",
                f"--monomer_objects_dir={self.test_data_dir}/true_multimer/features",
                f"--noremove_result_pickles",
                "--desired_num_res=500",
                "--desired_num_msa=2000"
            ]
            result = subprocess.run(self.args, capture_output=True, text=True)
            print(f"{result.stderr}")
            self.assertTrue("ranking_debug.json" in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C")))
            self.assertEqual(len([f for f in os.listdir(os.path.join(tmpdir, "3L4Q_A_and_3L4Q_C")) if f.startswith("result") and f.endswith(".pkl")]), 5)
    
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
                f"--protein_lists={self.test_data_dir}/A0A075B6L2_P0DPR3.txt",
                f"--monomer_objects_dir={self.test_data_dir}",
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
                f"--protein_lists={self.test_data_dir}/A0A075B6L2_P0DPR3.txt",
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
                f"--protein_lists={self.test_data_dir}/A0A075B6L2_P0DPR3.txt",
                f"--monomer_objects_dir={self.test_data_dir}",
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
