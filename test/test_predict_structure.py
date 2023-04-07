"""
Running the test script:
1. Batch job on gpu-el8
export PYTHONPATH=/g/kosinski/kosinski/devel/AlphaPulldown/:$PYTHONPATH
sbatch test_predict_structure.sh

2. Interactive session on gpu-el8
module load Anaconda3 
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
source activate AlphaPulldown

salloc -p gpu-el8 --ntasks 1 --cpus-per-task 8 --qos=highest --mem=16000 -C gaming -N 1 --gres=gpu:1 -t 05:00:00 

export PYTHONPATH=/g/kosinski/kosinski/devel/AlphaPulldown/:$PYTHONPATH
srun python test_predict_structure.py 

"""
import shutil
import tempfile
import unittest
import subprocess
import sys
import os

import alphapulldown

FAST=True
if FAST:
    from alphafold.model import config
    config.CONFIG_MULTIMER.model.embeddings_and_evoformer.evoformer_num_block = 1

class TestScript(unittest.TestCase):
    #Add setup that creates ampty output directory temprarily
    def setUp(self) -> None:
        #Create a temporary directory for the output
        self.output_dir = tempfile.mkdtemp()
        self.data_dir = "/scratch/AlphaFold_DBs/2.3.0/"
        #Get test_data directory as relative path to this script
        self.test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        self.protein_lists = os.path.join(self.test_data_dir, "tiny_monomeric_features_homodimer.txt")
        self.monomer_objects_dir = self.test_data_dir

        #Get path of the alphapulldown module
        alphapulldown_path = alphapulldown.__path__[0]
        #join the path with the script name
        self.script_path = os.path.join(alphapulldown_path, "run_multimer_jobs.py")

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

    def _runCommonTests(self, result):
        self.assertEqual(result.returncode, 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}")
        #Get the name of the first directory in the output directory
        dirname = os.listdir(self.output_dir)[0]
        #Check if the directory contains five files starting from ranked and ending with .pdb
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("ranked") and f.endswith(".pdb")]), 5)
        #Check if the directory contains five files starting from result and ending with .pkl
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("result") and f.endswith(".pkl")]), 5)
        #Check if the directory contains five files ending with png
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.endswith(".png")]), 5)
        #Check if the directory contains ranking_debug.json
        self.assertTrue("ranking_debug.json" in os.listdir(os.path.join(self.output_dir, dirname)))
        #Check if the directory contains timings.json
        self.assertTrue("timings.json" in os.listdir(os.path.join(self.output_dir, dirname)))

    def _runAfterRelaxTests(self, result):
        #Get the name of the first directory in the output directory
        print(os.listdir(os.path.join(self.output_dir, "P0DPR3_and_P0DPR3")))
        dirname = os.listdir(self.output_dir)[0]
        #Check if the directory contains five files starting from relaxed and ending with .pdb
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("relaxed") and f.endswith(".pdb")]), 5)

    def testRunWithoutAmberRelax(self):
        result = subprocess.run(self.args, capture_output=False, text=True)
        self._runCommonTests(result)

    def testRunWithAmberRelax(self):
        self.args.append("--amber_relax")
        result = subprocess.run(self.args, capture_output=False, text=True)
        self._runCommonTests(result)
        self._runAfterRelaxTests(result)

    def testRunWithAmberRelax_ResumeAfterAll5(self):
        """
        Test if the script can resume after all 5 models are finished, running amber relax on the 5 models
        """
        #Copy the example directory called "test" to the output directory
        shutil.copytree(os.path.join(self.test_data_dir,"P0DPR3_and_P0DPR3"), os.path.join(self.output_dir, "P0DPR3_and_P0DPR3"))
        #list content of the output directory
        self.args.append("--amber_relax")
        result = subprocess.run(self.args, capture_output=False, text=True)
        self._runCommonTests(result)
        self._runAfterRelaxTests(result)





if __name__ == '__main__':
    unittest.main()