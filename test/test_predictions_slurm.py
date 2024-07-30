import logging
import subprocess
import os
import time
from absl.testing import absltest, parameterized
import shutil
"""
Wrapper for running structure prediction tests
"""

class TestPredictStructure(parameterized.TestCase):
    def setUp(self) -> None:
        # Call the setUp method of the parent class
        super().setUp()
        # Create slurm_logs directory if it does not exist
        # Get the current time to create a unique directory
        t = time.localtime()
        self.path = f"slurm_logs/{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}_{t.tm_hour:02d}"
        if os.path.exists(self.path):
            logging.warning("Warning: slurm_logs directory already exists. Deleting it...")
            shutil.rmtree(self.path)
        os.makedirs(self.path)

    def _is_slurm_available(self):
        try:
            subprocess.run(["sinfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _run_test_locally(self, class_name: str, test_name: str, conda_env: str):
        command = [
            "bash", "-c",
            f"""
            eval "$(conda shell.bash hook)"
            conda activate {conda_env}
            module load CUDA/11.8.0
            module load cuDNN/8.7.0.84-CUDA-11.8.0

            MAXRAM=$(echo `ulimit -m` '/ 1024.0'|bc)
            GPUMEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|tail -1`
            export XLA_PYTHON_CLIENT_MEM_FRACTION=`echo "scale=3;$MAXRAM / $GPUMEM"|bc`
            export TF_FORCE_UNIFIED_MEMORY='1'
            echo "Running {class_name}::{test_name}"
            pytest -s test/check_predict_structure.py::{class_name}::{test_name}
            """
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        result.check_returncode()

    def _sbatch_command(self, i: int, class_name: str, test_name: str):
        if not self._is_slurm_available():
            self.skipTest("Slurm not available")

        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env is None:
            self.fail("CONDA_DEFAULT_ENV not set")

        script_content = f"""#!/bin/bash
#SBATCH --job-name=test_predict_structure
#SBATCH --time=01:00:00
#SBATCH --qos=normal
#SBATCH -p gpu-el8
#SBATCH -C gaming
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000

eval "$(conda shell.bash hook)"
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

conda activate {conda_env}

MAXRAM=$(echo `ulimit -m` '/ 1024.0'|bc)
GPUMEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|tail -1`
export XLA_PYTHON_CLIENT_MEM_FRACTION=`echo "scale=3;$MAXRAM / $GPUMEM"|bc`
export TF_FORCE_UNIFIED_MEMORY='1'
echo "Running {class_name}::{test_name}"
pytest -s test/check_predict_structure.py::{class_name}::{test_name}
        """

        script_path = f"{self.path}/test_{i}_{class_name}_{test_name}.sh"
        with open(script_path, 'w') as script_file:
            script_file.write(script_content)

        command = [
            "sbatch",
            f"--output={self.path}/test_{i}_{class_name}_{test_name}.log",
            script_path
        ]
        subprocess.run(command, check=True)

    @parameterized.named_parameters(
        {"testcase_name": "monomer", "i": 1, "class_name": "TestRunModes", "test_name": "test__monomer"},
        {"testcase_name": "dimer", "i": 2, "class_name": "TestRunModes", "test_name": "test__dimer"},
        {"testcase_name": "chopped_dimer", "i": 3, "class_name": "TestRunModes", "test_name": "test__chopped_dimer"},
        {"testcase_name": "homo_oligomer", "i": 4, "class_name": "TestRunModes", "test_name": "test__homo_oligomer"},
        {"testcase_name": "no_relax", "i": 5, "class_name": "TestResume", "test_name": "test__no_relax"},
        {"testcase_name": "relax_all", "i": 6, "class_name": "TestResume", "test_name": "test__relax_all"},
        {"testcase_name": "continue_relax", "i": 7, "class_name": "TestResume", "test_name": "test__continue_relax"},
        {"testcase_name": "continue_prediction", "i": 8, "class_name": "TestResume", "test_name": "test__continue_prediction"},
    )
    def test_predict_structure(self, i: int, class_name: str, test_name: str):
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if not self._is_slurm_available():
            print("Slurm not available, running test locally")
            self._run_test_locally(class_name, test_name, conda_env)
        else:
            self._sbatch_command(i, class_name, test_name)

if __name__ == '__main__':
    absltest.main()
