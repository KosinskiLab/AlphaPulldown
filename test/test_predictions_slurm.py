import subprocess
import os
import pytest
from absl.testing import absltest
import time
"""
Wrapper for test_predict_structure.sh and check_predict_structure.py
"""
#TODO: Test is passed if logs do not contain any errors.
class TestPredictStructure(absltest.TestCase):
    def setUp(self) -> None:
        # Call the setUp method of the parent class
        super().setUp()
        # create slurm_logs directory if it does not exist
        # get the current time to create a unique directory
        t = time.localtime()
        self.path = f"slurm_logs/{t.tm_year}-{t.tm_mon}-{t.tm_mday}_{t.tm_hour}:{t.tm_min}:{t.tm_sec}"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def _is_slurm_available(self):
        try:
            subprocess.run(["sinfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except FileNotFoundError:
            return False
        except subprocess.CalledProcessError:
            return False

    def _sbatch_command(self, i: int):
        if not self._is_slurm_available():
            pytest.skip("Slurm not available")

        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env is None:
            pytest.fail("CONDA_DEFAULT_ENV not set")
        command = [
            "sbatch",
            f"--array={i}",
            f"--output={self.path}/%j.testRun_{i}.log",
            "test/test_predict_structure.sh",
            conda_env]
        subprocess.run(command, check=True)

    def test_predict_monomer(self):
        self._sbatch_command(1)

    def test_without_relaxation(self):
        self._sbatch_command(2)

    def test_with_relaxation_all_models(self):
        self._sbatch_command(3)

    def test_resume_relaxation(self):
        self._sbatch_command(4)

    def test_resume_prediction(self):
        self._sbatch_command(5)

    def test_true_multimer(self):
        self._sbatch_command(6)


if __name__ == '__main__':
    absltest.main()