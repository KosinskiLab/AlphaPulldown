import subprocess
import os
import pytest
from absl.testing import absltest, parameterized
import time
"""
Wrapper for test_predict_structure.sh and check_predict_structure.py
"""
#TODO: Test is passed if logs do not contain any errors.
class TestPredictStructure(parameterized.TestCase):
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

    def _sbatch_command(self, i: int, name: str):
        if not self._is_slurm_available():
            pytest.skip("Slurm not available")

        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env is None:
            pytest.fail("CONDA_DEFAULT_ENV not set")
        command = [
            "sbatch",
            f"--array={i}",
            f"--output={self.path}/%j.test_{i}_{name}.log",
            "test/test_predict_structure.sh",
            conda_env,
            name
        ]
        subprocess.run(command, check=True)


    @parameterized.named_parameters(
        {"testcase_name": "predict_monomer", "i": 1, "name": "monomer"},
        {"testcase_name": "dimer", "i": 2, "name": "dimer"},
        {"testcase_name": "chopped_dimer", "i": 3, "name": "chopped_dimer"},
        {"testcase_name": "homo_oligomer", "i": 4, "name": "homo_oligomer"},
    )
    def test_predict_structure(self, i: int, name: str):
        self._sbatch_command(i, name)


if __name__ == '__main__':
    absltest.main()