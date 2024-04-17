import subprocess
import os
import pytest

"""
Just to be able to run all tests with a single command
"""

def is_slurm_available():
    try:
        subprocess.run(["sinfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False

def test_sbatch_command():
    if not is_slurm_available():
        pytest.skip("Slurm not available")

    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env is None:
        pytest.fail("CONDA_DEFAULT_ENV not set")
    number_of_tests = 6 #adust this number to the number of tests
    for i in range(1, number_of_tests + 1):
        command = [
            "sbatch",
            f"--array={i}",
            f"--output=%j.testRun_{i}.log",
            "test/test_predict_structure.sh",
            conda_env]
        subprocess.run(command, check=True)