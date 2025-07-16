import subprocess
from pathlib import Path
from absl.testing import absltest
import alphapulldown.scripts.create_individual_features as run_features_generation
import tempfile
import time
import os


class TestCreateIndividualFeaturesWithTemplates(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.TEST_DATA_DIR = Path(self.temp_dir.name)
        self.TEST_DATA_DIR = Path('debug')
        self.logs = self.TEST_DATA_DIR / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.fasta_file = Path(__file__).parent / "test_data" / "true_multimer" / "fastas" / "3L4Q.fa"


    def tearDown(self):
        print(self.TEST_DATA_DIR)
        print(os.listdir(self.TEST_DATA_DIR))
        self.temp_dir.cleanup()


    def generate_slurm_script(self):
        slurm_script_content = \
            f"""#!/bin/bash
#SBATCH --job-name=array
#SBATCH --time=10:00:00
#SBATCH -e {self.logs}/create_individual_features_%A_%a_err.txt
#SBATCH -o {self.logs}/create_individual_features_%A_%a_out.txt
#SBATCH --qos=low
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --mem=32000
#module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate AlphaPulldown

SEQ_INDEX=${{SLURM_ARRAY_TASK_ID:-1}}
USE_MMSEQS2=False
if [ "$SEQ_INDEX" -eq 1 ] || [ "$SEQ_INDEX" -eq 3 ]; then
    USE_MMSEQS2=True
fi

python {run_features_generation.__file__} \\
--use_precomputed_msas False \\
--save_msa_files True \\
--skip_existing False \\
--data_dir /scratch/AlphaFold_DBs/2.3.2 \\
--max_template_date 3021-01-01 \\
--fasta_paths {self.fasta_file} \\
--output_dir {self.TEST_DATA_DIR}/features_mmseqs_${{USE_MMSEQS2}} \\
--seq_index ${{SEQ_INDEX}} \\
--use_mmseqs2 ${{USE_MMSEQS2}}
    """
        script_file = Path(self.TEST_DATA_DIR) / "run_feature_generation.slurm"
        with open(script_file, 'w') as file:
            file.write(slurm_script_content)
        return script_file


    def submit_slurm_job(self, script_file):
        command = ['sbatch', '--array=1-4', str(script_file)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]  # Assumes sbatch output is "Submitted batch job <job_id>"
        return job_id


    def wait_for_slurm_jobs_to_finish(self, job_id, polling_interval=60):
        """Polls squeue and waits for the job to no longer be listed."""
        while True:
            result = subprocess.run(['squeue', '--job', job_id], capture_output=True, text=True)
            if job_id not in result.stdout:
                break  # Job has finished
            time.sleep(polling_interval)


    def compare_output_files(self):
        for prefix in ["True", "False"]:
            if prefix == 'True':
                for protein in ["3L4Q_A_mmseqs.pkl", "3L4Q_C_mmseqs.pkl"]:
                    assert (self.TEST_DATA_DIR / f"features_mmseqs_{prefix}" / protein).exists()
            if prefix == 'False':
                for protein in ["3L4Q_A.pkl", "3L4Q_C.pkl"]:
                    assert (self.TEST_DATA_DIR / f"features_mmseqs_{prefix}" / protein).exists()

    @absltest.skip("Skipping this test for now")
    def test_run_features_generation_all_parallel(self):
        slurm_script_file = self.generate_slurm_script()
        print(f"SLURM script generated at: {slurm_script_file}")
        job_id = self.submit_slurm_job(slurm_script_file)
        print(f"Submitted SLURM job {job_id}")
        self.wait_for_slurm_jobs_to_finish(job_id)
        print("All SLURM jobs have completed.")
        self.compare_output_files()


if __name__ == '__main__':
    absltest.main()
