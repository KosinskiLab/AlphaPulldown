import subprocess
from pathlib import Path
from absl.testing import absltest
import alphapulldown.create_individual_features_new as run_features_generation
import tempfile
import shutil


class TestCreateIndividualFeaturesWithTemplates(absltest.TestCase):

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.TEST_DATA_DIR = Path(self.temp_dir.name)
        original_test_data_dir = Path(__file__).parent / "test_data" / "true_multimer"
        shutil.copytree(original_test_data_dir, self.TEST_DATA_DIR, dirs_exist_ok=True)
        (self.TEST_DATA_DIR / 'features').mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def generate_slurm_script(self):
        slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name=array
#SBATCH --time=10:00:00
#SBATCH -e create_individual_features_%A_%a_err.txt
#SBATCH -o create_individual_features_%A_%a_out.txt
#SBATCH --qos=low
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --mem=32000
source activate AlphaPulldown

python {run_features_generation.__file__} \\
    --use_precomputed_msas True \\
    --save_msa_files True \\
    --skip_existing True \\
    --data_dir {self.TEST_DATA_DIR} \\
    --max_template_date 3021-01-01 \\
    --fasta_paths {self.TEST_DATA_DIR}/fastas/3L4Q_A.fasta,{self.TEST_DATA_DIR}/fastas/3L4Q_C.fasta \\
    --output_dir {self.TEST_DATA_DIR}/features \\
    --seq_index $SLURM_ARRAY_TASK_ID \\
    --use_mmseqs2 $(if [ $SLURM_ARRAY_TASK_ID -eq 1 ] || [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then echo True; else echo False; fi)
"""
        script_file = Path(self.TEST_DATA_DIR) / "run_feature_generation.slurm"
        with open(script_file, 'w') as file:
            file.write(slurm_script_content)
        return script_file

    def test_run_features_generation_all_parallel(self):
        slurm_script_file = self.generate_slurm_script()
        print(f"SLURM script generated at: {slurm_script_file}")
        with open (slurm_script_file, "r") as file:
            print(file.read())
        # Uncomment the next line if you actually want to submit the job
        # subprocess.run(['sbatch', '--array=1-4', str(slurm_script_file)], check=True)

if __name__ == '__main__':
    absltest.main()
