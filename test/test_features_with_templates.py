import subprocess
from pathlib import Path
from absl.testing import absltest
import alphapulldown.create_individual_features_with_templates as run_features_generation

class TestCreateIndividualFeaturesWithTemplates(absltest.TestCase):

    def setUp(self):
        super().setUp()
        self.TEST_DATA_DIR = Path(__file__).parent / "test_data" / "true_multimer"

    def test_main(self):
        # Remove existing files
        pkl_path_a = self.TEST_DATA_DIR / 'features' / '3L4Q_A.pkl'
        pkl_path_c = self.TEST_DATA_DIR / 'features' / '3L4Q_C.pkl'
        sto_path_a = self.TEST_DATA_DIR / 'features' / '3L4Q_A' / 'pdb_hits.sto'
        sto_path_c = self.TEST_DATA_DIR / 'features' / '3L4Q_C' / 'pdb_hits.sto'

        if pkl_path_a.exists():
            pkl_path_a.unlink()
        if pkl_path_c.exists():
            pkl_path_c.unlink()
        if sto_path_a.exists():
            sto_path_a.unlink()
        if sto_path_c.exists():
            sto_path_c.unlink()

        # Prepare the command and arguments
        cmd = [
            'python',
            run_features_generation.__file__,
            '--use_precomputed_msas', 'True',
            '--save_msa_files', 'True',
            '--skip_existing', 'True',
            '--data_dir', '/scratch/AlphaFold_DBs/2.3.2',
            '--max_template_date', '3021-01-01',
            '--threshold_clashes', '1000',
            '--hb_allowance', '0.4',
            '--plddt_threshold', '0',
            '--path_to_fasta', f"{self.TEST_DATA_DIR}/fastas",
            '--path_to_mmt', f"{self.TEST_DATA_DIR}/templates",
            '--description_file', f"{self.TEST_DATA_DIR}/description.csv",
            '--output_dir', f"{self.TEST_DATA_DIR}/features"
        ]

        # Run the command
        subprocess.run(cmd, check=True)

        # Check the output
        assert pkl_path_a.exists()
        assert pkl_path_c.exists()
        assert sto_path_a.exists()
        assert sto_path_c.exists()

if __name__ == '__main__':
    absltest.main()
