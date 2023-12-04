import subprocess
from pathlib import Path
from absl.testing import absltest
import alphapulldown.create_individual_features_with_templates as run_features_generation
import pickle
import  numpy as np
from alphapulldown.remove_clashes_low_plddt import extract_seqs
import tempfile
import shutil
import glob


class TestCreateIndividualFeaturesWithTemplates(absltest.TestCase):

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()  # Create a temporary directory
        self.TEST_DATA_DIR = Path(self.temp_dir.name)  # Use the temporary directory as the test data directory
        # Copy test data files to the temporary directory
        original_test_data_dir = Path(__file__).parent / "test_data" / "true_multimer"
        shutil.copytree(original_test_data_dir, self.TEST_DATA_DIR, dirs_exist_ok=True)
        # Create necessary directories
        (self.TEST_DATA_DIR / 'features').mkdir(parents=True, exist_ok=True)
        (self.TEST_DATA_DIR / 'templates').mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()  # Clean up the temporary directory

    def run_features_generation(self, file_name, chain_id, file_extension):
        # Ensure directories exist
        (self.TEST_DATA_DIR / 'features').mkdir(parents=True, exist_ok=True)
        (self.TEST_DATA_DIR / 'templates').mkdir(parents=True, exist_ok=True)
        # Remove existing files (should be done by tearDown, but just in case)
        pkl_path = self.TEST_DATA_DIR / 'features' / f'{file_name}_{chain_id}.pkl'
        sto_path = self.TEST_DATA_DIR / 'features' / f'{file_name}_{chain_id}' / 'pdb_hits.sto'
        template_path = self.TEST_DATA_DIR / 'templates' / f'{file_name}.{file_extension}'
        if pkl_path.exists():
            pkl_path.unlink()
        if sto_path.exists():
            sto_path.unlink()

        # Generate description.csv
        with open(f"{self.TEST_DATA_DIR}/description.csv", 'w') as desc_file:
            desc_file.write(f">{file_name}_{chain_id}, {file_name}.{file_extension}, {chain_id}\n")

        assert Path(f"{self.TEST_DATA_DIR}/fastas/{file_name}_{chain_id}.fasta").exists()

        def create_mock_file(file_path):
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            Path(file_path).touch(exist_ok=True)

        # Common root directory
        root_dir = self.TEST_DATA_DIR

        # Mock databases
        create_mock_file(root_dir / 'uniref90/uniref90.fasta')
        create_mock_file(root_dir / 'mgnify/mgy_clusters_2022_05.fa')
        create_mock_file(root_dir / 'uniprot/uniprot.fasta')
        create_mock_file(root_dir / 'bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt_hhm.ffindex')
        create_mock_file(root_dir / 'uniref30/UniRef30_2021_03_hhm.ffindex')
        create_mock_file(root_dir / 'pdb70/pdb70_hhm.ffindex')

        # Mock hhblits files
        hhblits_root = root_dir / 'bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'
        hhblits_files = ['_a3m.ffdata', '_a3m.ffindex', '_cs219.ffdata', '_cs219.ffindex', '_hhmm.ffdata',
                         '_hhmm.ffindex']
        for file in hhblits_files:
            create_mock_file(hhblits_root / file)

        # Mock uniclust30 files
        uniclust_db_root = root_dir / 'uniclust30/uniclust30_2018_08/uniclust30_2018_08'
        uniclust_db_files = ['_a3m_db', '_a3m.ffdata', '_a3m.ffindex', '.cs219', '_cs219.ffdata', '_cs219.ffindex',
                             '_hhm_db', '_hhm.ffdata', '_hhm.ffindex']
        for suffix in uniclust_db_files:
            create_mock_file(f"{uniclust_db_root}{suffix}")

        # Mock uniref30 files - Adjusted for the correct naming convention
        uniref_db_root = root_dir / 'uniref30/UniRef30_2021_03'
        uniref_db_files = ['_a3m.ffdata', '_a3m.ffindex', '_hmm.ffdata', '_hmm.ffindex', '_cs.ffdata', '_cs.ffindex']
        for suffix in uniref_db_files:
            create_mock_file(f"{uniref_db_root}{suffix}")

        # Prepare the command and arguments
        cmd = [
            'python',
            run_features_generation.__file__,
            '--use_precomputed_msas', 'True',
            '--save_msa_files', 'True',
            '--skip_existing', 'True',
            '--data_dir', f"{self.TEST_DATA_DIR}",
            '--max_template_date', '3021-01-01',
            '--threshold_clashes', '1000',
            '--hb_allowance', '0.4',
            '--plddt_threshold', '0',
            '--fasta_paths', f"{self.TEST_DATA_DIR}/fastas/{file_name}_{chain_id}.fasta",
            '--path_to_mmt', f"{self.TEST_DATA_DIR}/templates",
            '--description_file', f"{self.TEST_DATA_DIR}/description.csv",
            '--output_dir', f"{self.TEST_DATA_DIR}/features",
        ]
        print(" ".join(cmd))
        # Check the output
        subprocess.run(cmd, check=True)
        features_dir = self.TEST_DATA_DIR / 'features'

        # List all files in the directory
        for file in features_dir.iterdir():
            if file.is_file():
                print(file)
        print("pkl path")
        print(pkl_path)
        assert pkl_path.exists()
        assert sto_path.exists()

        with open(pkl_path, 'rb') as f:
            feats = pickle.load(f).feature_dict
        temp_sequence = feats['template_sequence'][0].decode('utf-8')
        target_sequence = feats['sequence'][0].decode('utf-8')
        atom_coords = feats['template_all_atom_positions'][0]
        # Check that template sequence is not empty
        assert len(temp_sequence) > 0
        # Check that the atom coordinates are not all 0
        assert (atom_coords.any()) > 0

        atom_seq, seqres_seq = extract_seqs(template_path, chain_id)
        print(f"target sequence: {target_sequence}")
        print(len(target_sequence))
        print(f"template sequence: {temp_sequence}")
        print(len(temp_sequence))
        print(f"seq-seqres: {seqres_seq}")
        if seqres_seq:
            print(len(seqres_seq))
        # SeqIO adds X for missing residues for atom-seq
        print(f"seq-atom: {atom_seq}")
        print(len(atom_seq))
        # Check that atoms for not missing residues are not all 0
        residue_has_nonzero_coords = []
        for number, (s, a) in enumerate(zip(atom_seq, atom_coords)):
            # no coordinates for missing residues
            if s == 'X':
                assert np.all(a == 0)
                residue_has_nonzero_coords.append(False)
            else:
                non_zero = np.any(a != 0)
                residue_has_nonzero_coords.append(non_zero)
                if non_zero:
                    if seqres_seq:
                        seqres = seqres_seq[number]
                    else:
                        seqres = None
                    if seqres:
                        assert (s in seqres_seq)
                    # first 4 coordinates are non zero
                    assert np.any(a[:4] != 0)
        #print(residue_has_nonzero_coords)
        #print(len(residue_has_nonzero_coords))

    def test_1a_run_features_generation(self):
        self.run_features_generation('3L4Q', 'A', 'cif')

    def test_2c_run_features_generation(self):
        self.run_features_generation('3L4Q', 'C', 'pdb')

    def test_3b_bizarre_filename(self):
        self.run_features_generation('RANdom_name1_.7-1_0', 'B', 'pdb')

    def test_4c_bizarre_filename(self):
        self.run_features_generation('RANdom_name1_.7-1_0', 'C', 'pdb')

    def test_5b_gappy_pdb(self):
        self.run_features_generation('GAPPY_PDB', 'B', 'pdb')

    def test_6a_mmseqs2(self): #need to fix API call
        file_name = '3L4Q'
        chain_id = 'A'
        file_extension = 'cif'
        # Ensure directories exist
        (self.TEST_DATA_DIR / 'features').mkdir(parents=True, exist_ok=True)
        (self.TEST_DATA_DIR / 'templates').mkdir(parents=True, exist_ok=True)
        # Remove existing files (should be done by tearDown, but just in case)
        pkl_path = self.TEST_DATA_DIR / 'features' / f'{file_name}_{chain_id}.pkl'
        a3m_path = self.TEST_DATA_DIR / 'features' / f'{file_name}_{chain_id}.a3m'
        template_path = self.TEST_DATA_DIR / 'templates' / f'{file_name}.{file_extension}'
        if pkl_path.exists():
            pkl_path.unlink()
        if a3m_path.exists():
            a3m_path.unlink()

        # Generate description.csv
        with open(f"{self.TEST_DATA_DIR}/description.csv", 'w') as desc_file:
            desc_file.write(f">{file_name}_{chain_id}, {file_name}.{file_extension}, {chain_id}\n")

        # Prepare the command and arguments
        cmd = [
            'python',
            run_features_generation.__file__,
            '--skip_existing', 'False',
            '--data_dir', '/scratch/AlphaFold_DBs/2.3.2',
            '--max_template_date', '3021-01-01',
            '--threshold_clashes', '1000',
            '--hb_allowance', '0.4',
            '--plddt_threshold', '0',
            '--fasta_paths', f"{self.TEST_DATA_DIR}/fastas/{file_name}_{chain_id}.fasta",
            '--path_to_mmt', f"{self.TEST_DATA_DIR}/templates",
            '--description_file', f"{self.TEST_DATA_DIR}/description.csv",
            '--output_dir', f"{self.TEST_DATA_DIR}/features",
            '--use_mmseqs2', 'True',
        ]
        print(" ".join(cmd))
        # Check the output
        subprocess.run(cmd, check=True)
        assert pkl_path.exists()
        assert a3m_path.exists()


    def test_7a_hetatoms(self): # TODO: compute and commit features and msas
        self.run_features_generation('hetatoms', 'A', 'pdb')



if __name__ == '__main__':
    absltest.main()
