import subprocess
from pathlib import Path
from absl.testing import absltest
import alphapulldown.create_individual_features_with_templates as run_features_generation
import pickle
import  numpy as np
from alphapulldown.create_custom_template_db import extract_seqs


class TestCreateIndividualFeaturesWithTemplates(absltest.TestCase):

    def setUp(self):
        super().setUp()
        self.TEST_DATA_DIR = Path(__file__).parent / "test_data" / "true_multimer"
        # Create necessary directories if they don't exist
        (self.TEST_DATA_DIR / 'features').mkdir(parents=True, exist_ok=True)
        (self.TEST_DATA_DIR / 'templates').mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Clean up any files or directories created during testing
        sto_files = list((self.TEST_DATA_DIR / 'features').glob('*/pdb_hits.sto'))
        for sto_file in sto_files:
            sto_file.unlink()
        desc_file = self.TEST_DATA_DIR / 'description.csv'
        if desc_file.exists():
            desc_file.unlink()

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
            desc_file.write(f"{file_name}_{chain_id}.fasta, {file_name}.{file_extension}, {chain_id}\n")

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
            '--output_dir', f"{self.TEST_DATA_DIR}/features",
        ]

        # Check the output
        subprocess.run(cmd, check=True)
        assert pkl_path.exists()
        assert sto_path.exists()

        with open(pkl_path, 'rb') as f:
            feats = pickle.load(f).feature_dict
        temp_sequence = feats['template_sequence'][0].decode('utf-8')
        target_sequence = feats['sequence'][0].decode('utf-8')
        atom_coords = feats['template_all_atom_positions'][0]
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
        print(f"seq-atom: {atom_seq}")
        print(len(atom_seq))
        # Check that atoms with non-zero coordinates are identical in seq-seqres and seq-atom
        residue_has_nonzero_coords = []
        atom_id = -1
        for number, (s, a) in enumerate(zip(temp_sequence, atom_coords)):
            # if mismatch between target and seqres
            if s == '-':
                assert np.all(a == 0)
                residue_has_nonzero_coords.append(False)
            else:
                non_zero = np.any(a != 0)
                residue_has_nonzero_coords.append(non_zero)
                if non_zero:
                    atom_id += 1
                    if seqres_seq:
                        seqres = seqres_seq[number]
                    else:
                        seqres = None
                    print(f"template-seq: {s} atom-seq: {atom_seq[atom_id]} seqres-seq: {seqres} id: {atom_id}")
                    if seqres:
                        assert (s == seqres_seq[number] or s == atom_seq[atom_id]) #seqres can be different from atomseq
                    else:
                        assert (s == atom_seq[atom_id])
                    assert np.any(a[:4] != 0)
        print(residue_has_nonzero_coords)
        print(len(residue_has_nonzero_coords))

    def test_1a_run_features_generation(self):
        self.run_features_generation('3L4Q', 'A', 'cif')

    def test_2c_run_features_generation(self):
        self.run_features_generation('3L4Q', 'C', 'pdb')

    def test_3b_bizarre_filename(self):
        self.run_features_generation('RANdom_name1_.7-1_0', 'B', 'pdb')

    def test_4c_bizarre_filename(self):
        self.run_features_generation('RANdom_name1_.7-1_0', 'C', 'pdb')


if __name__ == '__main__':
    absltest.main()
