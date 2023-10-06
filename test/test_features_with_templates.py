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

    def tearDown(self):
        # Any cleanup can go here
        pass

    def test_main(self):
        # Remove existing files
        pkl_path_a = self.TEST_DATA_DIR / 'features' / '3L4Q_A.pkl'
        pkl_path_c = self.TEST_DATA_DIR / 'features' / '3L4Q_C.pkl'
        sto_path_a = self.TEST_DATA_DIR / 'features' / '3L4Q_A' / 'pdb_hits.sto'
        sto_path_c = self.TEST_DATA_DIR / 'features' / '3L4Q_C' / 'pdb_hits.sto'
        cif_path = self.TEST_DATA_DIR / 'templates' / '3L4Q.pdb'

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
            '--output_dir', f"{self.TEST_DATA_DIR}/features",
        ]

        # Check the output
        subprocess.run(cmd, check=True)
        assert pkl_path_a.exists()
        assert pkl_path_c.exists()
        assert sto_path_a.exists()
        assert sto_path_c.exists()

        for chain in ['A', 'C']:
            if chain == 'A':
                pkl_path = pkl_path_a
            if chain == 'C':
                pkl_path = pkl_path_c
            with open(pkl_path, 'rb') as f:
                feats = pickle.load(f).feature_dict
            temp_sequence = feats['template_sequence'][0].decode('utf-8')
            target_sequence = feats['sequence'][0].decode('utf-8')
            atom_coords = feats['template_all_atom_positions'][0]
            assert len(temp_sequence) > 0
            # Check that the atom coordinates are not all 0
            assert (atom_coords.any()) > 0

            atom_seq, seqres_seq = extract_seqs(cif_path, chain)
            print(f"target sequence: {target_sequence}")
            print(len(target_sequence))
            print(f"template sequence: {temp_sequence}")
            print(len(temp_sequence))
            print(f"seq-seqres: {seqres_seq}")
            print(len(seqres_seq))
            print(f"seq-atom: {atom_seq}")
            print(len(atom_seq))
            # Check that atoms with non-zero coordinates are identical in seq-seqres and seq-atom
            residue_has_nonzero_coords = []
            atom_id = 0
            for s, a in zip(temp_sequence, atom_coords):
                    # if mismatch between target and seqres
                    if s == '-':
                        assert np.all(a == 0)
                        residue_has_nonzero_coords.append(False)
                    else:
                        non_zero = np.any(a != 0)
                        residue_has_nonzero_coords.append(non_zero)
                        if non_zero:
                            assert s == atom_seq[atom_id]
                            assert np.any(a[:4] != 0)
                            atom_id += 1
            print(residue_has_nonzero_coords)
            print(len(residue_has_nonzero_coords))


if __name__ == '__main__':
    absltest.main()
