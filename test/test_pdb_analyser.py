import os
from os.path import exists
from absl.testing import absltest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from alphapulldown.analysis_pipeline.pdb_analyser import PDBAnalyser
import alphapulldown


class TestPDBAnalyser(absltest.TestCase):

    @patch('alphapulldown.analysis_pipeline.pdb_analyser.PDBParser.get_structure')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.PandasPdb.read_pdb')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.os.path.exists')
    def setUp(self, mock_exists, mock_read_pdb, mock_get_structure):
        # Use the real path to the ranked_0.pdb file
        real_pdb_path = os.path.join(os.path.dirname(alphapulldown.__file__),
                                     'test/test_data/predictions/TEST_and_TEST/ranked_0.pdb')
        mock_exists.side_effect = lambda path: path == real_pdb_path

        mock_read_pdb.return_value = MagicMock(df={'ATOM': pd.DataFrame({
            'chain_id': ['A', 'A', 'B', 'B'],
            'atom_name': ['CB', 'CA', 'CB', 'CA'],
            'residue_name': ['ALA', 'GLY', 'ALA', 'GLY'],
            'x_coord': [1.0, 2.0, 3.0, 4.0],
            'y_coord': [1.0, 2.0, 3.0, 4.0],
            'z_coord': [1.0, 2.0, 3.0, 4.0]
        })})
        mock_get_structure.return_value = MagicMock()
        self.analyser = PDBAnalyser(real_pdb_path)

    def test_retrieve_C_beta_coords(self):
        chain_df = self.analyser.pdb_df[self.analyser.pdb_df['chain_id'] == 'A']
        coords = self.analyser.retrieve_C_beta_coords(chain_df)
        expected_coords = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        np.testing.assert_array_equal(coords, expected_coords)

    def test_obtain_interface_residues(self):
        chain_df_1 = self.analyser.pdb_df[self.analyser.pdb_df['chain_id'] == 'A']
        chain_df_2 = self.analyser.pdb_df[self.analyser.pdb_df['chain_id'] == 'B']
        residues = self.analyser.obtain_interface_residues(chain_df_1, chain_df_2, cutoff=5)
        self.assertIsNotNone(residues)

    @patch('alphapulldown.analysis_pipeline.pdb_analyser.pose_from_pdb')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.PDBIO.save')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.get_score_function')
    def test_calculate_binding_energy(self, mock_get_score_function, mock_save, mock_pose_from_pdb):
        mock_get_score_function.return_value = MagicMock(return_value=10.0)
        mock_pose_from_pdb.return_value = MagicMock()
        energy = self.analyser.calculate_binding_energy('A', 'B')
        self.assertEqual(energy, -10.0) # 10(A_B)-10(A)-10(B) = -10

    def test_calculate_average_pae(self):
        pae_mtx = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        chain_1_residues = np.array([0, 1])
        chain_2_residues = np.array([2])
        average_pae = self.analyser.calculate_average_pae(pae_mtx, 'A', 'B', chain_1_residues, chain_2_residues)
        expected_pae = (2 + 3) / 2
        self.assertEqual(average_pae, expected_pae)

    def test_calculate_average_plddt(self):
        chain_1_plddt = [0.8, 0.9]
        chain_2_plddt = [0.85, 0.95]
        chain_1_residues = np.array([0, 1])
        chain_2_residues = np.array([0, 1])
        average_plddt = self.analyser.calculate_average_plddt(chain_1_plddt, chain_2_plddt, chain_1_residues, chain_2_residues)
        expected_plddt = (0.8 + 0.9 + 0.85 + 0.95) / 4
        self.assertEqual(average_plddt, expected_plddt)

    def test_update_df(self):
        input_df = pd.DataFrame({
            'interface': ['A_B'],
            'value': [1]
        })
        updated_df = self.analyser.update_df(input_df)
        expected_df = pd.DataFrame({
            'interface': ['A_B', 'B_A'],
            'value': [1, 1]
        })
        pd.testing.assert_frame_equal(updated_df, expected_df)

if __name__ == '__main__':
    absltest.main()