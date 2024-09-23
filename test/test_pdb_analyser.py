import os
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
        self.assertEqual(energy, 0.0)

if __name__ == '__main__':
    absltest.main()