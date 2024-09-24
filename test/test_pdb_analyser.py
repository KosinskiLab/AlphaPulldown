import subprocess
from absl.testing import absltest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from alphapulldown.analysis_pipeline.pdb_analyser import PDBAnalyser

class MockResidue:
    def __init__(self, id):
        self.id = id

class MockChain:
    def __init__(self, id, residues):
        self.id = id
        self._residues = residues

    def get_residues(self):
        return iter(self._residues)

class MockModel:
    def __init__(self, chains):
        # Store chains in a dictionary keyed by chain ID
        self._chains = {chain.id: chain for chain in chains}

    def __iter__(self):
        return iter(self._chains.values())

    def __getitem__(self, chain_id):
        return self._chains[chain_id]

class TestPDBAnalyser(absltest.TestCase):

    @patch('alphapulldown.analysis_pipeline.pdb_analyser.os.path.exists')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.PDBParser')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.PandasPdb')
    def setUp(self, mock_pandaspdb, mock_pdbparser, mock_exists):
        # Set the path to a dummy value
        pdb_file_path = 'dummy_path.pdb'
        mock_exists.return_value = True  # Simulate that the file exists

        # Mock the PandasPdb instance
        mock_pandaspdb_instance = mock_pandaspdb.return_value
        # Ensure that read_pdb() returns the PandasPdb instance itself
        mock_pandaspdb_instance.read_pdb.return_value = mock_pandaspdb_instance
        # Set the df attribute on the PandasPdb instance
        mock_pandaspdb_instance.df = {'ATOM': pd.DataFrame({
            'chain_id': ['A', 'A', 'B', 'B'],
            'atom_number': [1, 2, 3, 4],
            'residue_number': [1, 2, 1, 2],
            'atom_name': ['CB', 'CA', 'CB', 'CA'],
            'residue_name': ['ALA', 'GLY', 'ALA', 'GLY'],
            'x_coord': [1.0, 2.0, 3.0, 4.0],
            'y_coord': [1.0, 2.0, 3.0, 4.0],
            'z_coord': [1.0, 2.0, 3.0, 4.0],
            'element_symbol': ['C', 'C', 'C', 'C']
        })}

        # Create residues
        residue_A1 = MockResidue((' ', 1, ' '))
        residue_A2 = MockResidue((' ', 2, ' '))
        residue_B1 = MockResidue((' ', 1, ' '))
        residue_B2 = MockResidue((' ', 2, ' '))

        # Create chains
        chain_A = MockChain('A', [residue_A1, residue_A2])
        chain_B = MockChain('B', [residue_B1, residue_B2])

        # Create model
        model = MockModel([chain_A, chain_B])

        # Set get_structure to return a list containing the model
        mock_pdbparser.return_value.get_structure.return_value = [model]

        # Initialize the PDBAnalyser with the pdb_file_path
        self.analyser = PDBAnalyser(pdb_file_path)
        # Now calculate_padding_of_chains will use the mocked data
        self.analyser.calculate_padding_of_chains()

    def test_calculate_average_pae(self):
        pae_mtx = np.array([
            [0, 1, 2, 3],
            [1, 0, 3, 4],
            [2, 3, 0, 5],
            [3, 4, 5, 0]
        ])
        chain_1_residues = np.array([0, 1])
        chain_2_residues = np.array([0, 1])
        average_pae = self.analyser.calculate_average_pae(pae_mtx, 'A', 'B', chain_1_residues, chain_2_residues)
        expected_pae = 3.0  # Calculated as shown above
        self.assertEqual(average_pae, expected_pae)

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

    @patch('alphapulldown.analysis_pipeline.pdb_analyser.get_score_function')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.PDBIO')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.pose_from_pdb')
    def test_calculate_binding_energy(self, mock_pose_from_pdb, mock_pdbio, mock_get_score_function):
        # Mock the score function to return a fixed score
        mock_score_function = MagicMock()
        mock_score_function.return_value = 10.0
        mock_get_score_function.return_value = mock_score_function

        # Mock the pose_from_pdb to return a mock pose
        mock_pose = MagicMock()
        mock_pose_from_pdb.return_value = mock_pose

        # Run the method under test
        energy = self.analyser.calculate_binding_energy('A', 'B')
        self.assertEqual(energy, -10.0)  # 10(A_B)-10(A)-10(B) = -10

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

    @patch('alphapulldown.analysis_pipeline.pdb_analyser.subprocess.run')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.os.listdir')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.pd.read_csv')
    def test_run_and_summarise_pi_score_success(self, mock_read_csv, mock_listdir, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0, stderr=b'', stdout=b'')
        mock_listdir.return_value = ['filter_intf_features.csv']
        mock_read_csv.return_value = pd.DataFrame({'interface': ['A_B'], 'value': [1]})

        result = self.analyser.run_and_summarise_pi_score(
            work_dir='/fake/dir',
            pdb_path='dummy_path.pdb'
        )

        self.assertFalse(result.empty)
        self.assertIn('interface', result.columns)
        self.assertIn('value', result.columns)

    @patch('alphapulldown.analysis_pipeline.pdb_analyser.subprocess.run')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.os.listdir')
    def test_run_and_summarise_pi_score_no_csv(self, mock_listdir, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0, stderr=b'', stdout=b'')
        mock_listdir.return_value = []

        result = self.analyser.run_and_summarise_pi_score(
            work_dir='/fake/dir',
            pdb_path='dummy_path.pdb'
        )

        expected_df = self.analyser._default_dataframe()
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df)

    @patch('alphapulldown.analysis_pipeline.pdb_analyser.subprocess.run')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.pd.read_csv')
    def test_run_and_summarise_pi_score_subprocess_error(self, mock_read_csv, mock_subprocess_run):
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd='dummy command', output=b'', stderr=b'Error'
        )

        result = self.analyser.run_and_summarise_pi_score(
            work_dir='/fake/dir',
            pdb_path='dummy_path.pdb'
        )

        expected_df = self.analyser._default_dataframe()
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df)

    @patch('alphapulldown.analysis_pipeline.pdb_analyser.subprocess.run')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.os.listdir')
    @patch('alphapulldown.analysis_pipeline.pdb_analyser.pd.read_csv')
    def test_run_and_summarise_pi_score_empty_csv(self, mock_subprocess_run, mock_listdir, mock_read_csv):
        mock_subprocess_run.return_value = MagicMock(returncode=0, stderr=b'', stdout=b'')
        mock_listdir.return_value = ['filter_intf_features.csv']
        mock_read_csv.side_effect = pd.errors.EmptyDataError('No data')

        result = self.analyser.run_and_summarise_pi_score(
            work_dir='/fake/dir',
            pdb_path='dummy_path.pdb'
        )

        expected_df = self.analyser._default_dataframe()
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df)


if __name__ == '__main__':
    absltest.main()
