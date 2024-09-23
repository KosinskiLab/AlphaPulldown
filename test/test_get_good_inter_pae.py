from absl.testing import absltest
from unittest.mock import patch, mock_open, MagicMock
import os
import json
import pandas as pd
from alphapulldown.analysis_pipeline.get_good_inter_pae import obtain_seq_lengths, main

class TestGetGoodInterPae(absltest.TestCase):

    @patch('get_good_inter_pae.os.path.exists')
    @patch('get_good_inter_pae.PDBParser.get_structure')
    @patch('get_good_inter_pae.PPBuilder.build_peptides')
    def test_obtain_seq_lengths(self, mock_build_peptides, mock_get_structure, mock_exists):
        mock_exists.return_value = True
        mock_get_structure.return_value = MagicMock()
        mock_build_peptides.return_value = [MagicMock(get_sequence=MagicMock(return_value='AAAA'))]

        result = obtain_seq_lengths('/fake/dir')
        self.assertEqual(result, [4])

    @patch('get_good_inter_pae.os.listdir')
    @patch('get_good_inter_pae.os.path.isfile')
    @patch('get_good_inter_pae.PDBAnalyser')
    @patch('get_good_inter_pae.obtain_pae_and_iptm')
    @patch('get_good_inter_pae.obtain_seq_lengths')
    @patch('get_good_inter_pae.examine_inter_pae')
    @patch('get_good_inter_pae.obtain_mpdockq')
    @patch('get_good_inter_pae.pd.DataFrame.to_csv')
    def test_main(self, mock_to_csv, mock_obtain_mpdockq, mock_examine_inter_pae, mock_obtain_seq_lengths, mock_obtain_pae_and_iptm, mock_PDBAnalyser, mock_isfile, mock_listdir):
        mock_listdir.return_value = ['job1']
        mock_isfile.return_value = True
        mock_obtain_pae_and_iptm.return_value = (MagicMock(), 0.5)
        mock_obtain_seq_lengths.return_value = [100, 100]
        mock_examine_inter_pae.return_value = True
        mock_obtain_mpdockq.return_value = (0.5, {'A': [0.9] * 100, 'B': [0.9] * 100})
        mock_PDBAnalyser.return_value = MagicMock()

        with patch('builtins.open', mock_open(read_data=json.dumps({'order': ['model1'], 'iptm+ptm': {'model1': 0.8}}))):
            with patch('get_good_inter_pae.FLAGS', MagicMock(output_dir='/fake/dir', cutoff=5)):
                main([])

        mock_to_csv.assert_called_once()

if __name__ == '__main__':
    absltest.main()