import unittest 
from unifold.dataset import process_xl_input,calculate_offsets,create_xl_features
from alphafold.data.pipeline_multimer import _FastaChain
import numpy as np
import gzip,pickle
import torch
class TestCreateObjects(unittest.TestCase):
    def setUp(self) -> None:
        self.crosslink_info ="./test/test_data/test_xl_input.pkl.gz"
        self.asym_id = [1]*10 + [2]*25 + [3]*40
        self.chain_id_map = {
            "A":_FastaChain(sequence='',description='chain1'),
            "B":_FastaChain(sequence='',description='chain2'),
            "C":_FastaChain(sequence='',description='chain3')
        }
        return super().setUp()
    
    def test1_calculate_offsets(self):
        offsets = calculate_offsets(self.asym_id)
        offsets = offsets.tolist()
        expected_offsets = [0,10,35,75]
        self.assertEqual(offsets,expected_offsets)

    def test2_create_xl_inputs(self):
        offsets = calculate_offsets(self.asym_id)
        xl_pickle = pickle.load(gzip.open(self.crosslink_info,'rb'))
        xl = create_xl_features(xl_pickle,offsets,chain_id_map = self.chain_id_map)
        expected_xl = torch.tensor([[10,35,0.01],
                                    [3,27,0.01],
                                    [5,56,0.01],
                                    [20,65,0.01]])
        self.assertTrue(torch.equal(xl,expected_xl))

    

if __name__ == "__main__":
    unittest.main()