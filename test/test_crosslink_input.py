import unittest 
from unifold.dataset import calculate_offsets,create_xl_features,bin_xl
from alphafold.data.pipeline_multimer import _FastaChain
import numpy as np
import gzip,pickle
import torch
class TestCreateObjects(unittest.TestCase):
    def setUp(self) -> None:
        self.crosslink_info ="./test/test_data/test_xl_input.pkl.gz"
        self.asym_id = torch.tensor([1]*10 + [2]*25 + [3]*40)
        self.chain_id_map = {
            "A":_FastaChain(sequence='',description='chain1'),
            "B":_FastaChain(sequence='',description='chain2'),
            "C":_FastaChain(sequence='',description='chain3')
        }
        self.bins = torch.arange(0,1.05,0.05)
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

    def test3_bin_xl(self):
        offsets = calculate_offsets(self.asym_id)
        xl_pickle = pickle.load(gzip.open(self.crosslink_info,'rb'))
        xl = create_xl_features(xl_pickle,offsets,chain_id_map = self.chain_id_map)
        num_res = len(self.asym_id)
        xl = bin_xl(xl,num_res)
        expected_xl = np.zeros((num_res,num_res,1))
        expected_xl[3,27,0] = expected_xl[27,3,0] = torch.bucketize(0.99,self.bins)
        expected_xl[10,35,0] = expected_xl[35,10,0] = torch.bucketize(0.99,self.bins)
        expected_xl[5,56,0] = expected_xl[56,5,0] = torch.bucketize(0.99,self.bins)
        expected_xl[20,65,0] = expected_xl[65,20,0] = torch.bucketize(0.99,self.bins)
        self.assertTrue(np.array_equal(xl,expected_xl))

if __name__ == "__main__":
    unittest.main()