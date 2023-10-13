import unittest 
from unifold.datset import process_xl_input
from alphafold.data.pipeline_multimer import _FastaChain
class TestCreateObjects(unittest.TestCase):
    def setUp(self) -> None:
        self.crosslink_info ="./test_data/test_crosslink_info.pkl.gz"
        self.asym_id = [1]*10 + [2]*25 + [3]*40
        self.chain_id_map = {
            "A":_FastaChain(sequence='',description='chain1'),
            "B":_FastaChain(sequence='',description='chain2'),
            "C":_FastaChain(sequence='',description='chain3')
        }
        return super().setUp()