import unittest
from alphapulldown.objects import MultimericObject
import pickle
from alphapulldown.folding_backend.alphalink_backend import AlphaLinkBackend

class TestAlphaLink2Backend(unittest.TestCase):

    def setUp(self) -> None:
        self.monomer1 = pickle.load(open("./test/test_data/H1134_A.pkl",'rb'))
        self.monomer2 = pickle.load(open("./test/test_data/H1134_B.pkl",'rb'))
        self.multimericObj = MultimericObject([self.monomer1,self.monomer2])
        self.alphalink2_weights = '/g/alphafold/alphalink_weights/AlphaLink-Multimer_SDA_v3.pt'
        self.xl_info = "./test/test_data/test_xl_input.pkl.gz"
        return super().setUp()
    
    def test_1_initialise_folding_backend(self):
        beckend = AlphaLinkBackend
        model_config = beckend.setup(self.alphalink2_weights,
                                                             self.xl_info)
        beckend.predict(**model_config,multimeric_object = self.multimericObj,output_dir= "./test/test_data")

if __name__ == "__main__":
    unittest.main()