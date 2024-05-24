import unittest
import tempfile
from alphapulldown.objects import MultimericObject
import pickle
from alphapulldown.folding_backend.alphalink_backend import AlphaLinkBackend

class TestAlphaLink2Backend(unittest.TestCase):

    def setUp(self) -> None:
        self.monomer1 = pickle.load(open("./test/test_data/H1142_A.pkl",'rb'))
        self.monomer2 = pickle.load(open("./test/test_data/H1142_B.pkl",'rb'))
        self.multimericObj = MultimericObject([self.monomer1,self.monomer2])
        self.alphalink2_weights = '/g/alphafold/alphalink_weights/AlphaLink-Multimer_SDA_v3.pt'
        self.xl_info = "./test/test_data/test_xl_input.pkl.gz"
        return super().setUp()
    
    def test_1_initialise_folding_backend(self):
        """Test initialising the backend"""
        beckend = AlphaLinkBackend
        model_config = beckend.setup(self.alphalink2_weights,crosslinks=self.xl_info)
    
    def test_2_test_prediction(self):
        """Test predicting the structure"""
        beckend = AlphaLinkBackend
        model_config = beckend.setup(self.alphalink2_weights,crosslinks=self.xl_info)
        with tempfile.TemporaryDirectory() as output_dir:
            objects_to_model = [{self.multimericObj: output_dir}]
            predicted_jobs = beckend.predict(**model_config, crosslinks=self.xl_info, objects_to_model=objects_to_model)
            for predicted_job in predicted_jobs:
                object_to_model, prediction_results = next(iter(predicted_job.items()))
                beckend.postprocess(prediction_results=prediction_results,
                                    multimeric_object=object_to_model, 
                                    output_dir=prediction_results['output_dir'])
                
            # test resume 
            predicted_jobs = beckend.predict(**model_config, crosslinks=self.xl_info, objects_to_model=objects_to_model)
            for predicted_job in predicted_jobs:
                object_to_model, prediction_results = next(iter(predicted_job.items()))
                beckend.postprocess(prediction_results=prediction_results,
                                    multimeric_object=object_to_model, 
                                    output_dir=prediction_results['output_dir'])

if __name__ == "__main__":
    unittest.main()