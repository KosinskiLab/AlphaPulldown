import shutil
import tempfile
import unittest
import sys
import os
import torch
from unifold.modules.alphafold import AlphaFold
from unifold.alphalink_inference import prepare_model_runner
from unifold.alphalink_inference import alphalink_prediction
from unifold.dataset import process_ap
from unifold.config import model_config
from alphapulldown.utils import create
from alphapulldown.run_multimer_jobs import predict_individual_jobs,create_custom_jobs

class _TestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.crosslink_file_path = os.path.join(os.path.dirname(__file__),"test_data/example_crosslink.pkl.gz")
        self.config_data_model_name = 'model_5_ptm_af2'
        self.config_alphafold_model_name = 'multimer_af2_crop'

class TestCrosslinkInference(_TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.output_dir = tempfile.mkdtemp()
        self.monomer_object_path = os.path.join(os.path.dirname(__file__),"test_data/")
        self.protein_list = os.path.join(os.path.dirname(__file__),"test_data/example_crosslinked_pair.txt")
        self.alphalink_weight = '/g/alphafold/alphalink_weights/AlphaLink-Multimer_SDA_v3.pt'
        self.multimerobject = create_custom_jobs(self.protein_list,self.monomer_object_path,job_index=1,pair_msa=True)[0]
        
    def test1_process_features(self):
        """Test whether the PyTorch model of AlphaLink can be initiated successfully"""
        configs = model_config(self.config_data_model_name)
        processed_features,_ = process_ap(config=configs.data,
                                          features=self.multimerobject.feature_dict,
                                          mode="predict",labels=None,
                                          seed=42,batch_idx=None,
                                          data_idx=None,is_distillation=False,
                                          chain_id_map = self.multimerobject.chain_id_map,
                                          crosslinks = self.crosslink_file_path
                                          )
                
    def test2_load_AlphaLink_weights(self):
        """This is testing weither loading the PyTorch checkpoint is sucessfull"""
        if torch.cuda.is_available():
            model_device = 'cuda:0'
        else:
            model_device = 'cpu'

        config = model_config(self.config_alphafold_model_name)
        model = AlphaFold(config)
        state_dict = torch.load(self.alphalink_weight)["ema"]["params"]
        state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(model_device)

    def test3_test_inference(self):
        if torch.cuda.is_available():
            model_device = 'cuda:0'
        else:
            model_device = 'cpu'
        model = prepare_model_runner(self.alphalink_weight,model_device=model_device)


if __name__ == '__main__':
    unittest.main()