import unittest 
import numpy as np
import gzip,pickle,shutil
from alphafold.data.templates import _build_query_to_hit_index_mapping
from alphapulldown.multimeric_template_utils import create_template_hit,exctract_multimeric_template_features_for_single_chain
class TestMultimericTemplateFeatures(unittest.TestCase):
    def setUp(self):
        self.mmcif_file = "./test/test_data/true_multimer/3L4Q.cif"
        self.monomer1 = pickle.load(open("./test/test_data/true_multimer/features/3L4Q_A.pkl",'rb'))
        self.monomer2 = pickle.load(open("./test/test_data/true_multimer/features/3L4Q_C.pkl",'rb'))
        self.kalign_binary_path = shutil.which('kalign')
    
    def test_1_create_template_hit(self):
        template_hit = create_template_hit(index=1, name='3l4q_A',query=self.monomer1.sequence)
        self.assertEqual(self.monomer1.sequence,template_hit.hit_sequence)
    
    def test_2_build_mapping(self):
        template_hit = create_template_hit(index=1, name='3l4q_A',query=self.monomer1.sequence)
        expected_mapping = {i:i for i in range(len(self.monomer1.sequence))}
        mapping = _build_query_to_hit_index_mapping(template_hit.query,
                                                    template_hit.hit_sequence,
                                                    template_hit.indices_hit,
                                                    template_hit.indices_query,
                                                    self.monomer1.sequence)
        self.assertEqual(expected_mapping, mapping)
    
    def test_3_extract_multimeric_template_features(self):
        single_hit_result = exctract_multimeric_template_features_for_single_chain(self.monomer1.sequence,
                                                                                   pdb_id='3l4q',
                                                                                   chain_id='C',
                                                                                   mmcif_file=self.mmcif_file)
        self.assertIsNotNone(single_hit_result.features)

if __name__ == "__main__":
    unittest.main()