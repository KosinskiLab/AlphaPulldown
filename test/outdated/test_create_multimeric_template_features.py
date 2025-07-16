from absl.testing import absltest
import pickle,shutil
from alphafold.data.templates import _build_query_to_hit_index_mapping
from alphapulldown.utils import multimeric_template_utils


class TestMultimericTemplateFeatures(absltest.TestCase):
    def setUp(self):
        self.mmcif_file = "./test/test_data/templates/3L4Q.cif"
        self.monomer1 = pickle.load(open("./test/test_data/features/3L4Q_A.3L4Q.cif.A.pkl",'rb'))
        self.monomer2 = pickle.load(open("./test/test_data/features/3L4Q_C.3L4Q.pdb.C.pkl",'rb'))
        self.kalign_binary_path = shutil.which('kalign')
        self.mmt_dir = './test/test_data/templates/'
        self.instruction_file = "./test/test_data/protein_lists/test_truemultimer.txt"
        self.data_dir = '/scratch/AlphaFold_DBs/2.3.2'

    @absltest.skip('attribute error')
    def test_1_create_template_hit(self):
        template_hit = multimeric_template_utils.create_template_hit(index=1, name='3l4q_A',query=self.monomer1.sequence)
        self.assertEqual(self.monomer1.sequence,template_hit.hit_sequence)

    @absltest.skip('attribute error')
    def test_2_build_mapping(self):
        template_hit = multimeric_template_utils.create_template_hit(index=1, name='3l4q_A',query=self.monomer1.sequence)
        expected_mapping = {i:i for i in range(len(self.monomer1.sequence))}
        mapping = _build_query_to_hit_index_mapping(template_hit.query,
                                                    template_hit.hit_sequence,
                                                    template_hit.indices_hit,
                                                    template_hit.indices_query,
                                                    self.monomer1.sequence)
        self.assertEqual(expected_mapping, mapping)
    
    def test_3_extract_multimeric_template_features(self):
        single_hit_result = multimeric_template_utils.extract_multimeric_template_features_for_single_chain(self.monomer1.sequence,
                                                                                   pdb_id='3L4Q',
                                                                                   chain_id='A',
                                                                                   mmcif_file=self.mmcif_file)
        self.assertIsNotNone(single_hit_result.features)
    
    def test_4_parse_instraction_file(self):
        """Test if the instruction csv table is parsed properly"""
        multimeric_template_meta = multimeric_template_utils.prepare_multimeric_template_meta_info(self.instruction_file,self.mmt_dir)
        self.assertIsInstance(multimeric_template_meta, dict)
        expected_dict = {"3L4Q_A":{"3L4Q.cif":"A"}, "3L4Q_C":{"3L4Q.pdb":"C"}}
        self.assertEqual(multimeric_template_meta,expected_dict)

if __name__ == "__main__":
    absltest.main()