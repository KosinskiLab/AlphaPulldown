from absl.testing import absltest
from alphapulldown.objects import MultimericObject
import pickle
import numpy as np


class TestCreateMultimericObject(absltest.TestCase):
    """A class that test major functions of creating feature_dict of a MultimericObject object"""

    def setUp(self) -> None:
        self.monomer1 = pickle.load(open("./test/test_data/features/3L4Q_A.3L4Q.cif.A.pkl", "rb"))
        self.monomer2 = pickle.load(open("./test/test_data/features/3L4Q_C.3L4Q.pdb.C.pkl", "rb"))
    
    def test_1_initiate_default_multimericobject(self) -> MultimericObject:
        multimer_obj = MultimericObject([self.monomer1, self.monomer2])
        return multimer_obj
    
    def test_1_initiate_multimericobject_without_msa_pairing(self) -> MultimericObject:
        multimer_obj = MultimericObject([self.monomer1, self.monomer2],pair_msa=False)
        return multimer_obj

    def test_2_check_residue_indexes(self):
        multimer_obj = self.test_1_initiate_default_multimericobject()
        seq_1_length = self.monomer1.feature_dict['seq_length'][0]
        seq_2_length = self.monomer2.feature_dict['seq_length'][0]
        expected_residue_index=np.array(list(range(seq_1_length)) + list(range(seq_2_length)))
        self.assertTrue(np.array_equal(multimer_obj.feature_dict['residue_index'],expected_residue_index))

if __name__=="__main__":
    absltest.main()