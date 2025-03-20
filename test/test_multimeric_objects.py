from absl.testing import absltest
from alphapulldown.objects import MonomericObject, MultimericObject
import pickle
import numpy as np
import os
from alphafold.common.residue_constants import ID_TO_HHBLITS_AA

class TestCreateMultimericObject(absltest.TestCase):
    """A class that tests creation of a MultimericObject feature_dict."""

    def setUp(self) -> None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        self.ap_features = os.path.join(test_dir, "test_data", "features")
        self.af_features = os.path.join(
            test_dir, "test_data", "predictions", "af_vs_ap", "A0A024R1R8+P61626_orig"
        )

        # Load pickled monomer features
        with open(os.path.join(self.ap_features, 'P61626.pkl'), 'rb') as f:
            self.monomer1 = pickle.load(f)
        with open(os.path.join(self.ap_features, 'A0A024R1R8.pkl'), 'rb') as f:
            self.monomer2 = pickle.load(f)

        # Load reference multimer features
        with open(os.path.join(self.af_features, "features.pkl"), 'rb') as f:
            self.af_multi_feats = pickle.load(f)

    def test_1_multimer_with_pairing(self):
        multimer_obj = MultimericObject([self.monomer1, self.monomer2], False)
        alphafold_feats = self.af_multi_feats
        ap_no_pairing = multimer_obj.feature_dict
        multimer_obj2 = MultimericObject([self.monomer1, self.monomer2], True)
        ap_with_pairing = multimer_obj2.feature_dict
        # Print shapes/types from features.pkl
        print("\n=== features.pkl keys ===")
        for k, v in sorted(alphafold_feats.items()):
            shape_str = v.shape if hasattr(v, "shape") else type(v)
            print(f"  {k}: {shape_str}")
            if k == 'msa':
                with open("af_msa.sto", 'w') as f:
                    f.write("# STOCKHOLM 1.0\n\n")
                    for i, row in enumerate(v):
                        seq = "".join(ID_TO_HHBLITS_AA[idx] for idx in row)
                        f.write(f"seq_{i} {seq}\n")
                    f.write("//\n")

        # Print shapes/types from the MultimericObject
        print("\n=== multimeric_objects_features.pkl no pairing===")
        for k, v in sorted(ap_no_pairing.items()):
            shape_str = v.shape if hasattr(v, "shape") else type(v)
            print(f"  {k}: {shape_str}")
            if k == 'msa':
                with open("ap_msa_no_pairing.sto", 'w') as f:
                    f.write("# STOCKHOLM 1.0\n\n")
                    for i, row in enumerate(v):
                        seq = "".join(ID_TO_HHBLITS_AA[idx] for idx in row)
                        f.write(f"seq_{i} {seq}\n")
                    f.write("//\n")


        print("\n=== multimeric_objects_features.pkl with pairing ===")
        for k, v in sorted(ap_with_pairing.items()):
            shape_str = v.shape if hasattr(v, "shape") else type(v)
            print(f"  {k}: {shape_str}")
            if k == 'msa':
                with open("ap_msa_with_pairing.sto", 'w') as f:
                    f.write("# STOCKHOLM 1.0\n\n")
                    for i, row in enumerate(v):
                        seq = "".join(ID_TO_HHBLITS_AA[idx] for idx in row)
                        f.write(f"seq_{i} {seq}\n")
                    f.write("//\n")


if __name__ == "__main__":
    absltest.main()