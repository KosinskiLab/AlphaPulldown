from absl.testing import absltest, parameterized
import os
import pickle
import numpy as np

from alphafold.common.residue_constants import ID_TO_HHBLITS_AA, MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
from alphapulldown.objects import MonomericObject, MultimericObject

print(MAP_HHBLITS_AATYPE_TO_OUR_AATYPE)
class TestCreateMultimericObject(parameterized.TestCase):
    """A class that tests creation of a MultimericObject feature_dict."""

    def setUp(self) -> None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        self.ap_features = os.path.join(
            test_dir, "test_data", "predictions", "af_vs_ap")
        self.af_features = os.path.join(
            test_dir, "test_data", "predictions", "af_vs_ap", "A0A024R1R8+P61626_orig"
        )

        # Load pickled monomer features
        with open(os.path.join(self.ap_features, 'A0A024R1R8_orig.pkl'), 'rb') as f:
            self.monomer1 = pickle.load(f)
        with open(os.path.join(self.ap_features, 'P61626_orig.pkl'), 'rb') as f:
            self.monomer2 = pickle.load(f)

        # Load reference multimer features
        with open(os.path.join(self.af_features, "features.pkl"), 'rb') as f:
            self.af_multi_feats = pickle.load(f)

        # Keys that are allowed to differ if pair_msa=False
        self.allowed_diff_no_pair = {
            "bert_mask",
            "cluster_bias_mask",
            "deletion_matrix",
            "msa",
            "msa_mask",
        }

        # Example: if certain keys (like 'aatype') are known to be different,
        # skip them entirely in all scenarios:
        self.keys_to_skip_entirely = {
            # "aatype",  # Uncomment if you want to skip comparing 'aatype' at all
        }

    @parameterized.named_parameters(
        ("pair_msa_true", True),
        ("pair_msa_false", False),
    )
    def test_multimeric_object(self, pair_msa: bool):
        """Test that the multimeric features match the reference, with or without MSA pairing."""
        # Build the MultimericObject
        multi_obj = MultimericObject([self.monomer1, self.monomer2], pair_msa=pair_msa)
        multi_feats = multi_obj.feature_dict

        # Reference features
        ref_feats = self.af_multi_feats

        # 1) Check that both have the same set of keys
        self.assertCountEqual(
            ref_feats.keys(),
            multi_feats.keys(),
            f"Keys differ from reference when pair_msa={pair_msa}",
        )

        #
        # 2) Compare each key carefully, collecting mismatch info.
        #
        mismatch_info = []
        for k in ref_feats.keys():
            if k in self.keys_to_skip_entirely:
                # Skip certain keys entirely (if you know they always mismatch).
                continue

            ref_val = ref_feats[k]
            test_val = multi_feats[k]

            # If pair_msa=False, skip strict comparison for the "allowed_diff_no_pair" keys
            if (not pair_msa) and (k in self.allowed_diff_no_pair):
                continue

            # Compare differently for np.ndarray vs scalar
            if isinstance(ref_val, np.ndarray):
                # First check shape
                if ref_val.shape != test_val.shape:
                    mismatch_info.append(
                        f"[{k}] shape mismatch: ref {ref_val.shape} vs test {test_val.shape}"
                    )
                    continue  # Skip elementwise compare if shape differs

                # Check exact elementwise differences
                diff_mask = (ref_val != test_val)
                n_diff = np.count_nonzero(diff_mask)
                if n_diff > 0:
                    total = diff_mask.size
                    pct = 100.0 * n_diff / total
                    mismatch_info.append(
                        f"[{k}] {n_diff}/{total} elements differ ({pct:.1f}%)."
                    )
            else:
                # Non-array comparison:
                if ref_val != test_val:
                    mismatch_info.append(
                        f"[{k}] scalar mismatch: ref={ref_val}, test={test_val}"
                    )

        #
        # If there's any mismatch info accumulated, fail and show it all.
        #
        if mismatch_info:
            mismatch_summary = (
                f"\nMismatch summary (pair_msa={pair_msa}):\n" + "\n".join(mismatch_info)
            )
            print(mismatch_summary)
            self.fail(mismatch_summary)

        # If we get here, everything matched
        # (except for keys that we explicitly skipped)
        print(f"Multimeric features match the reference under pair_msa={pair_msa}.")

        #
        # (Optional) Dump shapes and MSAs for debugging
        #
        print(f"\n=== Reference features.pkl (pair_msa={pair_msa}) ===")

        OUR_AATYPE_TO_ID_HHBLITS_AA = {v: k for k, v in enumerate(MAP_HHBLITS_AATYPE_TO_OUR_AATYPE)}
        for k, v in sorted(ref_feats.items()):
            shape_str = v.shape if hasattr(v, "shape") else type(v)
            print(f"  {k}: {shape_str}")
            if k == 'msa':
                with open("af_msa.sto", 'w') as f:
                    f.write("# STOCKHOLM 1.0\n\n")
                    for i, row in enumerate(v):
                        seq = "".join(ID_TO_HHBLITS_AA[OUR_AATYPE_TO_ID_HHBLITS_AA[idx]] for idx in row)
                        f.write(f"seq_{i} {seq}\n")
                    f.write("//\n")

        print(f"\n=== MultimericObject features (pair_msa={pair_msa}) ===")
        for k, v in sorted(multi_feats.items()):
            shape_str = v.shape if hasattr(v, "shape") else type(v)
            print(f"  {k}: {shape_str}")
            if k == 'msa':
                suffix = "with" if pair_msa else "no"
                with open(f"ap_msa_{suffix}_pairing.sto", 'w') as f:
                    f.write("# STOCKHOLM 1.0\n\n")
                    for i, row in enumerate(v):
                        seq = "".join(ID_TO_HHBLITS_AA[OUR_AATYPE_TO_ID_HHBLITS_AA[idx]] for idx in row)
                        f.write(f"seq_{i} {seq}\n")
                    f.write("//\n")


if __name__ == "__main__":
    absltest.main()
