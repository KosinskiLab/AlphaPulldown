from absl.testing import absltest, parameterized
import os
import pickle
import numpy as np
#from alphafold.common.residue_constants import ID_TO_HHBLITS_AA


class TestCreateMonomericObject(parameterized.TestCase):
    """A class that tests creation of a MonomericObject feature_dict."""

    def setUp(self) -> None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        self.ap_features = os.path.join(
            test_dir, "test_data", "predictions", "af_vs_ap")
        self.af_features = os.path.join(
            test_dir, "test_data", "predictions", "af_vs_ap", "A0A024R1R8"
        )

        # Load pickled monomer features
        with open(os.path.join(self.ap_features, 'A0A024R1R8_orig.pkl'), 'rb') as f:
            self.ap_feats = pickle.load(f).feature_dict

        # Load reference monomeric features
        with open(os.path.join(self.af_features, "features.pkl"), 'rb') as f:
            self.af_feats = pickle.load(f)

        # Example: if certain keys (like 'aatype') are known to be different,
        # skip them entirely in all scenarios:
        self.keys_to_skip_entirely = {
            # "aatype",  # Uncomment if you want to skip comparing 'aatype' at all
        }

    def test_monomeric_object(self):
        """Test that the multimeric features match the reference, with or without MSA pairing."""

        #
        # 1) Compare each key carefully, collecting mismatch info.
        #
        mismatch_info = []
        for k in self.af_feats.keys():
            if k in self.keys_to_skip_entirely:
                # Skip certain keys entirely (if you know they always mismatch).
                continue

            ref_val = self.af_feats[k]
            test_val = self.ap_feats[k]

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
                f"\nMismatch summary :\n" + "\n".join(mismatch_info)
            )
            self.fail(mismatch_summary)
            #print(mismatch_summary)
        # If we get here, everything matched
        # (except for keys that we explicitly skipped)
        print(f"All good!")



if __name__ == "__main__":
    absltest.main()
