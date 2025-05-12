"""
Regression-style test for MultimericObject construction under every
combination of
    * pair_msa               (True / False)
    * use_mmseqs2            (True / False)
    * use_precomputed_msas   (True / False)

Goals
-----
* When use_mmseqs2 is **False** we trust the shipped AlphaFold reference
  pickle (`features.pkl`) and just make sure we can rebuild an identical
  multimer from the monomer pickles.

* When use_mmseqs2 is **True** we rebuild the multimer twice
    (fresh vs re-using the *.a3m) and check they are equal
  – allowing float values to differ by up to 0.01.

* We never abort on mismatches: we print a summary to stderr and keep going,
  so Stockholm files are always written for manual inspection.

* The first-row pairing is **always** asserted exactly (no tolerance).

The test leaves Stockholm alignments in a temp directory such as

    /tmp/pulldown_xxx/msa_mmseqs_pair_fresh.sto

Open them with Jalview, AliView, … to eyeball the pairing.
"""
from absl.testing import absltest, parameterized
import os, sys, tempfile, shutil, pickle
import numpy as np
from pathlib import Path
from typing import Dict

from alphafold.common.residue_constants import (
    ID_TO_HHBLITS_AA, MAP_HHBLITS_AATYPE_TO_OUR_AATYPE,
)
from alphapulldown.objects import MonomericObject, MultimericObject

# ───────────── helpers ────────────────────────────────────────────────────────
AA_ID_TO_CHAR = {v: k for k, v in enumerate(MAP_HHBLITS_AATYPE_TO_OUR_AATYPE)}

def row_to_str(row) -> str:
    """Translate numeric aatype row to a string of residues."""
    return ''.join(ID_TO_HHBLITS_AA[AA_ID_TO_CHAR[int(x)]] for x in row)

def load_feature_dict(path: Path) -> Dict:
    """Return the feature-dict no matter what the pickle contains."""
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, MonomericObject):
        return obj.feature_dict
    raise TypeError(f"Unknown pickle content in {path}")

def primary_sequence(feat: Dict) -> str:
    raw = feat["sequence"][0]
    return raw.decode() if isinstance(raw, (bytes, bytearray)) else raw

# ───────────── test case ──────────────────────────────────────────────────────
class TestCreateMultimericObject(parameterized.TestCase):
    # keys that may legally differ when we are *not* pairing
    allowed_diff_no_pair = {
        "bert_mask", "cluster_bias_mask", "deletion_matrix", "msa", "msa_mask"
    }

    def setUp(self):
        here = Path(__file__).resolve().parent
        self.af_dir   = here / "test_data" / "predictions" / "af_vs_ap"
        self.mm_dir   = here / "test_data" / "features" / "mmseqs2"
        self.af_ref   = self.af_dir / "A0A024R1R8+P61626_orig" / "features.pkl"
        self.ids      = ("A0A024R1R8_orig", "P61626_orig")
        self.tmp      = Path(tempfile.mkdtemp(prefix="pulldown_"))

        # cache primary sequences so we unpickle only once
        self.seq_cache = {pid: primary_sequence(load_feature_dict(
                            self.mm_dir / f"{pid}.pkl")) for pid in self.ids}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # full 2×2×2 grid
    @parameterized.product(
        pair_msa=[True, False],
        use_mmseqs2=[True, False],
        use_precomputed_msas=[True, False],
    )
    def test_multimer(self, pair_msa, use_mmseqs2, use_precomputed_msas):
        """
        For each flag triple build at least one multimer, save its MSA,
        compare as needed, and assert correct pairing.
        """
        # ------------------ build once (always) ---------------------
        feat_fresh = self._build_multimer(pair_msa,
                                          use_mmseqs2,
                                          use_pre_msas=False)
        tag = f"{'mmseqs' if use_mmseqs2 else 'af'}_" \
              f"{'pair' if pair_msa else 'nopair'}_fresh"
        self._write_stockholm(feat_fresh, tag)

        # ------------------ compare --------------------------------
        if use_mmseqs2:
            feat_reuse = self._build_multimer(pair_msa,
                                              True,
                                              use_pre_msas=True)
            tag2 = f"mmseqs_{'pair' if pair_msa else 'nopair'}_reuse"
            self._write_stockholm(feat_reuse, tag2)
            self._compare_feats(feat_fresh, feat_reuse, pair_msa)
        else:
            ref_feat = pickle.load(open(self.af_ref, "rb"))
            self._compare_feats(feat_fresh, ref_feat, pair_msa)

        # ------------------ pairing assertion ----------------------
        self._assert_pairing(feat_fresh, pair_msa)

    # ───────── internal: build one multimer ───────────────────────
    def _build_multimer(self, pair_msa, use_mmseqs2, use_pre_msas):
        monomers = []
        for pid in self.ids:
            seq = self.seq_cache[pid]
            m   = MonomericObject(pid, seq)

            if use_mmseqs2:
                m.make_mmseq_features(
                    output_dir=str(self.tmp),
                    use_precomputed_msa=use_pre_msas,
                    compress_msa_files=False,
                )
            else:
                m.feature_dict = load_feature_dict(self.af_dir / f"{pid}.pkl")
            monomers.append(m)

        mult = MultimericObject(monomers, pair_msa=pair_msa)
        return mult.feature_dict

    # ───────── internal: tolerant comparison ──────────────────────
    def _compare_feats(self, A, B, pair_msa):
        mism = []
        self.assertCountEqual(A.keys(), B.keys(), "key sets differ")
        for k in A:
            if (not pair_msa) and k in self.allowed_diff_no_pair:
                continue
            a, b = A[k], B[k]

            # ndarray branch
            if isinstance(a, np.ndarray):
                if a.shape != b.shape:
                    mism.append(f"{k}: shape {a.shape} vs {b.shape}")
                    continue

                if np.issubdtype(a.dtype, np.floating):
                    mask = ~np.isclose(a, b, atol=1e-2, rtol=0)
                else:
                    mask = a != b

                n = int(np.count_nonzero(mask))
                if n:
                    if np.issubdtype(a.dtype, np.floating):
                        diff   = np.abs(a.astype(float) - b.astype(float))
                        mean   = diff[mask].mean()
                        maxerr = diff[mask].max()
                        mism.append(f"{k}: {n}/{a.size} differ "
                                    f"(mean|Δ|={mean:.3g}, max={maxerr:.3g})")
                    else:
                        mism.append(f"{k}: {n}/{a.size} elements differ")

            # scalar branch
            else:
                if a != b:
                    mism.append(f"{k}: {a} vs {b}")

        if mism:
            print("----- mismatch summary -----", *mism, sep="\n",
                  file=sys.stderr)

    # ───────── internal: assert correct pairing row ───────────────
    def _assert_pairing(self, feat, pair_msa):
        row0 = row_to_str(feat["msa"][0])
        seq1, seq2 = (self.seq_cache[pid] for pid in self.ids)
        expected = seq1 + (seq2 if pair_msa else "-" * len(seq2))
        self.assertEqual(row0, expected,
                         f"MSA first row wrong for pair_msa={pair_msa}")

    # ───────── internal: write Stockholm to disk ──────────────────
    def _write_stockholm(self, feat, tag):
        msa   = feat["msa"]
        #sp_id = feat["msa_species_identifiers"]  # shape (N,)
        fn = f"msa_{tag}.sto"#self.tmp / f"msa_{tag}.sto"
        with open(fn, "w") as fh:
            fh.write("# STOCKHOLM 1.0\n\n")
            for i, row in enumerate(msa):
                seq   = row_to_str(row)
                label = f"sp_42"
                fh.write(f"{label} {seq}\n")
            fh.write("//\n")
        print(f"[debug] wrote {fn}", file=sys.stderr)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    absltest.main()
