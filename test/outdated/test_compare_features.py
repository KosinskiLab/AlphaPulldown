"""Test that AlphaPulldown multimeric features replicate AlphaFold's own multimer features.

This test will:
1.  Generate *monomeric* pickles with ``create_individual_features.py`` (one per chain).
2.  Generate a *multimeric* ``features.pkl`` with the full ``run_alphafold.py`` multimer pipeline.
3.  Build an in‑memory ``MultimericObject`` from the monomer pickles (with and without MSA pairing).
4.  Compare the resulting feature‑dict to AlphaFold's reference feature‑dict, failing on any unexpected
    differences.

Prerequisites
-------------
* ``create_individual_features.py`` and ``run_alphafold.py`` must be importable from ``$PATH`` or the
  current conda/environment ``bin`` directory.
* All AlphaFold databases referenced by the commands *must* be present at the indicated locations.
* A CUDA‑capable GPU is required for the full AlphaFold multimer run.

If either command is missing or fails, the test will raise a clear ``RuntimeError``.
"""
from __future__ import annotations

import pathlib
import pickle
import subprocess
from typing import List

import numpy as np
from absl.testing import absltest, parameterized

# AlphaFold / AlphaPulldown imports
from alphafold.common.residue_constants import MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
from alphapulldown.objects import MultimericObject

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _run_shell(cmd: str, *, cwd: pathlib.Path | None = None) -> None:
    """Run *cmd* with ``subprocess.run`` and ``shell=True``.

    Aborts the whole test module on failure, because subsequent comparisons would
    be meaningless without the reference pickles.
    """
    print(f"[RUN] {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed with code {exc.returncode}: {cmd}") from exc


# -----------------------------------------------------------------------------
# Test case
# -----------------------------------------------------------------------------

class TestCreateMultimericObject(parameterized.TestCase):
    """Compare AlphaPulldown‑generated multimeric features to AlphaFold reference."""

    # ------------------------------------------------------------------
    # Paths & commands
    # ------------------------------------------------------------------
    _REPO_ROOT = pathlib.Path(__file__).resolve().parent
    _TEST_DATA = _REPO_ROOT / "test_data"
    _FASTAS_DIR = _TEST_DATA / "fastas"
    _PRED_DIR = _TEST_DATA / "predictions" / "af_vs_ap"
    _FASTA = _FASTAS_DIR / "A0A024R1R8+P61626_orig.fasta"

    _ALPHAFOLD_DB_232 = pathlib.Path("/g/alphafold/AlphaFold_DBs/2.3.2")

    # Mapping of monomer pickle filenames (after create_individual_features.py)
    _MONOMER_PICKLES = [
        _PRED_DIR / "A0A024R1R8_orig.pkl",
        _PRED_DIR / "P61626_orig.pkl",
    ]

    # AlphaFold multimer output (run_alphafold.py)
    _MULTIMER_PICKLE = _PRED_DIR / "A0A024R1R8+P61626_orig" / "features.pkl"

    # ------------------------------------------------------------------
    # Class‑level set‑up (runs once per test run, not per parameterisation)
    # ------------------------------------------------------------------
    @classmethod
    def setUpClass(cls) -> None:  # noqa: D401
        super().setUpClass()

        # Make sure the prediction folder exists.
        cls._PRED_DIR.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # 0. Wipe any pickles from a previous test run
        # ------------------------------------------------------------------
        for f in cls._MONOMER_PICKLES:
            if f.exists():
                f.unlink()
        if cls._MULTIMER_PICKLE.exists():
            cls._MULTIMER_PICKLE.unlink()

        # ------------------------------------------------------------------
        # 1. Generate *monomeric* pickles
        # ------------------------------------------------------------------
        create_cmd = (
            "create_individual_features.py "
            f"--output_dir={cls._PRED_DIR} "
            f"--fasta_paths={cls._FASTA} "
            "--max_template_date=2050-10-10 "
            "--use_precomputed_msas "
            f"--data_dir={cls._ALPHAFOLD_DB_232}"
        )
        _run_shell(create_cmd)

        # ------------------------------------------------------------------
        # 2. Generate *multimeric* features.pkl
        # ------------------------------------------------------------------
        run_cmd = (
            "run_alphafold.py "
            f"--output_dir={cls._PRED_DIR} "
            f"--fasta_paths={cls._FASTA} "
            "--max_template_date=2050-10-10 "
            "--model_preset=multimer "
            "--use_precomputed_msas "
            f"--data_dir={cls._ALPHAFOLD_DB_232}/ "
            f"--uniref90_database_path={cls._ALPHAFOLD_DB_232}/uniref90/uniref90.fasta "
            f"--mgnify_database_path={cls._ALPHAFOLD_DB_232}/mgnify/mgy_clusters_2022_05.fa "
            f"--template_mmcif_dir={cls._ALPHAFOLD_DB_232}/pdb_mmcif/mmcif_files "
            f"--obsolete_pdbs_path={cls._ALPHAFOLD_DB_232}/pdb_mmcif/obsolete.dat "
            "--use_gpu_relax=true "
            f"--bfd_database_path={cls._ALPHAFOLD_DB_232}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt "
            f"--uniref30_database_path={cls._ALPHAFOLD_DB_232}/uniref30/UniRef30_2023_02 "
            f"--pdb_seqres_database_path={cls._ALPHAFOLD_DB_232}/pdb_seqres/pdb_seqres.txt "
            f"--uniprot_database_path={cls._ALPHAFOLD_DB_232}/uniprot/uniprot.fasta "
            "--features_only"
        )
        _run_shell(run_cmd)

        # ------------------------------------------------------------------
        # 3. Load the freshly-generated pickles
        # ------------------------------------------------------------------
        with open(cls._MONOMER_PICKLES[0], "rb") as fh:
            cls.monomer1_feats = pickle.load(fh)
        with open(cls._MONOMER_PICKLES[1], "rb") as fh:
            cls.monomer2_feats = pickle.load(fh)
        with open(cls._MULTIMER_PICKLE, "rb") as fh:
            cls.af_multimer_feats = pickle.load(fh)

        # Sets used by the comparison logic
        cls._ALLOWED_DIFF_NO_PAIR = {
            "bert_mask",
            "cluster_bias_mask",
            "deletion_matrix",
            "msa",
            "msa_mask",
        }
        cls._SKIP_KEYS = set()
        cls._OUR_AATYPE_TO_ID = {
            v: k for k, v in enumerate(MAP_HHBLITS_AATYPE_TO_OUR_AATYPE)
        }

    # ------------------------------------------------------------------
    # The *actual* parameterised test
    # ------------------------------------------------------------------

    @parameterized.named_parameters(("pair_msa_true", True), ("pair_msa_false", False))
    def test_multimeric_object(self, pair_msa: bool):  # noqa: D401
        """Check that feature‑dicts are identical to AlphaFold reference."""
        # Build the Multi object from monomer pickles.
        multi_obj = MultimericObject(
            [self.monomer1_feats, self.monomer2_feats], pair_msa=pair_msa
        )
        multi_feats = multi_obj.feature_dict
        ref_feats = self.af_multimer_feats

        # 1) Key set equality ---------------------------------------------------
        self.assertCountEqual(ref_feats.keys(), multi_feats.keys())

        # 2) Element‑wise comparison -------------------------------------------
        mismatches: List[str] = []
        for k in ref_feats.keys():
            if k in self._SKIP_KEYS:
                continue  # completely ignore this key

            if (not pair_msa) and (k in self._ALLOWED_DIFF_NO_PAIR):
                continue  # differences expected here when pair_msa == False

            ref_val, test_val = ref_feats[k], multi_feats[k]

            if isinstance(ref_val, np.ndarray):
                if ref_val.shape != test_val.shape:
                    mismatches.append(
                        f"[{k}] shape mismatch: {ref_val.shape} vs {test_val.shape}"
                    )
                    continue
                diff_mask = ref_val != test_val
                n_diff = int(np.count_nonzero(diff_mask))
                if n_diff:
                    total = int(diff_mask.size)
                    pct = 100.0 * n_diff / total
                    mismatches.append(
                        f"[{k}] {n_diff}/{total} elements differ ({pct:.1f} %)"
                    )
            else:
                if ref_val != test_val:
                    mismatches.append(f"[{k}] scalar mismatch: {ref_val!r} vs {test_val!r}")

        if mismatches:
            self.fail(
                "\n".join(
                    [f"Mismatch summary (pair_msa={pair_msa}):"] + mismatches
                )
            )

    # ------------------------------------------------------------------
    # Convenience: human‑readable dump (only when running this file directly)
    # ------------------------------------------------------------------
    def _dump_feature_shapes(self, feats: dict, tag: str) -> None:
        print(f"\n=== {tag} ===")
        for k, v in sorted(feats.items()):
            shape = v.shape if hasattr(v, "shape") else type(v)
            print(f"{k:>25}: {shape}")

    @classmethod
    def main(cls) -> None:  # pragma: no cover – manual helper
        """CLI helper to debug shapes/MSAs without running pytest."""
        tester = cls("_dummy_method")  # type: ignore[arg-type]
        tester._dump_feature_shapes(cls.af_multimer_feats, "Reference AlphaFold feats")
        tester._dump_feature_shapes(
            MultimericObject(
                [cls.monomer1_feats, cls.monomer2_feats], pair_msa=True
            ).feature_dict,
            "AlphaPulldown feats (pair_msa=True)",
        )


if __name__ == "__main__":
    # Allow running as ``python test_compare_features.py`` for quick debugging.
    absltest.main()
