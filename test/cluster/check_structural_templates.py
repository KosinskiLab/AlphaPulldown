#!/usr/bin/env python
"""GPU smoke test for the whole structural-template chain.

Runs the real thing end to end: ESMFold folds a query on a GPU, Foldseek searches
that structure against a real local structure database, and AlphaFold 2's own
featuriser turns the hits into a ``template_*`` feature block. The assertions are
about that block being *structurally valid* -- right keys, shapes, dtypes,
non-empty, and a residue mapping that stays inside the query -- because a broken
offset or a mis-shaped feature is accepted silently by everything downstream and
only shows up as a bad model.

Nothing here runs in CI. It skips cleanly whenever the GPU, the Foldseek binary,
the structure database, the ESMFold weights or the mmCIF directory is missing.

Configuration, all environment variables:

- ``FOLDSEEK_BINARY``            Foldseek executable (default: found on PATH)
- ``FOLDSEEK_DATABASE_PATH``     prefix of a Foldseek database of PDB chains
- ``FOLDSEEK_DATABASE_ID``       its immutable build name (default: a placeholder)
- ``ESMFOLD_MODEL_DIR``          local ESMFold checkpoint directory
- ``ALPHAFOLD_DATA_DIR``         AlphaFold 2 databases; supplies the mmCIF directory
- ``TEMPLATE_MMCIF_DIR``         overrides the mmCIF directory directly
- ``RUN_GPU_FUNCTIONAL_TESTS``   set to 1 to run where a GPU is present

Submit it with ``python test/cluster/run_structural_templates.py``.
"""

from __future__ import annotations

import json
import logging
import lzma
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
from absl.testing import absltest, parameterized

pytestmark = [pytest.mark.cluster, pytest.mark.gpu, pytest.mark.external_tools]

import alphapulldown  # noqa: E402
from alphapulldown.structural_templates import (  # noqa: E402
    EsmfoldSettings,
    EsmfoldStructurePredictor,
    FoldseekSearchSettings,
    FoldseekTemplateSearcher,
    PredictedStructureCache,
    StructureDatabaseSpec,
    SubprocessFoldseekProcess,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "test"

# Short enough to fold quickly, long enough that a template alignment is
# meaningful. Ubiquitin; any small well-represented protein would do.
QUERY_NAME = "ubiquitin"
QUERY_SEQUENCE = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)

MAX_TEMPLATE_DATE = os.getenv("MAX_TEMPLATE_DATE", "2024-01-01")

TEMPLATE_FEATURE_SPECS = {
    "template_aatype": (np.floating, 3),
    "template_all_atom_masks": (np.floating, 3),
    "template_all_atom_positions": (np.floating, 4),
    "template_domain_names": (np.object_, 1),
    "template_sequence": (np.object_, 1),
    "template_sum_probs": (np.floating, 2),
}


# --------------------------------------------------------------------------- #
#                              skip conditions                                #
# --------------------------------------------------------------------------- #
def _has_nvidia_gpu() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"], capture_output=True, text=True, check=False
        )
    except OSError:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _foldseek_binary() -> str | None:
    binary = os.getenv("FOLDSEEK_BINARY") or shutil.which("foldseek")
    return binary if binary and Path(binary).is_file() else None


def _template_mmcif_dir() -> Path | None:
    configured = os.getenv("TEMPLATE_MMCIF_DIR")
    if configured:
        directory = Path(configured)
    else:
        data_dir = os.getenv("ALPHAFOLD_DATA_DIR")
        if not data_dir:
            return None
        directory = Path(data_dir) / "pdb_mmcif" / "mmcif_files"
    return directory if directory.is_dir() else None


def _skip_reason() -> str | None:
    """One place that decides whether this smoke test can run at all."""
    if os.getenv("RUN_GPU_FUNCTIONAL_TESTS", "").lower() not in ("1", "true", "yes"):
        if os.getenv("CI", "").lower() in ("1", "true", "yes"):
            return "GPU smoke tests are disabled on CI."
        if not _has_nvidia_gpu():
            return "A GPU is required; set RUN_GPU_FUNCTIONAL_TESTS=1 to override."
    if _foldseek_binary() is None:
        return "Foldseek is not installed; set FOLDSEEK_BINARY."
    if not os.getenv("FOLDSEEK_DATABASE_PATH"):
        return "Set FOLDSEEK_DATABASE_PATH to a local Foldseek structure database."
    if not Path(f"{os.environ['FOLDSEEK_DATABASE_PATH']}.index").is_file():
        return (
            "FOLDSEEK_DATABASE_PATH does not look like a Foldseek database "
            f"({os.environ['FOLDSEEK_DATABASE_PATH']}.index is missing)."
        )
    model_dir = os.getenv("ESMFOLD_MODEL_DIR")
    if not model_dir or not Path(model_dir).is_dir():
        return "Set ESMFOLD_MODEL_DIR to a local ESMFold checkpoint directory."
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        return f"ESMFold needs torch and transformers: {exc}"
    if _template_mmcif_dir() is None:
        return "Set ALPHAFOLD_DATA_DIR or TEMPLATE_MMCIF_DIR to an mmCIF directory."
    return None


# --------------------------------------------------------------------------- #
#                                 helpers                                     #
# --------------------------------------------------------------------------- #
def _searcher(
    cache_dir: Path, scratch_dir: Path
) -> tuple[FoldseekTemplateSearcher, PredictedStructureCache]:
    structures = PredictedStructureCache(
        cache_dir=cache_dir,
        predictor=EsmfoldStructurePredictor(
            EsmfoldSettings(
                model_dir=Path(os.environ["ESMFOLD_MODEL_DIR"]),
                device=os.getenv("ESMFOLD_DEVICE", "cuda"),
                chunk_size=(
                    int(os.environ["ESMFOLD_CHUNK_SIZE"])
                    if os.getenv("ESMFOLD_CHUNK_SIZE")
                    else None
                ),
            )
        ),
    )
    searcher = FoldseekTemplateSearcher(
        settings=FoldseekSearchSettings(
            database=StructureDatabaseSpec(
                name="foldseek",
                path=Path(os.environ["FOLDSEEK_DATABASE_PATH"]),
                identifier=os.getenv("FOLDSEEK_DATABASE_ID", "cluster-smoke-db"),
            ),
            temp_dir=scratch_dir,
            cache_dir=cache_dir,
            threads=int(os.getenv("FOLDSEEK_THREADS", "4")),
        ),
        structures=structures,
        foldseek_process=SubprocessFoldseekProcess(_foldseek_binary()),
    )
    return searcher, structures


def _assert_valid_template_features(
    case: unittest.TestCase, features: dict, query_length: int
) -> int:
    """Assert an AlphaFold 2 template block is well formed, and return its depth."""
    for name in TEMPLATE_FEATURE_SPECS:
        case.assertIn(name, features, f"missing template feature {name}")

    depth = int(np.asarray(features["template_domain_names"]).shape[0])
    case.assertGreater(depth, 0, "the structural search produced no templates")

    for name, (kind, dimensions) in TEMPLATE_FEATURE_SPECS.items():
        array = np.asarray(features[name])
        case.assertEqual(array.ndim, dimensions, f"{name} has shape {array.shape}")
        case.assertEqual(array.shape[0], depth, f"{name} has shape {array.shape}")
        case.assertTrue(
            np.issubdtype(array.dtype, kind), f"{name} has dtype {array.dtype}"
        )

    aatype = np.asarray(features["template_aatype"])
    masks = np.asarray(features["template_all_atom_masks"])
    positions = np.asarray(features["template_all_atom_positions"])
    case.assertEqual(aatype.shape[1], query_length)
    case.assertEqual(aatype.shape[2], 22)
    case.assertEqual(masks.shape[1:], (query_length, 37))
    case.assertEqual(positions.shape[1:], (query_length, 37, 3))
    case.assertTrue(np.all(np.isfinite(positions)), "template positions are not finite")

    for index in range(depth):
        modelled = masks[index].sum(axis=-1) > 0
        count = int(modelled.sum())
        # A template that covers nothing, or claims to cover residues the query
        # does not have, is exactly what a broken alignment offset produces.
        case.assertGreater(count, 0, f"template {index} maps no residue at all")
        case.assertLessEqual(count, query_length)
        case.assertTrue(
            np.allclose(positions[index][~modelled], 0.0),
            f"template {index} has coordinates where nothing is modelled",
        )
        one_hot = aatype[index][modelled]
        case.assertTrue(
            np.allclose(one_hot.sum(axis=-1), 1.0),
            f"template {index} aatype is not one-hot where it is modelled",
        )
    return depth


# --------------------------------------------------------------------------- #
#                                  tests                                      #
# --------------------------------------------------------------------------- #
class TestStructuralTemplates(parameterized.TestCase):
    """ESMFold on a GPU, then Foldseek, then real AlphaFold 2 featurisation."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        reason = _skip_reason()
        if reason:
            raise unittest.SkipTest(reason)
        cls.workspace = Path(tempfile.mkdtemp(prefix="structural_templates_"))

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        workspace = getattr(cls, "workspace", None)
        if workspace and workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)

    def test_esmfold_and_foldseek_produce_valid_template_features(self):
        """The full chain, in process, ending in an AlphaFold 2 feature block."""
        from alphafold.data import templates

        searcher, structures = _searcher(
            cache_dir=self.workspace / "cache",
            scratch_dir=self.workspace / "scratch",
        )
        output = searcher.query(f">{QUERY_NAME}\n{QUERY_SEQUENCE}\n")
        self.assertTrue(output.strip(), "Foldseek returned no alignments")

        hits = searcher.get_template_hits(output, QUERY_SEQUENCE)
        self.assertTrue(hits, "no Foldseek hit survived parsing")
        for hit in hits:
            # Every hit has to name a chain the featuriser can actually open.
            pdb_id, chain_id = templates._get_pdb_id_and_chain(hit)
            self.assertTrue(chain_id)
            self.assertTrue(
                (Path(_template_mmcif_dir()) / f"{pdb_id}.cif").is_file(),
                f"hit {hit.name} has no mmCIF file in the template directory",
            )

        featuriser = templates.HhsearchHitFeaturizer(
            mmcif_dir=str(_template_mmcif_dir()),
            max_template_date=MAX_TEMPLATE_DATE,
            max_hits=20,
            kalign_binary_path=shutil.which("kalign") or "kalign",
            release_dates_path=None,
            obsolete_pdbs_path=None,
        )
        result = featuriser.get_templates(
            query_sequence=QUERY_SEQUENCE, hits=list(hits)
        )

        depth = _assert_valid_template_features(
            self, dict(result.features), len(QUERY_SEQUENCE)
        )
        logger.info("Structural search produced %d usable template(s)", depth)

        # The expensive step must not be repeated on a second query.
        self.assertTrue(
            structures.cached(QUERY_SEQUENCE),
            "the predicted structure was not cached",
        )

    def test_the_feature_cli_writes_features_with_structural_templates(self):
        """The same chain through create_individual_features.py, on the CLI."""
        data_dir = os.getenv("ALPHAFOLD_DATA_DIR")
        if not data_dir or not Path(data_dir).is_dir():
            self.skipTest("Set ALPHAFOLD_DATA_DIR to run the CLI leg of this test.")

        output_dir = self.workspace / "features"
        output_dir.mkdir(parents=True, exist_ok=True)
        fasta = self.workspace / "query.fasta"
        fasta.write_text(f">{QUERY_NAME}\n{QUERY_SEQUENCE}\n", encoding="utf-8")

        script = Path(alphapulldown.__path__[0]) / "scripts" / "create_individual_features.py"
        command = [
            sys.executable,
            str(script),
            f"--fasta_paths={fasta}",
            f"--data_dir={data_dir}",
            f"--output_dir={output_dir}",
            f"--max_template_date={MAX_TEMPLATE_DATE}",
            # Templates are the point here; a full MSA run is not.
            "--skip_msa",
            "--save_msa_files=True",
            "--use_foldseek_templates",
            f"--foldseek_binary_path={_foldseek_binary()}",
            f"--foldseek_database_path={os.environ['FOLDSEEK_DATABASE_PATH']}",
            f"--foldseek_database_id={os.getenv('FOLDSEEK_DATABASE_ID', 'cluster-smoke-db')}",
            f"--esmfold_model_dir={os.environ['ESMFOLD_MODEL_DIR']}",
            f"--esmfold_device={os.getenv('ESMFOLD_DEVICE', 'cuda')}",
            f"--structural_template_cache_dir={self.workspace / 'cache'}",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            self.fail(
                "create_individual_features.py failed:\n"
                f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
            )

        pickle_path = output_dir / f"{QUERY_NAME}.pkl"
        if not pickle_path.is_file():
            pickle_path = output_dir / f"{QUERY_NAME}.pkl.xz"
        self.assertTrue(pickle_path.is_file(), f"no feature pickle in {output_dir}")

        opener = lzma.open if pickle_path.suffix == ".xz" else open
        with opener(pickle_path, "rb") as handle:
            monomer = pickle.load(handle)
        features = getattr(monomer, "feature_dict", monomer)
        _assert_valid_template_features(self, dict(features), len(QUERY_SEQUENCE))

        # The searcher's own hits file lands beside the sequence search's.
        hits_files = list(output_dir.rglob("pdb_hits.m8"))
        self.assertTrue(hits_files, "no pdb_hits.m8 was written")
        self.assertTrue(hits_files[0].read_text(encoding="utf-8").strip())

        metadata = sorted(output_dir.glob(f"{QUERY_NAME}_feature_metadata_*.json*"))
        self.assertEqual(len(metadata), 1, f"expected one metadata file, got {metadata}")
        opener = lzma.open if metadata[0].suffix == ".xz" else open
        with opener(metadata[0], "rt", encoding="utf-8") as handle:
            recorded = json.load(handle)
        other = recorded.get("other", {})
        self.assertEqual(str(other.get("use_foldseek_templates")), "True")
        self.assertIn("foldseek_database_id", other)


if __name__ == "__main__":
    absltest.main()
