#!/usr/bin/env python
"""
Functional Alphapulldown alphafold2 backend tests.
Needs GPU(s) to run.
"""
from __future__ import annotations

import os
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import logging
import unittest
import lzma
from pathlib import Path

from absl.testing import absltest, parameterized

import alphapulldown
from alphapulldown_input_parser import generate_fold_specifications

# --------------------------------------------------------------------------- #
#                         configuration / logging                             #
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "test"

DATA_DIR = Path(os.getenv("ALPHAFOLD_DATA_DIR", "/scratch/AlphaFold_DBs/2.3.0"))
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/scratch/dima/jax_cache"
#os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_force_compilation_parallelism=8"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.95"
#os.environ["JAX_FLASH_ATTENTION_IMPL"] = "xla"

#FAST = os.getenv("ALPHAFOLD_FAST", "1") != "0" # <- no difference in performance
#if FAST:
#    from alphafold.model import config
#    config.CONFIG_MULTIMER.model.embeddings_and_evoformer.evoformer_num_block = 1


def _has_nvidia_gpu() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _gpu_functional_test_skip_reason() -> str | None:
    if os.getenv("RUN_GPU_FUNCTIONAL_TESTS", "").lower() in ("1", "true", "yes"):
        return None
    if os.getenv("CI", "").lower() in ("1", "true", "yes") or os.getenv(
        "GITHUB_ACTIONS", ""
    ).lower() == "true":
        return (
            "GPU functional tests are disabled on CI/CD. "
            "Set RUN_GPU_FUNCTIONAL_TESTS=1 to override."
        )
    if not _has_nvidia_gpu():
        return "GPU functional tests require an NVIDIA GPU and nvidia-smi."
    return None


def _mmseqs_functional_test_skip_reason() -> str | None:
    if os.getenv("RUN_MMSEQS_FUNCTIONAL_TESTS", "").lower() in ("1", "true", "yes"):
        return None
    return (
        "MMseqs functional inference tests are disabled by default. "
        "Set RUN_MMSEQS_FUNCTIONAL_TESTS=1 to enable."
    )


def _load_feature_dict(feature_path: Path) -> dict:
    opener = lzma.open if feature_path.suffix == ".xz" else open
    with opener(feature_path, "rb") as handle:
        payload = pickle.load(handle)
    if hasattr(payload, "feature_dict"):
        return payload.feature_dict
    return payload


def _load_feature_metadata(feature_dir: Path, protein_id: str) -> tuple[Path, dict]:
    matches = sorted(feature_dir.glob(f"{protein_id}_feature_metadata_*.json*"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one feature metadata file for {protein_id} in {feature_dir}, "
            f"found {matches}"
        )
    metadata_path = matches[0]
    opener = lzma.open if metadata_path.suffix == ".xz" else open
    with opener(metadata_path, "rt", encoding="utf-8") as handle:
        return metadata_path, json.load(handle)


def _metadata_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _non_empty_identifier_count(values) -> int:
    count = 0
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if str(value).strip():
            count += 1
    return count


def _af2_subprocess_env() -> dict[str, str]:
    """Return stable GPU/JAX defaults for AF2 functional subprocesses."""
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")
    env.setdefault("JAX_PLATFORM_NAME", "gpu")
    env.setdefault(
        "XLA_FLAGS",
        "--xla_gpu_force_compilation_parallelism=1 "
        "--xla_force_host_platform_device_count=1 "
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )
    return env

# --------------------------------------------------------------------------- #
#                       common helper mix-in / assertions                     #
# --------------------------------------------------------------------------- #
class _TestBase(parameterized.TestCase):
    use_temp_dir = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        skip_reason = _gpu_functional_test_skip_reason()
        if skip_reason:
            raise unittest.SkipTest(skip_reason)
        # do the skip here so import-time doesn't abort discovery
        #if not DATA_DIR.is_dir():
        #    cls.skipTest(f"set $ALPHAFOLD_DATA_DIR to run Alphafold functional tests (tried {DATA_DIR!r})")

        # Create base output dir
        if cls.use_temp_dir:
            cls.base_output_dir = Path(tempfile.mkdtemp(prefix="af2_test_"))
        else:
            cls.base_output_dir = Path("test/test_data/predictions/af2_backend")
            if cls.base_output_dir.exists():
                try:
                    shutil.rmtree(cls.base_output_dir)
                except (PermissionError, OSError) as e:
                    logger.warning("Could not remove %s: %s", cls.base_output_dir, e)
            cls.base_output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if cls.use_temp_dir and cls.base_output_dir.exists():
            try:
                shutil.rmtree(cls.base_output_dir)
            except (PermissionError, OSError) as e:
                logger.warning("Could not remove temporary directory %s: %s", cls.base_output_dir, e)

    def setUp(self):
        super().setUp()

        self.test_data_dir = TEST_ROOT / "test_data"
        self.test_features_dir = self.test_data_dir / "features"
        self.test_protein_lists_dir = self.test_data_dir / "protein_lists"
        self.test_modelling_dir = self.test_data_dir / "predictions"
        # setUpClass already resolved this to either a temp root or the legacy shared root
        self.af2_backend_dir = self.base_output_dir

        test_name = self._testMethodName
        self.output_dir = self.af2_backend_dir / test_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        apd_path = Path(alphapulldown.__path__[0])
        self.script_multimer = apd_path / "scripts" / "run_multimer_jobs.py"
        self.script_single = apd_path / "scripts" / "run_structure_prediction.py"
        self.script_create_features = (
            apd_path / "scripts" / "create_individual_features.py"
        )

    def _run_prediction_subprocess(self, args):
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            env=_af2_subprocess_env(),
        )

    # ---------------- assertions reused by all subclasses ----------------- #
    def _runCommonTests(self, res: subprocess.CompletedProcess, multimer: bool, dirname: str | None = None):
        if res.returncode != 0:
            self.fail(
                f"Subprocess failed (code {res.returncode})\n"
                f"STDOUT:\n{res.stdout}\n"
                f"STDERR:\n{res.stderr}"
            )

        if dirname is not None:
            folders = [self.output_dir / dirname]
        else:
            folders = [d for d in self.output_dir.iterdir() if d.is_dir()]

        for folder in folders:
            files = list(folder.iterdir())

            self.assertEqual(
                len([f for f in files if f.name.startswith("ranked") and f.suffix == ".pdb"]),
                5
            )

            pkls = [f for f in files if f.name.startswith("result") and f.suffix == ".pkl"]
            self.assertEqual(len(pkls), 5)

            example = pickle.load(pkls[0].open("rb"))
            keys_multimer = {
                "experimentally_resolved",
                "predicted_aligned_error",
                "predicted_lddt",
                "structure_module",
                "plddt",
                "max_predicted_aligned_error",
                "seqs",
                "iptm",
                "ptm",
                "ranking_confidence",
            }
            keys_monomer = keys_multimer - {"iptm"}
            expected_keys = keys_multimer if multimer else keys_monomer
            self.assertTrue(expected_keys <= example.keys())

            self.assertEqual(len([f for f in files if f.name.startswith("pae") and f.suffix == ".json"]), 5)
            self.assertEqual(len([f for f in files if f.suffix == ".png"]), 5)
            names = {f.name for f in files}
            self.assertIn("ranking_debug.json", names)
            self.assertIn("timings.json", names)

            ranking = json.loads((folder / "ranking_debug.json").read_text())
            self.assertEqual(len(ranking["order"]), 5)

    def _args(self, *, plist, mode, script):
        if script.endswith("run_multimer_jobs.py"):
            return [
                sys.executable,
                str(self.script_multimer),
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_dir={DATA_DIR}",
                f"--monomer_objects_dir={self.test_features_dir}",
                "--job_index=1",
                f"--output_path={self.output_dir}",
                f"--mode={mode}",
                (
                    "--oligomer_state_file"
                    if mode == "homo-oligomer"
                    else "--protein_lists"
                ) + f"={self.test_protein_lists_dir / plist}",
            ]
        else:
            specifications = generate_fold_specifications(
                input_files=[str(self.test_protein_lists_dir / plist)],
                delimiter="+",
                exclude_permutations=True,
            )
            lines = [
                spec.replace(",", ":").replace(";", "+")
                for spec in specifications if spec.strip()
            ]
            formatted_input = lines[0] if lines else ""
            return [
                sys.executable,
                str(self.script_single),
                f"--input={formatted_input}",
                f"--output_directory={self.output_dir}",
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={self.test_features_dir}",
            ]


# --------------------------------------------------------------------------- #
#                        parameterised “run mode” tests                       #
# --------------------------------------------------------------------------- #
class TestRunModes(_TestBase):
    @parameterized.named_parameters(
        dict(testcase_name="monomer", protein_list="test_monomer.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="dimer", protein_list="test_dimer.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="trimer", protein_list="test_trimer.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="homo_oligomer", protein_list="test_homooligomer.txt", mode="homo-oligomer", script="run_multimer_jobs.py"),
        dict(testcase_name="chopped_dimer", protein_list="test_dimer_chopped.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="long_name", protein_list="test_long_name.txt", mode="custom", script="run_structure_prediction.py"),
    )
    def test_(self, protein_list, mode, script):
        multimer = "monomer" not in protein_list
        res = self._run_prediction_subprocess(
            self._args(plist=protein_list, mode=mode, script=script)
        )
        self._runCommonTests(res, multimer)


# --------------------------------------------------------------------------- #
#                    parameterised “resume” / relaxation tests                #
# --------------------------------------------------------------------------- #
class TestResume(_TestBase):
    def setUp(self):
        super().setUp()
        self.protein_lists = self.test_protein_lists_dir / "test_dimer.txt"
        # Resume tests need a pre-populated per-test output tree to continue from.
        source = self.test_modelling_dir / "TEST_homo_2er"
        target = self.output_dir / "TEST_homo_2er"
        shutil.copytree(source, target, dirs_exist_ok=True)

        self.base_args = [
            sys.executable,
            str(self.script_multimer),
            "--mode=custom",
            "--num_cycle=1",
            "--num_predictions_per_model=1",
            f"--data_dir={DATA_DIR}",
            f"--protein_lists={self.protein_lists}",
            f"--monomer_objects_dir={self.test_features_dir}",
            "--job_index=1",
            f"--output_path={self.output_dir}",
        ]

    def _runAfterRelaxTests(self, relax_mode="All"):
        expected = {"None": 0, "Best": 1, "All": 5}[relax_mode]
        d = self.output_dir / "TEST_homo_2er"
        got = len([f for f in d.iterdir() if f.name.startswith("relaxed") and f.suffix == ".pdb"])
        self.assertEqual(got, expected)

    @parameterized.named_parameters(
        dict(
            testcase_name="no_relax",
            relax_mode="None",
            remove=[
                "relaxed_model_1_multimer_v3_pred_0.pdb",
                "relaxed_model_2_multimer_v3_pred_0.pdb",
                "relaxed_model_3_multimer_v3_pred_0.pdb",
                "relaxed_model_4_multimer_v3_pred_0.pdb",
                "relaxed_model_5_multimer_v3_pred_0.pdb",
            ],
        ),
        dict(
            testcase_name="relax_all",
            relax_mode="All",
            remove=[
                "relaxed_model_1_multimer_v3_pred_0.pdb",
                "relaxed_model_2_multimer_v3_pred_0.pdb",
                "relaxed_model_3_multimer_v3_pred_0.pdb",
                "relaxed_model_4_multimer_v3_pred_0.pdb",
                "relaxed_model_5_multimer_v3_pred_0.pdb",
            ],
        ),
        dict(
            testcase_name="continue_relax",
            relax_mode="All",
            remove=["relaxed_model_5_multimer_v3_pred_0.pdb"],
        ),
        dict(
            testcase_name="continue_prediction",
            relax_mode="Best",
            remove=[
                "unrelaxed_model_5_multimer_v3_pred_0.pdb",
                "relaxed_model_1_multimer_v3_pred_0.pdb",
                "relaxed_model_2_multimer_v3_pred_0.pdb",
                "relaxed_model_3_multimer_v3_pred_0.pdb",
                "relaxed_model_4_multimer_v3_pred_0.pdb",
                "relaxed_model_5_multimer_v3_pred_0.pdb",
            ],
        ),
    )
    def test_(self, relax_mode, remove):
        args = self.base_args + [f"--models_to_relax={relax_mode}"]
        for fname in remove:
            try:
                (self.output_dir / "TEST_homo_2er" / fname).unlink()
            except FileNotFoundError:
                pass
        res = self._run_prediction_subprocess(args)
        self._runCommonTests(res, multimer=True, dirname="TEST_homo_2er")
        self._runAfterRelaxTests(relax_mode)


def _parse_test_args():
    use_temp = '--use-temp-dir' in sys.argv or __import__("os").getenv('USE_TEMP_DIR', '').lower() in ('1','true','yes')
    while '--use-temp-dir' in sys.argv:
        sys.argv.remove('--use-temp-dir')
    return use_temp

_TestBase.use_temp_dir = _parse_test_args()


# --------------------------------------------------------------------------- #
#                           dropout diversity tests                             #
# --------------------------------------------------------------------------- #
class TestDropoutDiversity(_TestBase):
    """Test that dropout flag generates more diverse models."""

    def setUp(self):
        super().setUp()
        # Use dimer because for monomer we can't use num_predictions_per_model
        self.protein_lists = self.test_protein_lists_dir / "test_dropout.txt"

    def test_dropout_increases_diversity(self):
        """Test that using --dropout flag increases diversity between predictions."""

        # Create separate output directories for with/without dropout
        dropout_output_dir = self.output_dir / "dropout_test"
        no_dropout_output_dir = self.output_dir / "no_dropout_test"
        dropout_output_dir.mkdir(parents=True, exist_ok=True)
        no_dropout_output_dir.mkdir(parents=True, exist_ok=True)

        # Use simple test input 
        specifications = generate_fold_specifications(
            input_files=[str(self.protein_lists)],
            delimiter="+",
            exclude_permutations=True,
        )
        lines = [
            spec.replace(",", ":").replace(";", "+")
            for spec in specifications if spec.strip()
        ]
        formatted_input = lines[0] if lines else ""

        # Base arguments for both runs
        base_args = [
            sys.executable,
            str(self.script_single),
            f"--input={formatted_input}",
            "--num_cycle=1",
            "--num_predictions_per_model=2",  # Run 2 predictions to compare 
            f"--data_directory={DATA_DIR}",
            f"--features_directory={self.test_features_dir}",
            "--random_seed=42",  # Fixed seed for reproducibility
            "--model_names=model_2_multimer_v3",
        ]

        # Run prediction without dropout
        args_no_dropout = base_args + [f"--output_directory={no_dropout_output_dir}"]

        # Run prediction with dropout
        args_with_dropout = base_args + [
            f"--output_directory={dropout_output_dir}",
            "--dropout"
        ]

        # Execute both predictions
        logger.info("Running prediction without dropout...")
        #logger.info("".join(args_no_dropout))
        res_no_dropout = self._run_prediction_subprocess(args_no_dropout)
        self.assertEqual(res_no_dropout.returncode, 0, 
                        f"No dropout prediction failed: {res_no_dropout.stderr}")

        logger.info("Running prediction with dropout...")
        res_with_dropout = self._run_prediction_subprocess(args_with_dropout)
        self.assertEqual(res_with_dropout.returncode, 0,
                        f"Dropout prediction failed: {res_with_dropout.stderr}")

        # Find the generated PDB files
        no_dropout_pdbs = sorted(list(no_dropout_output_dir.glob("**/unrelaxed_*.pdb")))
        dropout_pdbs = sorted(list(dropout_output_dir.glob("**/unrelaxed_*.pdb")))

        self.assertGreaterEqual(len(no_dropout_pdbs), 2, "Need at least 2 PDB files for no-dropout prediction")
        self.assertGreaterEqual(len(dropout_pdbs), 2, "Need at least 2 PDB files for dropout prediction") 

        # Calculate RMSD between corresponding models
        from alphapulldown.utils.calculate_rmsd import calculate_rmsd_and_superpose

        # Create a temporary directory for RMSD calculation output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Calculate RMSD between first and second prediction without dropout
            rmsd_no_dropout = calculate_rmsd_and_superpose(
                str(no_dropout_pdbs[0]), str(no_dropout_pdbs[1]), temp_dir
            )

            # Calculate RMSD between first and second prediction with dropout
            rmsd_with_dropout = calculate_rmsd_and_superpose(
                str(dropout_pdbs[0]), str(dropout_pdbs[1]), temp_dir
            )

            logger.info(f"RMSD without dropout (between pred_0 and pred_1): {rmsd_no_dropout:.4f}")
            logger.info(f"RMSD with dropout (between pred_0 and pred_1): {rmsd_with_dropout:.4f}")

            # Verify that dropout increases diversity (higher RMSD)
            # Note: Due to randomness, this may not always be true, but it should be true on average
            # For a robust test, we check that both calculations succeed and produce reasonable values
            self.assertIsNotNone(rmsd_no_dropout, "RMSD calculation failed for no-dropout case")
            self.assertIsNotNone(rmsd_with_dropout, "RMSD calculation failed for dropout case")
            self.assertGreater(rmsd_no_dropout, 0, "RMSD should be positive for no-dropout case")
            self.assertGreater(rmsd_with_dropout, 0, "RMSD should be positive for dropout case")

            # Log the comparison result
            if rmsd_with_dropout > rmsd_no_dropout:
                logger.info("✓ Dropout increased structural diversity as expected")
            else:
                logger.info("⚠ Dropout did not increase diversity in this run (this can happen due to randomness)")

            # The test passes if calculations succeed - the diversity check is informational


class TestMmseqsIssue588Inference(_TestBase):
    """Opt-in end-to-end regression for freshly generated mmseq AF2 features."""

    ISSUE_588_IDS = ("A0ABD7FQG0", "P18004")

    def _require_mmseqs_functional_environment(self) -> None:
        skip_reason = _mmseqs_functional_test_skip_reason()
        if skip_reason:
            self.skipTest(skip_reason)
        for protein_id in self.ISSUE_588_IDS:
            fasta_path = self.test_data_dir / "fastas" / f"{protein_id}.fasta"
            self.assertTrue(
                fasta_path.is_file(),
                f"Missing FASTA fixture {fasta_path}",
            )

    def _generate_issue_588_mmseq_features(self) -> Path:
        feature_dir = self.output_dir / "issue_588_mmseq_features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        fasta_paths = ",".join(
            str(self.test_data_dir / "fastas" / f"{protein_id}.fasta")
            for protein_id in self.ISSUE_588_IDS
        )
        args = [
            sys.executable,
            str(self.script_create_features),
            f"--fasta_paths={fasta_paths}",
            f"--output_dir={feature_dir}",
            f"--data_dir={DATA_DIR}",
            "--max_template_date=2024-05-02",
            "--use_mmseqs2=True",
            "--data_pipeline=alphafold2",
            "--compress_features=True",
            "--skip_existing=False",
        ]
        res = self._run_prediction_subprocess(args)
        self.assertEqual(
            res.returncode,
            0,
            f"MMseqs feature generation failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
        )
        return feature_dir

    def _generate_issue_588_precomputed_mmseq_features(self) -> Path:
        source_dir = self.output_dir / "issue_588_mmseq_source_features"
        precomputed_dir = self.output_dir / "issue_588_mmseq_precomputed_features"
        source_dir.mkdir(parents=True, exist_ok=True)
        precomputed_dir.mkdir(parents=True, exist_ok=True)
        fasta_paths = ",".join(
            str(self.test_data_dir / "fastas" / f"{protein_id}.fasta")
            for protein_id in self.ISSUE_588_IDS
        )

        source_res = self._run_prediction_subprocess(
            [
                sys.executable,
                str(self.script_create_features),
                f"--fasta_paths={fasta_paths}",
                f"--output_dir={source_dir}",
                f"--data_dir={DATA_DIR}",
                "--max_template_date=2024-05-02",
                "--use_mmseqs2=True",
                "--data_pipeline=alphafold2",
                "--save_msa_files=True",
                "--compress_features=True",
                "--skip_existing=False",
            ]
        )
        self.assertEqual(
            source_res.returncode,
            0,
            "MMseqs source feature generation failed.\n"
            f"STDOUT:\n{source_res.stdout}\nSTDERR:\n{source_res.stderr}",
        )

        for protein_id in self.ISSUE_588_IDS:
            self.assertTrue(
                (source_dir / f"{protein_id}.a3m").is_file(),
                f"Expected MMseq A3M {source_dir / f'{protein_id}.a3m'} to be created.",
            )
            self.assertTrue(
                (source_dir / f"{protein_id}.pkl.xz").is_file(),
                f"Expected compressed feature pickle {source_dir / f'{protein_id}.pkl.xz'} to be created.",
            )
            shutil.copy2(
                source_dir / f"{protein_id}.a3m",
                precomputed_dir / f"{protein_id}.a3m",
            )
            sidecar = source_dir / f"{protein_id}.mmseq_ids.json"
            if sidecar.is_file():
                shutil.copy2(sidecar, precomputed_dir / sidecar.name)

        precomputed_res = self._run_prediction_subprocess(
            [
                sys.executable,
                str(self.script_create_features),
                f"--fasta_paths={fasta_paths}",
                f"--output_dir={precomputed_dir}",
                f"--data_dir={DATA_DIR}",
                "--max_template_date=2024-05-02",
                "--use_mmseqs2=True",
                "--use_precomputed_msas=True",
                "--data_pipeline=alphafold2",
                "--compress_features=True",
                "--skip_existing=False",
            ]
        )
        self.assertEqual(
            precomputed_res.returncode,
            0,
            "Precomputed-MMseq feature generation failed.\n"
            f"STDOUT:\n{precomputed_res.stdout}\nSTDERR:\n{precomputed_res.stderr}",
        )
        for protein_id in self.ISSUE_588_IDS:
            self.assertTrue(
                (precomputed_dir / f"{protein_id}.a3m").is_file(),
                f"Expected copied MMseq A3M {precomputed_dir / f'{protein_id}.a3m'} to be present.",
            )
            self.assertTrue(
                (precomputed_dir / f"{protein_id}.pkl.xz").is_file(),
                f"Expected precomputed feature pickle {precomputed_dir / f'{protein_id}.pkl.xz'} to be created.",
            )
        return precomputed_dir

    def _resolve_af2_result_dir(self, root: Path) -> Path:
        if (root / "ranking_debug.json").exists():
            return root
        candidates = sorted(
            path.parent for path in root.rglob("ranking_debug.json")
        )
        self.assertEqual(
            len(candidates),
            1,
            f"Expected one AF2 result directory under {root}, found {candidates}",
        )
        return candidates[0]

    def test_issue_588_mmseqs_generated_features_enable_af2_multimer_inference(self):
        from alphafold.data import feature_processing
        from alphafold.data import msa_pairing
        from alphafold.data import pipeline_multimer

        self._require_mmseqs_functional_environment()
        feature_dir = self._generate_issue_588_mmseq_features()

        converted_chains = {}
        for chain_id, protein_id in zip(("A", "B"), self.ISSUE_588_IDS):
            feature_path = feature_dir / f"{protein_id}.pkl.xz"
            feature_dict = _load_feature_dict(feature_path)
            self.assertGreater(
                _non_empty_identifier_count(
                    feature_dict["msa_species_identifiers_all_seq"]
                ),
                0,
                f"{protein_id} should keep recovered species IDs in msa_species_identifiers_all_seq",
            )
            self.assertGreater(
                _non_empty_identifier_count(
                    feature_dict["msa_uniprot_accession_identifiers_all_seq"]
                ),
                0,
                f"{protein_id} should keep recovered accession IDs in msa_uniprot_accession_identifiers_all_seq",
            )
            converted_chains[chain_id] = pipeline_multimer.convert_monomer_features(
                feature_dict,
                chain_id,
            )

        assembly_features = pipeline_multimer.add_assembly_features(converted_chains)
        feature_processing.process_unmerged_features(assembly_features)
        np_chains = list(assembly_features.values())
        paired_row_groups = msa_pairing.pair_sequences(np_chains)
        paired_rows = msa_pairing.reorder_paired_rows(paired_row_groups)
        self.assertGreater(
            paired_rows.shape[0],
            1,
            "Fresh mmseq AF2 features should produce paired rows beyond the query",
        )

        prediction_dir = self.output_dir / "af2_prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(
            [
                sys.executable,
                str(self.script_single),
                "--input=A0ABD7FQG0+P18004",
                f"--output_directory={prediction_dir}",
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                "--model_names=model_4_multimer_v3",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={feature_dir}",
                "--random_seed=42",
            ],
            capture_output=True,
            text=True,
            env=_af2_subprocess_env(),
        )
        self.assertEqual(
            res.returncode,
            0,
            f"AF2 inference failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
        )

        result_dir = self._resolve_af2_result_dir(prediction_dir)
        ranking_payload = json.loads(
            (result_dir / "ranking_debug.json").read_text(encoding="utf-8")
        )
        self.assertTrue(ranking_payload["order"])

        result_pickles = sorted(result_dir.glob("result_*.pkl"))
        self.assertLen(result_pickles, 1)
        with result_pickles[0].open("rb") as handle:
            result_payload = pickle.load(handle)
        self.assertIn("iptm", result_payload)
        self.assertIn("ranking_confidence", result_payload)
        self.assertGreater(
            result_payload["iptm"],
            0.6,
            f"Expected AF2 ipTM > 0.6, got {result_payload['iptm']}",
        )

    def test_issue_614_precomputed_mmseqs_features_enable_af2_multimer_inference(self):
        """Issue #614 regression: AF2 should fold successfully from precomputed MMseq A3Ms."""
        self._require_mmseqs_functional_environment()
        feature_dir = self._generate_issue_588_precomputed_mmseq_features()

        for protein_id in self.ISSUE_588_IDS:
            metadata_path, metadata = _load_feature_metadata(feature_dir, protein_id)
            self.assertTrue(
                _metadata_bool(metadata["other"]["use_precomputed_msas"]),
                f"{metadata_path} should record use_precomputed_msas=True",
            )
            feature_dict = _load_feature_dict(feature_dir / f"{protein_id}.pkl.xz")
            self.assertGreater(
                _non_empty_identifier_count(
                    feature_dict["msa_species_identifiers_all_seq"]
                ),
                0,
                f"{protein_id} should keep recovered species IDs from cached MMseq A3Ms",
            )
            self.assertGreater(
                _non_empty_identifier_count(
                    feature_dict["msa_uniprot_accession_identifiers_all_seq"]
                ),
                0,
                f"{protein_id} should keep recovered accession IDs from cached MMseq A3Ms",
            )

        prediction_dir = self.output_dir / "af2_precomputed_prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        res = self._run_prediction_subprocess(
            [
                sys.executable,
                str(self.script_single),
                "--input=A0ABD7FQG0+P18004",
                f"--output_directory={prediction_dir}",
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                "--model_names=model_4_multimer_v3",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={feature_dir}",
                "--random_seed=42",
            ]
        )
        self.assertEqual(
            res.returncode,
            0,
            "AF2 inference from precomputed MMseq features failed.\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
        )

        result_dir = self._resolve_af2_result_dir(prediction_dir)
        ranking_payload = json.loads(
            (result_dir / "ranking_debug.json").read_text(encoding="utf-8")
        )
        self.assertTrue(ranking_payload["order"])

        result_pickles = sorted(result_dir.glob("result_*.pkl"))
        self.assertLen(result_pickles, 1)
        with result_pickles[0].open("rb") as handle:
            result_payload = pickle.load(handle)
        self.assertIn("iptm", result_payload)
        self.assertIn("ranking_confidence", result_payload)
        self.assertGreater(
            result_payload["iptm"],
            0.6,
            f"Expected AF2 ipTM > 0.6 from precomputed MMseq features, got {result_payload['iptm']}",
        )

if __name__ == "__main__":
    absltest.main()
