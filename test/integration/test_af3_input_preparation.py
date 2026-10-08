"""AF3 input preparation from real test data, without model weights or a GPU.

These build AF3 fold inputs from AlphaPulldown features and AF3 JSON, and check
what AF3 would receive: chains, residue numbering, MSAs, templates, job names
and viewer output. They need AF3's compiled extension and CCD, so they run in the
AF3 image's test stage and skip elsewhere.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from absl.testing import absltest, parameterized

try:
    importlib.import_module("alphafold3.cpp")
except ImportError as exc:
    pytest.skip(
        f"AlphaFold 3 test dependencies are unavailable: {exc}",
        allow_module_level=True,
    )

from alphapulldown.objects import MultimericObject
from alphapulldown.utils.modelling_setup import (
    create_custom_info,
    create_interactors,
    parse_fold,
)

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))
from af3_fixtures import (  # noqa: E402
    _a3m_query_sequence,
    _aligned_a3m_row_length,
    _protein_entries_from_af3_input,
    _load_feature_metadata,
    _metadata_bool,
    _non_empty_a3m_payload_rows,
    _load_feature_dict,
    _load_feature_payload,
    _write_feature_payload,
    _non_empty_identifier_count,
)
from af3_fixtures import Af3TestData  # noqa: E402


class TestAlphaFold3InputPreparation(Af3TestData, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.test_data_dir = TEST_ROOT / "test_data"
        self.test_fastas_dir = self.test_data_dir / "fastas"
        self.test_features_dir = self.test_data_dir / "features"
        temp_dir = tempfile.TemporaryDirectory(prefix="af3_input_")
        self.addCleanup(temp_dir.cleanup)
        self.output_dir = Path(temp_dir.name)

    def _copy_real_feature_fixture(
        self,
        *,
        source_dir: Path,
        protein_id: str,
        target_dir: Path,
    ) -> Path:
        copied_feature_path = None
        for pattern in (
            f"{protein_id}.pkl",
            f"{protein_id}.pkl.xz",
            f"{protein_id}.a3m",
            f"{protein_id}_feature_metadata_*.json*",
        ):
            for source_path in sorted(source_dir.glob(pattern)):
                target_path = target_dir / source_path.name
                shutil.copy2(source_path, target_path)
                if source_path.name.startswith(f"{protein_id}.pkl"):
                    copied_feature_path = target_path

        self.assertIsNotNone(
            copied_feature_path,
            f"Missing real feature fixture for {protein_id} in {source_dir}",
        )
        return copied_feature_path

    @staticmethod
    def _synthetic_accession_ids(species_ids: np.ndarray) -> np.ndarray:
        identifiers = []
        for index, value in enumerate(species_ids):
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            identifiers.append(
                f"ACC{index:05d}".encode("utf-8") if str(value).strip() else b""
            )
        return np.asarray(identifiers, dtype=object)

    def _prepare_mixed_identifier_fixture_dir(self) -> Path:
        """Materialize real AF2 fixtures with mixed identifier enrichment.

        The underlying MSA rows come from repo fixtures in `test/test_data`.
        We only adjust the identifier sidecars so one chain looks enriched while
        the other reproduces the "no species enrichment / no accession IDs"
        failure mode from issue #614's AF3 follow-up comment.
        """
        feature_dir = self.output_dir / "mixed_identifier_features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        source_dir = self.test_features_dir / "af2_features" / "protein"

        enriched_feature_path = self._copy_real_feature_fixture(
            source_dir=source_dir,
            protein_id="A0A024R1R8",
            target_dir=feature_dir,
        )
        unenriched_feature_path = self._copy_real_feature_fixture(
            source_dir=source_dir,
            protein_id="P61626",
            target_dir=feature_dir,
        )

        enriched_payload = _load_feature_payload(enriched_feature_path)
        enriched_feature_dict = (
            enriched_payload.feature_dict
            if hasattr(enriched_payload, "feature_dict")
            else enriched_payload
        )
        enriched_feature_dict["msa_uniprot_accession_identifiers"] = (
            self._synthetic_accession_ids(
                np.asarray(enriched_feature_dict["msa_species_identifiers"])
            )
        )
        enriched_feature_dict["msa_uniprot_accession_identifiers_all_seq"] = (
            self._synthetic_accession_ids(
                np.asarray(enriched_feature_dict["msa_species_identifiers_all_seq"])
            )
        )
        _write_feature_payload(enriched_feature_path, enriched_payload)

        unenriched_payload = _load_feature_payload(unenriched_feature_path)
        unenriched_feature_dict = (
            unenriched_payload.feature_dict
            if hasattr(unenriched_payload, "feature_dict")
            else unenriched_payload
        )
        unenriched_feature_dict["msa_species_identifiers"] = np.asarray(
            [b""] * int(np.asarray(unenriched_feature_dict["msa"]).shape[0]),
            dtype=object,
        )
        unenriched_feature_dict["msa_species_identifiers_all_seq"] = np.asarray(
            [b""] * int(np.asarray(unenriched_feature_dict["msa_all_seq"]).shape[0]),
            dtype=object,
        )
        unenriched_feature_dict.pop("msa_uniprot_accession_identifiers", None)
        unenriched_feature_dict.pop(
            "msa_uniprot_accession_identifiers_all_seq", None
        )
        _write_feature_payload(unenriched_feature_path, unenriched_payload)

        return feature_dir

    def test_issue_588_mmseqs_af2_features_produce_sane_af3_chain_input_msas(self):
        """Issue #588 regression: verify AF3 input construction from exact AF2/mmseqs2 pkl fixtures."""
        from alphapulldown.folding_backend.alphafold3_backend import process_fold_input

        issue_588_dir = self.test_features_dir / "issue_588"
        for protein_id in ("A0ABD7FQG0", "P18004"):
            metadata_path, metadata = _load_feature_metadata(issue_588_dir, protein_id)
            other = metadata["other"]
            self.assertTrue(
                _metadata_bool(other["use_mmseqs2"]),
                f"{metadata_path} is not a mmseqs2-generated AF2 fixture.",
            )
            self.assertEqual(other["data_pipeline"], "alphafold2")
            self.assertFalse(_metadata_bool(other["re_search_templates_mmseqs2"]))

        fold_input_obj = self._prepare_fold_input(
            fold_spec="A0ABD7FQG0+P18004",
            feature_dir=issue_588_dir,
            debug_msas=True,
        )

        protein_chains = [chain for chain in fold_input_obj.chains if hasattr(chain, "sequence")]
        chain_sequences = {chain.id: chain.sequence for chain in protein_chains}
        self.assertEqual(sorted(chain_sequences), ["A", "B"])

        job_name = fold_input_obj.sanitised_name()
        summary_path = self.output_dir / f"{job_name}_af2_to_af3_translation_summary.json"
        self.assertTrue(summary_path.is_file(), f"Missing translation summary {summary_path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertTrue(summary["paired_rows_valid"])
        self.assertTrue(summary["unpaired_rows_valid"])
        self.assertLen(summary["translation_modes"], 1)
        translation_mode = summary["translation_modes"][0]
        self.assertIn(
            translation_mode,
            {
                "af3_species_pairing_from_af2_individual_msas",
                "manual_unpaired_from_af2_multimer",
            },
        )
        self.assertLen(summary["chains"], 2)

        for chain_summary in summary["chains"]:
            chain_id = chain_summary["chain_id"]
            expected_sequence = chain_sequences[chain_id]
            if translation_mode == "af3_species_pairing_from_af2_individual_msas":
                self.assertGreater(
                    chain_summary["paired_msa_row_count"],
                    0,
                    f"Expected non-empty paired MSA rows for chain {chain_id}",
                )
                self.assertGreater(
                    chain_summary["unpaired_msa_row_count"],
                    0,
                    f"Expected non-empty unpaired MSA rows for chain {chain_id}",
                )
            else:
                self.assertEqual(chain_summary["paired_msa_row_count"], 0)
                self.assertGreater(
                    chain_summary["unpaired_msa_row_count"],
                    0,
                    f"Expected non-empty unpaired MSA rows for chain {chain_id}",
                )

            for msa_kind in ("paired_input", "unpaired_input"):
                msa_path = self.output_dir / f"{job_name}_chain-{chain_id}_{msa_kind}.a3m"
                self.assertTrue(msa_path.is_file(), f"Missing debug MSA {msa_path}")
                msa_text = msa_path.read_text(encoding="utf-8")
                if msa_text:
                    self.assertEqual(_a3m_query_sequence(msa_text), expected_sequence)
                payload_sequences = _non_empty_a3m_payload_rows(msa_text)
                if translation_mode == "af3_species_pairing_from_af2_individual_msas":
                    self.assertGreater(
                        len(payload_sequences),
                        0,
                        f"Expected payload rows in {msa_path}",
                    )
                elif msa_kind == "unpaired_input":
                    self.assertGreater(
                        len(payload_sequences),
                        0,
                        f"Expected payload rows in {msa_path}",
                    )
                else:
                    self.assertEqual(payload_sequences, [])
                for payload_sequence in payload_sequences:
                    self.assertEqual(
                        _aligned_a3m_row_length(payload_sequence),
                        len(expected_sequence),
                        f"Aligned row length mismatch in {msa_path}",
                    )

        process_fold_input(
            fold_input=fold_input_obj,
            model_runner=None,
            output_dir=str(self.output_dir),
            buckets=(512,),
        )
        input_json = self.output_dir / f"{job_name}_data.json"
        written = json.loads(input_json.read_text(encoding="utf-8"))
        protein_entries = {
            protein_entry["id"]: protein_entry
            for protein_entry in _protein_entries_from_af3_input(written)
        }
        self.assertEqual(set(protein_entries), set(chain_sequences))

        for chain_id, protein_entry in protein_entries.items():
            expected_sequence = chain_sequences[chain_id]
            self.assertEqual(protein_entry["sequence"], expected_sequence)
            if translation_mode == "af3_species_pairing_from_af2_individual_msas":
                self.assertEqual(
                    _a3m_query_sequence(protein_entry["pairedMsa"]),
                    expected_sequence,
                )
                self.assertEqual(
                    _a3m_query_sequence(protein_entry["unpairedMsa"]),
                    expected_sequence,
                )
            else:
                self.assertEqual(protein_entry["pairedMsa"], "")
                self.assertEqual(
                    _a3m_query_sequence(protein_entry["unpairedMsa"]),
                    expected_sequence,
                )
            # These exact issue-588 fixtures are AF2/mmseqs2-derived and were
            # generated without MMseqs template re-search. Empty templates
            # document fixture provenance here, not an AF3 conversion failure.
            self.assertEqual(protein_entry["templates"], [])

    def test_af3_prepare_input_preserves_templates_for_templated_af2_pkl_features(self):
        """Positive control: templated AF2 pkl inputs should keep templates in AF3 JSON."""
        from alphapulldown.folding_backend.alphafold3_backend import process_fold_input

        feature_dir = self.test_features_dir / "af2_features" / "protein"
        fold_input_obj = self._prepare_fold_input(
            fold_spec="P61626",
            feature_dir=feature_dir,
        )

        self.assertLen(fold_input_obj.chains, 1)
        self.assertGreater(len(fold_input_obj.chains[0].templates), 0)

        process_fold_input(
            fold_input=fold_input_obj,
            model_runner=None,
            output_dir=str(self.output_dir),
            buckets=(512,),
        )
        input_json = self.output_dir / f"{fold_input_obj.sanitised_name()}_data.json"
        written = json.loads(input_json.read_text(encoding="utf-8"))
        protein_entries = _protein_entries_from_af3_input(written)

        self.assertLen(protein_entries, 1)
        self.assertGreater(len(protein_entries[0]["templates"]), 0)
        self.assertTrue(
            all(template["mmcif"] for template in protein_entries[0]["templates"])
        )

    def test_af3_real_fixture_pipeline_tolerates_mixed_missing_accession_ids(self):
        """AF3 prep should tolerate a real mixed-enrichment multimer feature set."""
        from alphapulldown.folding_backend.alphafold3_backend import (
            AlphaFold3Backend,
            process_fold_input,
        )
        from alphapulldown.scripts import run_structure_prediction

        feature_dir = self._prepare_mixed_identifier_fixture_dir()

        enriched_feature_dict = _load_feature_dict(feature_dir / "A0A024R1R8.pkl")
        self.assertGreater(
            _non_empty_identifier_count(
                enriched_feature_dict["msa_uniprot_accession_identifiers_all_seq"]
            ),
            0,
        )
        unenriched_feature_dict = _load_feature_dict(feature_dir / "P61626.pkl")
        self.assertEqual(
            _non_empty_identifier_count(
                unenriched_feature_dict["msa_species_identifiers_all_seq"]
            ),
            0,
        )
        self.assertNotIn(
            "msa_uniprot_accession_identifiers_all_seq",
            unenriched_feature_dict,
        )

        script_flags = SimpleNamespace(
            pair_msa=True,
            multimeric_template=False,
            description_file=None,
            path_to_mmt=None,
            threshold_clashes=1000,
            hb_allowance=0.4,
            plddt_threshold=0,
            save_features_for_multimeric_object=False,
            features_directory=[str(feature_dir)],
            use_ap_style=False,
        )

        with mock.patch.object(run_structure_prediction, "FLAGS", script_flags):
            parsed = run_structure_prediction.parse_fold(
                ["A0A024R1R8+P61626"],
                [str(feature_dir)],
                "+",
            )
            data = run_structure_prediction.create_custom_info(parsed)
            all_interactors = run_structure_prediction.create_interactors(
                data,
                [str(feature_dir)],
            )
            self.assertLen(all_interactors, 1)
            self.assertLen(all_interactors[0], 2)
            object_to_model, prepared_output_dir = (
                run_structure_prediction.pre_modelling_setup(
                    all_interactors[0],
                    output_dir=str(self.output_dir / "mixed_identifier_prediction"),
                )
            )

        mappings = AlphaFold3Backend.prepare_input(
            objects_to_model=[
                {"object": object_to_model, "output_dir": prepared_output_dir}
            ],
            random_seed=42,
            debug_msas=True,
        )
        self.assertLen(mappings, 1)
        fold_input_obj, (
            prepared_output_dir,
            resolve_msa_overlaps,
        ) = next(iter(mappings[0].items()))

        process_fold_input(
            fold_input=fold_input_obj,
            model_runner=None,
            output_dir=prepared_output_dir,
            buckets=(512,),
            resolve_msa_overlaps=resolve_msa_overlaps,
        )

        job_name = fold_input_obj.sanitised_name()
        summary_path = (
            Path(prepared_output_dir)
            / f"{job_name}_af2_to_af3_translation_summary.json"
        )
        self.assertTrue(summary_path.is_file(), f"Missing translation summary {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertLen(summary["chains"], 2)
        self.assertTrue(summary["unpaired_rows_valid"])

        input_json = Path(prepared_output_dir) / f"{job_name}_data.json"
        self.assertTrue(input_json.is_file(), f"Missing AF3 input JSON {input_json}")
        written = json.loads(input_json.read_text(encoding="utf-8"))
        protein_entries = {
            protein_entry["id"]: protein_entry
            for protein_entry in _protein_entries_from_af3_input(written)
        }
        self.assertEqual(set(protein_entries), {"A", "B"})
        for chain in fold_input_obj.chains:
            if not hasattr(chain, "sequence"):
                continue
            protein_entry = protein_entries[chain.id]
            self.assertEqual(protein_entry["sequence"], chain.sequence)
            self.assertEqual(
                _a3m_query_sequence(protein_entry["unpairedMsa"]),
                chain.sequence,
            )

    def test_af3_on_the_fly_pairing_from_json_features(self):
        """
        Build a dimer from two AF3 JSON monomer feature files that only contain
        unpairedMsa. Ensure backend writes combined *_data.json where protein
        chains have pairedMsa populated (promoted from unpairedMsa) so AF3 can
        perform cross-chain pairing downstream. Skip model inference.
        """
        # Input JSONs (use repo-relative paths via test_features_dir)
        json_a = self.test_features_dir / "af3_features/protein/A0A024R1R8_af3_input.json"
        json_b = self.test_features_dir / "af3_features/protein/P61626_af3_input.json"

        # Prepare objects_to_model input to backend: two JSON inputs merged into one complex
        from alphapulldown.folding_backend.alphafold3_backend import AlphaFold3Backend, process_fold_input

        objects_to_model = [
            {"object": {"json_input": str(json_a)}, "output_dir": str(self.output_dir)},
            {"object": {"json_input": str(json_b)}, "output_dir": str(self.output_dir)},
        ]

        # Use backend to prepare the combined input
        mappings = AlphaFold3Backend.prepare_input(objects_to_model=objects_to_model, random_seed=42)
        self.assertEqual(len(mappings), 1)
        fold_input_obj, out_dir = next(iter(mappings[0].items()))

        # Ask the backend helper to write *_data.json without inference
        res = process_fold_input(
            fold_input=fold_input_obj,
            model_runner=None,
            output_dir=str(self.output_dir),
            buckets=(512,),
        )
        self.assertIsNotNone(res)

        out_path = self.output_dir / f"{fold_input_obj.sanitised_name()}_data.json"

        # Load JSON and verify that each protein chain now has pairedMsa populated
        # (promoted from unpairedMsa) and unpairedMsa cleared.
        with open(out_path, "rt") as f:
            data = json.load(f)

        # JSON structure depends on AF3 version; check sequences[*].protein fields
        sequences = data.get("sequences", [])
        self.assertGreaterEqual(len(sequences), 2, "Expected at least two chains in combined input")

        # For protein entries, ensure at least one of pairedMsa/unpairedMsa is present
        # and that our pipeline can promote unpaired -> paired (non-empty strings present in at least one field)
        num_proteins = 0
        num_with_promoted_paired = 0
        for seq_entry in sequences:
            if "protein" in seq_entry:
                num_proteins += 1
                protein = seq_entry["protein"]
                paired = protein.get("pairedMsa", "") or ""
                unpaired = protein.get("unpairedMsa", None)
                # After promotion we expect pairedMsa to be non-empty and unpairedMsa to be ""
                if isinstance(paired, str) and len(paired) > 0 and (unpaired == "" or unpaired is None):
                    num_with_promoted_paired += 1

        self.assertGreaterEqual(num_proteins, 2, "Expected two protein chains in the dimer test")
        self.assertEqual(num_with_promoted_paired, num_proteins, "All protein chains must have pairedMsa populated and unpairedMsa cleared")

        # Finally, assert that original monomer JSON has empty pairedMsa, to validate that
        # we started from unpaired-only features.
        with open(json_a, "rt") as f:
            a_data = json.load(f)
        with open(json_b, "rt") as f:
            b_data = json.load(f)
        def _paired_empty(d):
            for seq_entry in d.get("sequences", []):
                if "protein" in seq_entry:
                    if seq_entry["protein"].get("pairedMsa", None):
                        return False
            return True
        self.assertTrue(_paired_empty(a_data))
        self.assertTrue(_paired_empty(b_data))

        print("✓ Combined AF3 input JSON created; per-chain MSAs present for backend pairing")

    def test_af3_custom_residue_ids_round_trip_through_json_and_structure(self):
        """Custom AF3 residue IDs must survive JSON and structure conversion."""
        from alphafold3.common import folding_input
        from alphafold3.constants import chemical_components

        expected_residue_ids = [2, 3, 4, 5, 8, 9, 10]
        chain = folding_input.ProteinChain(
            id="A",
            sequence="SSHEKKK",
            ptms=[],
            residue_ids=expected_residue_ids,
            unpaired_msa="",
            paired_msa="",
            templates=[],
        )
        fold_input = folding_input.Input(
            name="gap_test",
            chains=[chain],
            rng_seeds=[1],
        )

        round_tripped = folding_input.Input.from_json(fold_input.to_json())
        self.assertEqual(
            list(round_tripped.protein_chains[0].residue_ids),
            expected_residue_ids,
        )

        struc = round_tripped.to_structure(ccd=chemical_components.Ccd())
        self.assertEqual(struc.present_residues.id.tolist(), expected_residue_ids)

    def test_af3_custom_residue_ids_propagate_to_token_features(self):
        """AF3 token features must retain custom gapped residue numbering."""
        from alphafold3.common import folding_input
        from alphafold3.constants import chemical_components
        from alphafold3.model import features as af3_features
        from alphafold3.model.atom_layout import atom_layout
        from alphafold3.model.network import featurization as af3_featurization

        expected_residue_ids = [1, 2, 3, 4, 8, 9, 10]
        chain = folding_input.ProteinChain(
            id="A",
            sequence="ACDEFGH",
            ptms=[],
            residue_ids=expected_residue_ids,
            unpaired_msa="",
            paired_msa="",
            templates=[],
        )
        fold_input = folding_input.Input(
            name="gap_token_test",
            chains=[chain],
            rng_seeds=[1],
        )
        ccd = chemical_components.Ccd()
        struc = fold_input.to_structure(ccd=ccd)
        flat_layout = atom_layout.atom_layout_from_structure(struc)
        all_tokens, _, _ = af3_features.tokenizer(
            flat_layout,
            ccd=ccd,
            max_atoms_per_token=24,
            flatten_non_standard_residues=False,
            logging_name="gap_token_test",
        )
        padding_shapes = af3_features.PaddingShapes(
            num_tokens=len(all_tokens.atom_name),
            msa_size=1,
            num_chains=1,
            num_templates=0,
            num_atoms=24 * len(all_tokens.atom_name),
        )
        token_features = af3_features.TokenFeatures.compute_features(
            all_tokens=all_tokens,
            padding_shapes=padding_shapes,
        )

        self.assertEqual(
            token_features.residue_index[:len(expected_residue_ids)].tolist(),
            expected_residue_ids,
        )
        self.assertEqual(
            sorted(set(token_features.asym_id[:len(expected_residue_ids)].tolist())),
            [1],
        )

        relative_encoding = np.asarray(
            af3_featurization.create_relative_encoding(
                token_features,
                max_relative_idx=4,
                max_relative_chain=2,
            )
        )
        inter_chain_bin = 2 * 4 + 1
        self.assertEqual(relative_encoding[3, 4, inter_chain_bin], 0)
        self.assertEqual(np.argmax(relative_encoding[3, 4, : 2 * 4 + 2]), 0)

    def test_af3_duplicate_residue_ids_survive_empty_structure_round_trip(self):
        """AF3 must preserve duplicate residue IDs when rebuilding empty structures."""
        from alphafold3.common import folding_input
        from alphafold3.constants import chemical_components
        from alphafold3.model.atom_layout import atom_layout

        expected_residue_ids = list(range(1, 11)) + list(range(2, 6)) + list(range(12, 16))
        chain = folding_input.ProteinChain(
            id="A",
            sequence="ACDEFGHIKLCDEFMNPQ",
            ptms=[],
            residue_ids=expected_residue_ids,
            unpaired_msa="",
            paired_msa="",
            templates=[],
        )
        fold_input = folding_input.Input(
            name="duplicate_residue_ids_test",
            chains=[chain],
            rng_seeds=[1],
        )
        ccd = chemical_components.Ccd()
        struc = fold_input.to_structure(ccd=ccd)
        flat_layout = atom_layout.atom_layout_from_structure(struc)
        all_physical_residues = atom_layout.residues_from_structure(struc)
        rebuilt = atom_layout.make_structure(
            flat_layout,
            atom_coords=np.zeros((flat_layout.atom_name.shape[0], 3), dtype=np.float32),
            name="duplicate_residue_ids_test",
            all_physical_residues=all_physical_residues,
        )

        self.assertEqual(rebuilt.present_residues.id.tolist(), expected_residue_ids)

    def test_af3_output_job_name_compacts_long_homomer_names(self):
        """AF3 job names should stay readable and below common filename limits."""
        from alphapulldown.folding_backend.alphafold3_backend import AlphaFold3Backend

        parsed = parse_fold(
            ["A0A075B6L2:10:1-3:4-5:6-7:7-8"],
            [str(self.test_features_dir)],
            "+",
        )
        data = create_custom_info(parsed)
        all_interactors = create_interactors(data, [str(self.test_features_dir)])
        self.assertLen(all_interactors, 1)
        self.assertLen(all_interactors[0], 10)

        object_to_model = MultimericObject(interactors=all_interactors[0], pair_msa=True)
        mappings = AlphaFold3Backend.prepare_input(
            objects_to_model=[{"object": object_to_model, "output_dir": str(self.output_dir)}],
            random_seed=42,
        )
        self.assertLen(mappings, 1)
        fold_input_obj, _ = next(iter(mappings[0].items()))

        self.assertEqual(
            fold_input_obj.sanitised_name(),
            "A0A075B6L2_1-3_4-5_6-7_7-8__x10",
        )
        self.assertLessEqual(len(fold_input_obj.sanitised_name()), 200)
        expected_sequence = "".join(
            self._get_region_sequences(
                "A0A075B6L2",
                [(1, 3), (4, 5), (6, 7), (8, 8)],
            )
        )
        self.assertTrue(
            all(chain.sequence == expected_sequence for chain in fold_input_obj.chains)
        )
        self.assertTrue(
            all(list(chain.residue_ids) == [1, 2, 3, 4, 5, 6, 7, 8] for chain in fold_input_obj.chains)
        )

    def test_af3_output_job_name_hashes_overlong_unique_compound_names(self):
        """AF3 job names should fall back to a deterministic hash suffix when needed."""
        from alphapulldown.folding_backend.alphafold3_backend import (
            _build_output_job_name,
        )

        fragments = [
            f"protein_{index:02d}_{'verylongsegment' * 4}"
            for index in range(12)
        ]
        objects_to_model = [
            {
                "object": {
                    "json_input": str(
                        Path("/tmp") / f"{fragment}_af3_input.json"
                    )
                },
                "output_dir": str(self.output_dir),
            }
            for fragment in fragments
        ]

        readable_name = "_and_".join(fragments)
        self.assertGreater(len(readable_name), 200)

        job_name = _build_output_job_name(objects_to_model)
        expected_digest = hashlib.sha1(
            readable_name.encode("utf-8")
        ).hexdigest()[:12]

        self.assertLessEqual(len(job_name), 200)
        self.assertTrue(job_name.endswith(f"__{expected_digest}"))
        self.assertRegex(job_name, r"__[0-9a-f]{12}$")
        self.assertEqual(job_name, _build_output_job_name(objects_to_model))

    def test_af3_prepare_input_accepts_monomer_plus_ligand_json(self):
        """AF3 mixed protein+ligand JSON inputs must survive prepare_input cloning."""
        from alphafold3.common import folding_input
        from alphapulldown.folding_backend.alphafold3_backend import (
            AlphaFold3Backend,
            process_fold_input,
        )

        parsed = parse_fold(
            ["A0A024R1R8+ligand.json"],
            [str(self.test_features_dir)],
            "+",
        )
        data = create_custom_info(parsed)
        all_interactors = create_interactors(data, [str(self.test_features_dir)])
        self.assertLen(all_interactors, 1)
        self.assertLen(all_interactors[0], 2)

        objects_to_model = [
            {"object": obj, "output_dir": str(self.output_dir)}
            for obj in all_interactors[0]
        ]
        mappings = AlphaFold3Backend.prepare_input(
            objects_to_model=objects_to_model,
            random_seed=42,
        )
        self.assertLen(mappings, 1)
        fold_input_obj, _ = next(iter(mappings[0].items()))

        self.assertEqual([chain.id for chain in fold_input_obj.chains], ["A", "L"])
        self.assertIsInstance(fold_input_obj.chains[0], folding_input.ProteinChain)
        self.assertIsInstance(fold_input_obj.chains[1], folding_input.Ligand)
        self.assertEqual(list(fold_input_obj.chains[1].ccd_ids), ["ATP"])

        process_fold_input(
            fold_input=fold_input_obj,
            model_runner=None,
            output_dir=str(self.output_dir),
            buckets=(512,),
        )
        input_json = self.output_dir / f"{fold_input_obj.sanitised_name()}_data.json"
        with open(input_json, "rt") as handle:
            written = json.load(handle)

        protein_entries = [
            sequence_entry["protein"]
            for sequence_entry in written.get("sequences", [])
            if "protein" in sequence_entry
        ]
        ligand_entries = [
            sequence_entry["ligand"]
            for sequence_entry in written.get("sequences", [])
            if "ligand" in sequence_entry
        ]
        self.assertLen(protein_entries, 1)
        self.assertLen(ligand_entries, 1)
        self.assertEqual(ligand_entries[0]["id"], "L")
        self.assertEqual(ligand_entries[0]["ccdCodes"], ["ATP"])

    def test_af3_prepare_input_skips_invalid_json_templates_for_ptm_input(self):
        """Malformed inline JSON templates should be dropped instead of crashing AF3."""
        from alphafold3.common import folding_input
        from alphapulldown.folding_backend.alphafold3_backend import (
            AlphaFold3Backend,
            process_fold_input,
        )

        json_input = self.test_features_dir / "protein_with_ptms.json"
        raw_payload = json.loads(json_input.read_text())
        expected_protein = raw_payload["sequences"][0]["protein"]

        mappings = AlphaFold3Backend.prepare_input(
            objects_to_model=[
                {
                    "object": {"json_input": str(json_input)},
                    "output_dir": str(self.output_dir),
                }
            ],
            random_seed=42,
        )
        self.assertLen(mappings, 1)
        fold_input_obj, _ = next(iter(mappings[0].items()))

        self.assertEqual([chain.id for chain in fold_input_obj.chains], ["P"])
        self.assertLen(fold_input_obj.chains, 1)
        self.assertIsInstance(fold_input_obj.chains[0], folding_input.ProteinChain)
        self.assertEqual(list(fold_input_obj.chains[0].ptms), [("HYS", 1), ("2MG", 15)])
        self.assertEqual(list(fold_input_obj.chains[0].templates), [])

        process_fold_input(
            fold_input=fold_input_obj,
            model_runner=None,
            output_dir=str(self.output_dir),
            buckets=(512,),
        )
        input_json = self.output_dir / f"{fold_input_obj.sanitised_name()}_data.json"
        with open(input_json, "rt") as handle:
            written = json.load(handle)

        protein_entries = [
            sequence_entry["protein"]
            for sequence_entry in written.get("sequences", [])
            if "protein" in sequence_entry
        ]
        self.assertLen(protein_entries, 1)
        self.assertEqual(protein_entries[0]["id"], "P")
        self.assertEqual(protein_entries[0]["sequence"], expected_protein["sequence"])
        self.assertEqual(
            protein_entries[0]["modifications"],
            expected_protein["modifications"],
        )
        self.assertEqual(protein_entries[0]["templates"], [])

    def test_af3_prepare_input_keeps_valid_json_templates(self):
        """Valid inline JSON templates should survive prepare_input and JSON write-out."""
        from alphafold3.common import folding_input
        from alphapulldown.folding_backend.alphafold3_backend import (
            AlphaFold3Backend,
            process_fold_input,
        )

        json_input = (
            self.test_features_dir
            / "af3_features"
            / "protein"
            / "P61626_af3_input.json"
        )
        raw_payload = json.loads(json_input.read_text())
        expected_protein = raw_payload["sequences"][0]["protein"]
        expected_template_count = len(expected_protein["templates"])
        self.assertGreater(expected_template_count, 0)

        mappings = AlphaFold3Backend.prepare_input(
            objects_to_model=[
                {
                    "object": {"json_input": str(json_input)},
                    "output_dir": str(self.output_dir),
                }
            ],
            random_seed=42,
        )
        self.assertLen(mappings, 1)
        fold_input_obj, _ = next(iter(mappings[0].items()))

        self.assertEqual([chain.id for chain in fold_input_obj.chains], ["A"])
        self.assertLen(fold_input_obj.chains, 1)
        self.assertIsInstance(fold_input_obj.chains[0], folding_input.ProteinChain)
        self.assertLen(fold_input_obj.chains[0].templates, expected_template_count)

        process_fold_input(
            fold_input=fold_input_obj,
            model_runner=None,
            output_dir=str(self.output_dir),
            buckets=(512,),
        )
        input_json = self.output_dir / f"{fold_input_obj.sanitised_name()}_data.json"
        with open(input_json, "rt") as handle:
            written = json.load(handle)

        protein_entries = [
            sequence_entry["protein"]
            for sequence_entry in written.get("sequences", [])
            if "protein" in sequence_entry
        ]
        self.assertLen(protein_entries, 1)
        self.assertEqual(protein_entries[0]["id"], "A")
        self.assertEqual(
            len(protein_entries[0]["templates"]),
            expected_template_count,
        )
        self.assertTrue(
            all(template["mmcif"] for template in protein_entries[0]["templates"])
        )
        self.assertTrue(
            all(template["queryIndices"] for template in protein_entries[0]["templates"])
        )
        self.assertTrue(
            all(template["templateIndices"] for template in protein_entries[0]["templates"])
        )

    def test_af3_viewer_output_renumbers_gapped_residue_ids_for_viewers(self):
        """Viewer-safe AF3 output must use sequential label IDs for gapped chains."""
        from alphafold3.common import folding_input
        from alphafold3.constants import chemical_components
        from alphafold3.model import model as af3_model
        from alphapulldown.folding_backend.alphafold3_backend import (
            _make_viewer_compatible_inference_result,
        )

        original_residue_ids = [2, 3, 4, 5, 8, 9, 10]
        chain = folding_input.ProteinChain(
            id="A",
            sequence="ACDEFGH",
            ptms=[],
            residue_ids=original_residue_ids,
            unpaired_msa="",
            paired_msa="",
            templates=[],
        )
        fold_input = folding_input.Input(
            name="gapped_residue_ids_for_viewers",
            chains=[chain],
            rng_seeds=[1],
        )
        struc = fold_input.to_structure(ccd=chemical_components.Ccd())
        inference_result = af3_model.InferenceResult(
            predicted_structure=struc,
            metadata={
                "token_chain_ids": ["A"] * len(original_residue_ids),
                "token_res_ids": original_residue_ids,
            },
        )

        viewer_result = _make_viewer_compatible_inference_result(inference_result)

        self.assertEqual(
            viewer_result.predicted_structure.present_residues.id.tolist(),
            list(range(1, len(original_residue_ids) + 1)),
        )
        self.assertEqual(
            viewer_result.metadata["token_res_ids"],
            list(range(1, len(original_residue_ids) + 1)),
        )
        self.assertEqual(
            viewer_result.predicted_structure.residues_table.auth_seq_id.tolist(),
            [str(residue_id) for residue_id in original_residue_ids],
        )
        self.assertEqual(
            viewer_result.predicted_structure.residues_table.insertion_code.tolist(),
            ["."] * len(original_residue_ids),
        )
        self.assertEqual(
            viewer_result.metadata["token_auth_res_ids"],
            [str(residue_id) for residue_id in original_residue_ids],
        )
        self.assertEqual(
            viewer_result.metadata["token_auth_res_labels"],
            [str(residue_id) for residue_id in original_residue_ids],
        )

    def test_af3_viewer_output_uses_insertion_codes_for_duplicate_residue_ids(self):
        """Viewer-safe AF3 output must preserve IDs and disambiguate with insertions."""
        from alphafold3.common import folding_input
        from alphafold3.constants import chemical_components
        from alphafold3.model import model as af3_model
        from alphapulldown.folding_backend.alphafold3_backend import (
            _make_viewer_compatible_inference_result,
        )

        original_residue_ids = (
            list(range(1, 11)) + list(range(2, 6)) + list(range(12, 16))
        )
        chain = folding_input.ProteinChain(
            id="A",
            sequence="ACDEFGHIKLCDEFMNPQ",
            ptms=[],
            residue_ids=original_residue_ids,
            unpaired_msa="",
            paired_msa="",
            templates=[],
        )
        fold_input = folding_input.Input(
            name="duplicate_residue_ids_for_chimerax",
            chains=[chain],
            rng_seeds=[1],
        )
        struc = fold_input.to_structure(ccd=chemical_components.Ccd())
        inference_result = af3_model.InferenceResult(
            predicted_structure=struc,
            metadata={
                "token_chain_ids": ["A"] * len(original_residue_ids),
                "token_res_ids": original_residue_ids,
            },
        )

        viewer_result = _make_viewer_compatible_inference_result(
            inference_result
        )

        self.assertEqual(
            viewer_result.predicted_structure.present_residues.id.tolist(),
            list(range(1, len(original_residue_ids) + 1)),
        )
        self.assertEqual(
            viewer_result.metadata["token_res_ids"],
            list(range(1, len(original_residue_ids) + 1)),
        )
        self.assertEqual(
            viewer_result.predicted_structure.residues_table.auth_seq_id.tolist(),
            [str(residue_id) for residue_id in original_residue_ids],
        )
        self.assertEqual(
            viewer_result.predicted_structure.residues_table.insertion_code.tolist(),
            ['.'] * 10 + ['A'] * 4 + ['.'] * 4,
        )
        self.assertEqual(
            viewer_result.metadata["token_auth_res_ids"],
            [str(residue_id) for residue_id in original_residue_ids],
        )
        self.assertEqual(
            viewer_result.metadata["token_pdb_ins_codes"],
            ['.'] * 10 + ['A'] * 4 + ['.'] * 4,
        )
        self.assertEqual(
            viewer_result.metadata["token_auth_res_labels"],
            [str(i) for i in range(1, 11)]
            + [f"{i}A" for i in range(2, 6)]
            + [str(i) for i in range(12, 16)],
        )

    def test_af3_viewer_output_handles_many_tokens_for_one_residue(self):
        """Viewer metadata must not crash when many tokens map to one residue."""
        from alphafold3.common import folding_input
        from alphafold3.constants import chemical_components
        from alphafold3.model import model as af3_model
        from alphapulldown.folding_backend.alphafold3_backend import (
            _make_viewer_compatible_inference_result,
        )

        chain = folding_input.ProteinChain(
            id="L",
            sequence="A",
            ptms=[],
            residue_ids=[1],
            unpaired_msa="",
            paired_msa="",
            templates=[],
        )
        fold_input = folding_input.Input(
            name="many_tokens_one_residue",
            chains=[chain],
            rng_seeds=[1],
        )
        struc = fold_input.to_structure(ccd=chemical_components.Ccd())
        token_count = 40
        inference_result = af3_model.InferenceResult(
            predicted_structure=struc,
            metadata={
                "token_chain_ids": ["L"] * token_count,
                "token_res_ids": [1] * token_count,
            },
        )

        viewer_result = _make_viewer_compatible_inference_result(inference_result)

        self.assertEqual(
            viewer_result.metadata["token_res_ids"],
            list(range(1, token_count + 1)),
        )
        self.assertEqual(
            viewer_result.metadata["token_auth_res_ids"],
            ["1"] * token_count,
        )
        self.assertEqual(
            viewer_result.metadata["token_pdb_ins_codes"][:27],
            ["."] + [chr(ord("A") + index) for index in range(26)],
        )
        self.assertEqual(
            viewer_result.metadata["token_pdb_ins_codes"][27:],
            ["."] * (token_count - 27),
        )
        self.assertEqual(
            viewer_result.metadata["token_auth_res_labels"][:27],
            ["1"] + [f"1{chr(ord('A') + index)}" for index in range(26)],
        )
        self.assertEqual(
            viewer_result.metadata["token_auth_res_labels"][27],
            "1[28]",
        )
        self.assertEqual(
            viewer_result.metadata["token_auth_res_labels"][-1],
            "1[40]",
        )

    def test_af3_keeps_discontinuous_chopped_regions_in_one_gapped_chain(self):
        """AF3 must keep multi-region chopped inputs as one gapped protein chain."""
        from alphapulldown.folding_backend.alphafold3_backend import (
            AlphaFold3Backend,
            process_fold_input,
        )

        parsed = parse_fold(
            ["TEST+A0A075B6L2:1-10:2-5:12-15"],
            [str(self.test_features_dir)],
            "+",
        )
        data = create_custom_info(parsed)
        all_interactors = create_interactors(data, [str(self.test_features_dir)])
        self.assertLen(all_interactors, 1)
        self.assertLen(all_interactors[0], 2)

        object_to_model = MultimericObject(interactors=all_interactors[0], pair_msa=True)
        objects_to_model = [{"object": object_to_model, "output_dir": str(self.output_dir)}]

        mappings = AlphaFold3Backend.prepare_input(
            objects_to_model=objects_to_model,
            random_seed=42,
        )
        self.assertLen(mappings, 1)
        fold_input_obj, _ = next(iter(mappings[0].items()))

        chopped_region_sequences = self._get_region_sequences(
            "A0A075B6L2",
            [(1, 10), (2, 5), (12, 15)],
        )
        concatenated_chopped_sequence = "".join(chopped_region_sequences)
        expected_sequences = [
            self._get_sequence_for_protein("TEST"),
            concatenated_chopped_sequence,
        ]
        expected_chopped_residue_ids = (
            list(range(1, 11))
            + [2, 3, 4, 5]
            + list(range(12, 16))
        )
        actual_sequences = [chain.sequence for chain in fold_input_obj.chains]
        self.assertCountEqual(actual_sequences, expected_sequences)
        self.assertLen(actual_sequences, 2)

        chopped_chains = [
            chain for chain in fold_input_obj.chains
            if chain.sequence == concatenated_chopped_sequence
        ]
        self.assertLen(chopped_chains, 1)
        self.assertEqual(
            list(chopped_chains[0].residue_ids),
            expected_chopped_residue_ids,
        )

        process_fold_input(
            fold_input=fold_input_obj,
            model_runner=None,
            output_dir=str(self.output_dir),
            buckets=(512,),
        )
        input_json = self.output_dir / f"{fold_input_obj.sanitised_name()}_data.json"
        with open(input_json, "rt") as handle:
            data = json.load(handle)

        protein_entries = [
            sequence_entry["protein"]
            for sequence_entry in data.get("sequences", [])
            if "protein" in sequence_entry
        ]
        self.assertLen(protein_entries, 2)
        self.assertCountEqual(
            [entry["sequence"] for entry in protein_entries],
            expected_sequences,
        )
        chopped_entries = [
            entry for entry in protein_entries
            if entry["sequence"] == concatenated_chopped_sequence
        ]
        self.assertLen(chopped_entries, 1)
        self.assertEqual(
            chopped_entries[0]["residueIds"],
            expected_chopped_residue_ids,
        )

        print("✓ AF3 input keeps discontinuous chopped regions as one gapped chain")

    def test_af3_keeps_two_out_of_order_gapped_copies_as_two_chains(self):
        """AF3 must keep two copied out-of-order gapped regions as two chains."""
        from alphapulldown.folding_backend.alphafold3_backend import (
            AlphaFold3Backend,
            process_fold_input,
        )

        parsed = parse_fold(
            ["A0A075B6L2:2:8-10:2-5"],
            [str(self.test_features_dir)],
            "+",
        )

        data = create_custom_info(parsed)
        all_interactors = create_interactors(data, [str(self.test_features_dir)])
        self.assertLen(all_interactors, 1)
        self.assertLen(all_interactors[0], 2)

        objects_to_model = [{"object": all_interactors[0], "output_dir": str(self.output_dir)}]
        mappings = AlphaFold3Backend.prepare_input(
            objects_to_model=objects_to_model,
            random_seed=42,
        )
        self.assertLen(mappings, 1)
        fold_input_obj, _ = next(iter(mappings[0].items()))

        expected_regions = [(8, 10), (2, 5)]
        expected_sequence = "".join(
            self._get_region_sequences("A0A075B6L2", expected_regions)
        )
        expected_residue_ids = [8, 9, 10, 2, 3, 4, 5]

        self.assertEqual(
            [chain.id for chain in fold_input_obj.chains],
            ["A", "B"],
        )
        self.assertEqual(
            [chain.sequence for chain in fold_input_obj.chains],
            [expected_sequence, expected_sequence],
        )
        self.assertEqual(
            [list(chain.residue_ids) for chain in fold_input_obj.chains],
            [expected_residue_ids, expected_residue_ids],
        )

        process_fold_input(
            fold_input=fold_input_obj,
            model_runner=None,
            output_dir=str(self.output_dir),
            buckets=(512,),
        )
        input_json = self.output_dir / f"{fold_input_obj.sanitised_name()}_data.json"
        with open(input_json, "rt") as handle:
            written = json.load(handle)

        protein_entries = [
            sequence_entry["protein"]
            for sequence_entry in written.get("sequences", [])
            if "protein" in sequence_entry
        ]
        self.assertLen(protein_entries, 1)
        self.assertEqual(protein_entries[0]["id"], ["A", "B"])
        self.assertEqual(protein_entries[0]["sequence"], expected_sequence)
        self.assertEqual(protein_entries[0]["residueIds"], expected_residue_ids)

        print("✓ AF3 input keeps two copied out-of-order gapped regions as two chains")

    def test_af3_json_feature_ranges_collapse_into_one_gapped_chain(self):
        """AF3 JSON feature files with ranges must collapse into one gapped chain."""
        from alphapulldown.folding_backend.alphafold3_backend import (
            AlphaFold3Backend,
            process_fold_input,
        )

        feature_dir = self.test_features_dir / "af3_features" / "protein"
        json_filename = "A0A024R1R8_af3_input.json"
        parsed = parse_fold(
            [f"{json_filename}:2-5:8-10"],
            [str(feature_dir)],
            "+",
        )
        self.assertEqual(
            parsed,
            [[
                {
                    "json_input": str(feature_dir / json_filename),
                    "regions": [(2, 5), (8, 10)],
                }
            ]],
        )

        data = create_custom_info(parsed)
        all_interactors = create_interactors(data, [str(feature_dir)])
        self.assertLen(all_interactors, 1)
        self.assertLen(all_interactors[0], 1)
        self.assertIsInstance(all_interactors[0][0], dict)

        objects_to_model = [{"object": all_interactors[0][0], "output_dir": str(self.output_dir)}]
        mappings = AlphaFold3Backend.prepare_input(
            objects_to_model=objects_to_model,
            random_seed=42,
        )
        self.assertLen(mappings, 1)
        fold_input_obj, _ = next(iter(mappings[0].items()))

        json_sequences = self._get_sequence_from_json(
            "af3_features/protein/A0A024R1R8_af3_input.json"
        )
        self.assertLen(json_sequences, 1)
        full_sequence = json_sequences[0][1]
        expected_sequence = full_sequence[1:5] + full_sequence[7:10]
        expected_residue_ids = [2, 3, 4, 5, 8, 9, 10]
        self.assertEqual(
            [chain.sequence for chain in fold_input_obj.chains],
            [expected_sequence],
        )
        self.assertEqual(
            fold_input_obj.sanitised_name(),
            "A0A024R1R8__2-5_8-10",
        )
        self.assertEqual(
            [list(chain.residue_ids) for chain in fold_input_obj.chains],
            [expected_residue_ids],
        )

        process_fold_input(
            fold_input=fold_input_obj,
            model_runner=None,
            output_dir=str(self.output_dir),
            buckets=(512,),
        )
        input_json = self.output_dir / f"{fold_input_obj.sanitised_name()}_data.json"
        with open(input_json, "rt") as handle:
            written = json.load(handle)

        protein_entries = [
            sequence_entry["protein"]
            for sequence_entry in written.get("sequences", [])
            if "protein" in sequence_entry
        ]
        self.assertLen(protein_entries, 1)
        self.assertEqual(protein_entries[0]["sequence"], expected_sequence)
        self.assertEqual(protein_entries[0]["residueIds"], expected_residue_ids)

        print("✓ AF3 JSON feature ranges collapse into one gapped chain")



if __name__ == "__main__":
    absltest.main()
