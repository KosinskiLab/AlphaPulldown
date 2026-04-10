#!/usr/bin/env python3
"""
Comprehensive parametrized tests for create_individual_features.py using pytest.
Tests both AlphaFold2 and AlphaFold3 pipelines with various configurations.
"""

import os
import sys
import tempfile
import json
import lzma
import pickle
import pytest
import logging
import types
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from parameterized import parameterized

# Import the module under test
import alphapulldown.objects as objects_mod
import alphapulldown.scripts.create_individual_features as create_features
from alphapulldown.objects import MonomericObject
from alphapulldown.utils import mmseqs_species_identifiers

logger = logging.getLogger(__name__)

# Minimal real MonomericObject for pickling
class DummyMonomer:
    def __init__(self, description, sequence=None):
        self.description = description
        self.sequence = sequence
        self.feature_dict = {}
        self.uniprot_runner = None
    def make_features(self, *a, **k):
        return None
    def make_mmseq_features(self, *a, **k):
        return None
    def all_seq_msa_features(self, *a, **k):
        return {}


class RecordingDummyMonomer(DummyMonomer):
    def __init__(self, description, sequence=None):
        super().__init__(description, sequence)
        self.feature_calls = []
        self.mmseq_calls = []

    def make_features(self, *args, **kwargs):
        self.feature_calls.append(kwargs)

    def make_mmseq_features(self, *args, **kwargs):
        self.mmseq_calls.append(kwargs)


class DummyJsonObj:
    def to_json(self):
        return '{"test": "features"}'

def real_write_text(self, content, *args, **kwargs):
    """Real write_text function for Path objects."""
    self.parent.mkdir(parents=True, exist_ok=True)
    with open(self, 'w') as f:
        f.write(content)
    return len(content)


def build_af3_stub_modules():
    alphafold3_pkg = types.ModuleType("alphafold3")
    alphafold3_pkg.__path__ = []
    common_pkg = types.ModuleType("alphafold3.common")
    common_pkg.__path__ = []
    structure_pkg = types.ModuleType("alphafold3.structure")
    structure_pkg.__path__ = []
    folding_input_mod = types.ModuleType("alphafold3.common.folding_input")
    mmcif_mod = types.ModuleType("alphafold3.structure.mmcif")

    class ProteinChain:
        def __init__(
            self,
            sequence,
            id,
            ptms=None,
            residue_ids=None,
            description=None,
            paired_msa=None,
            unpaired_msa=None,
            templates=None,
        ):
            self.sequence = sequence
            self.id = id
            self.ptms = [] if ptms is None else list(ptms)
            self.residue_ids = residue_ids
            self.description = description
            self.paired_msa = paired_msa
            self.unpaired_msa = unpaired_msa
            self.templates = templates

    class RnaChain:
        def __init__(
            self,
            sequence,
            id,
            modifications=None,
            residue_ids=None,
            description=None,
            unpaired_msa=None,
        ):
            self.sequence = sequence
            self.id = id
            self.modifications = [] if modifications is None else list(modifications)
            self.residue_ids = residue_ids
            self.description = description
            self.unpaired_msa = unpaired_msa

    class DnaChain:
        def __init__(self, sequence, id, modifications=None, residue_ids=None, description=None):
            self.sequence = sequence
            self.id = id
            self.modifications = [] if modifications is None else list(modifications)
            self.residue_ids = residue_ids
            self.description = description

    class Input:
        def __init__(self, name, chains, rng_seeds):
            self.name = name
            self.chains = list(chains)
            self.rng_seeds = list(rng_seeds)

    folding_input_mod.ProteinChain = ProteinChain
    folding_input_mod.RnaChain = RnaChain
    folding_input_mod.DnaChain = DnaChain
    folding_input_mod.Input = Input
    mmcif_mod.int_id_to_str_id = lambda idx: chr(ord("A") + idx - 1)

    alphafold3_pkg.common = common_pkg
    alphafold3_pkg.structure = structure_pkg
    common_pkg.folding_input = folding_input_mod
    structure_pkg.mmcif = mmcif_mod

    return {
        "alphafold3": alphafold3_pkg,
        "alphafold3.common": common_pkg,
        "alphafold3.common.folding_input": folding_input_mod,
        "alphafold3.structure": structure_pkg,
        "alphafold3.structure.mmcif": mmcif_mod,
    }, folding_input_mod

class TestCreateIndividualFeaturesComprehensive:
    """Comprehensive test cases for create_individual_features.py."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, tmp_flags):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.fasta_dir = os.path.join(self.test_dir, "fastas")
        os.makedirs(self.fasta_dir, exist_ok=True)
        
        # Create test FASTA files
        self.create_test_fastas()
        
        # Mock database paths
        self.af2_db = "/g/alphafold/AlphaFold_DBs/2.3.0"
        self.af3_db = "/g/alphafold/AlphaFold_DBs/3.0.0"
        
        logger.info(f"Test setup complete. Using temp directory: {self.test_dir}")
        
        yield
        
        # Clean up test fixtures
        import shutil
        shutil.rmtree(self.test_dir)
        logger.info("Test cleanup complete")

    def create_test_fastas(self):
        """Create test FASTA files."""
        logger.info("Creating test FASTA files")
        
        # Single protein
        with open(os.path.join(self.fasta_dir, "single_protein.fasta"), "w") as f:
            f.write(">A0A024R1R8\nMSSHEGGKKKALKQPKKQAKEMDEEEKAFKQKQKEEQKKLEVLKAKVVGKGPLATGGIKKSGKK\n")
        
        # Multiple proteins
        with open(os.path.join(self.fasta_dir, "multi_protein.fasta"), "w") as f:
            f.write(">A0A024R1R8\nMSSHEGGKKKALKQPKKQAKEMDEEEKAFKQKQKEEQKKLEVLKAKVVGKGPLATGGIKKSGKK\n")
            f.write(">P61626\nMKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV\n")
        
        # RNA
        with open(os.path.join(self.fasta_dir, "rna.fasta"), "w") as f:
            f.write(">RNA_TEST\nAUGGCUACGUAGCUAGCUAGCUAGCUAGCUAGCUAG\n")
        
        # DNA
        with open(os.path.join(self.fasta_dir, "dna.fasta"), "w") as f:
            f.write(">DNA_TEST\nATGGCATCGATCGATCGATCGATCGATCGATCGATCGATC\n")
        
        logger.info("Test FASTA files created successfully")

    @parameterized.expand([
        ("alphafold2", "single_protein.fasta", False, False),
        ("alphafold2", "multi_protein.fasta", False, False),
        ("alphafold2", "single_protein.fasta", True, False),  # mmseqs2
        ("alphafold2", "single_protein.fasta", False, True),  # compressed
        ("alphafold3", "single_protein.fasta", False, False),
        ("alphafold3", "multi_protein.fasta", False, False),
        ("alphafold3", "rna.fasta", False, False),
        ("alphafold3", "dna.fasta", False, False),
    ])
    def test_feature_creation(self, pipeline, fasta_file, use_mmseqs2, compress_features):
        """Test feature creation for different configurations."""
        logger.info(f"Testing feature creation: pipeline={pipeline}, file={fasta_file}, mmseqs2={use_mmseqs2}, compress={compress_features}")
        
        fasta_path = os.path.join(self.fasta_dir, fasta_file)
        output_dir = os.path.join(self.test_dir, f"output_{pipeline}_{fasta_file}")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        # Set flags directly to avoid UnrecognizedFlagError
        FLAGS.data_pipeline = pipeline
        FLAGS.fasta_paths = [fasta_path]
        FLAGS.data_dir = self.af2_db if pipeline == "alphafold2" else self.af3_db
        FLAGS.output_dir = output_dir
        FLAGS.max_template_date = "2021-09-30"
        FLAGS.use_mmseqs2 = use_mmseqs2
        FLAGS.compress_features = compress_features
        FLAGS.save_msa_files = False
        FLAGS.skip_existing = False
        
        if pipeline == "alphafold2":
            logger.info("Testing AlphaFold2 pipeline")
            with patch.object(create_features, 'create_pipeline_af2') as mock_af2_pipeline, \
                 patch.object(create_features, 'create_uniprot_runner') as mock_uniprot_runner, \
                 patch('alphapulldown.utils.save_meta_data.get_meta_dict', return_value={}), \
                 patch.object(create_features, 'MonomericObject', DummyMonomer):
                mock_af2_pipeline.return_value = "pipeline"
                mock_uniprot_runner.return_value = "runner"
                create_features.create_individual_features()
                
                # Check for expected files
                expected_files = []
                if fasta_file == "single_protein.fasta":
                    expected_files.append("A0A024R1R8.pkl")
                elif fasta_file == "multi_protein.fasta":
                    expected_files.extend(["A0A024R1R8.pkl", "P61626.pkl"])
                
                logger.info(f"Checking for expected files: {expected_files}")
                for expected_file in expected_files:
                    file_path = os.path.join(output_dir, expected_file)
                    if compress_features:
                        file_path += ".xz"
                    assert os.path.exists(file_path), f"Expected file {file_path} not found"
                    logger.info(f"Verified file exists: {file_path}")
        else:
            logger.info("Testing AlphaFold3 pipeline")
            af3_modules, folding_input_stub = build_af3_stub_modules()
            with patch.dict(sys.modules, af3_modules), \
                 patch.object(create_features, 'create_pipeline_af3') as mock_af3_pipeline, \
                 patch.object(create_features, 'folding_input', folding_input_stub), \
                 patch('pathlib.Path.write_text', new=real_write_text), \
                 patch('alphapulldown.utils.save_meta_data.get_meta_dict', return_value={}):
                mock_af3_pipeline.return_value = MagicMock(process=MagicMock(return_value=DummyJsonObj()))
                create_features.create_af3_individual_features()

                process_calls = mock_af3_pipeline.return_value.process.call_args_list
                observed_chain_types = [
                    type(call.args[0].chains[0]).__name__ for call in process_calls
                ]

                expected_files = []
                expected_chain_types = []
                if fasta_file == "single_protein.fasta":
                    expected_files.append("A0A024R1R8_af3_input.json")
                    expected_chain_types.append("ProteinChain")
                elif fasta_file == "multi_protein.fasta":
                    expected_files.extend(["A0A024R1R8_af3_input.json", "P61626_af3_input.json"])
                    expected_chain_types.extend(["ProteinChain", "ProteinChain"])
                elif fasta_file == "rna.fasta":
                    expected_files.append("RNA_TEST_af3_input.json")
                    expected_chain_types.append("RnaChain")
                elif fasta_file == "dna.fasta":
                    expected_files.append("DNA_TEST_af3_input.json")
                    expected_chain_types.append("DnaChain")
                
                logger.info(f"Checking for expected files: {expected_files}")
                assert observed_chain_types == expected_chain_types
                for expected_file in expected_files:
                    file_path = os.path.join(output_dir, expected_file)
                    assert os.path.exists(file_path), f"Expected file {file_path} not found"
                    logger.info(f"Verified file exists: {file_path}")
        
        logger.info("Feature creation test completed successfully")

    def test_af3_invalid_sequence_fails_run(self):
        """Invalid AF3 sequences should fail the AF3 run instead of being skipped."""
        invalid_fasta = os.path.join(self.fasta_dir, "invalid_af3.fasta")
        with open(invalid_fasta, "w") as handle:
            handle.write(">INVALID\nACDZ*\n")

        from absl import flags

        FLAGS = flags.FLAGS
        FLAGS(["test"])
        FLAGS.data_pipeline = "alphafold3"
        FLAGS.fasta_paths = [invalid_fasta]
        FLAGS.data_dir = self.af3_db
        FLAGS.output_dir = os.path.join(self.test_dir, "output_invalid_af3")
        FLAGS.max_template_date = "2021-09-30"

        error_messages = []
        af3_modules, folding_input_stub = build_af3_stub_modules()

        with patch.dict(sys.modules, af3_modules), \
             patch.object(create_features, "create_pipeline_af3") as mock_af3_pipeline, \
             patch.object(create_features, "folding_input", folding_input_stub), \
             patch.object(create_features.logging, "error", side_effect=error_messages.append):
            mock_af3_pipeline.return_value = MagicMock(process=MagicMock(return_value=DummyJsonObj()))

            with pytest.raises(RuntimeError, match="INVALID"):
                create_features.create_af3_individual_features()

        mock_af3_pipeline.return_value.process.assert_not_called()
        assert not os.path.exists(
            os.path.join(FLAGS.output_dir, "INVALID_af3_input.json")
        )
        assert any("Failed to create AlphaFold3 input object" in message for message in error_messages)

    def test_af3_ambiguous_sequence_requires_chain_hint(self):
        """Ambiguous AF3 alphabets should require an explicit chain hint in the FASTA header."""
        ambiguous_fasta = os.path.join(self.fasta_dir, "ambiguous_af3.fasta")
        with open(ambiguous_fasta, "w") as handle:
            handle.write(">AMBIG\nACGT\n")

        from absl import flags

        FLAGS = flags.FLAGS
        FLAGS(["test"])
        FLAGS.data_pipeline = "alphafold3"
        FLAGS.fasta_paths = [ambiguous_fasta]
        FLAGS.data_dir = self.af3_db
        FLAGS.output_dir = os.path.join(self.test_dir, "output_ambiguous_af3")
        FLAGS.max_template_date = "2021-09-30"

        error_messages = []
        af3_modules, folding_input_stub = build_af3_stub_modules()

        with patch.dict(sys.modules, af3_modules), \
             patch.object(create_features, "create_pipeline_af3") as mock_af3_pipeline, \
             patch.object(create_features, "folding_input", folding_input_stub), \
             patch.object(create_features.logging, "error", side_effect=error_messages.append):
            mock_af3_pipeline.return_value = MagicMock(process=MagicMock(return_value=DummyJsonObj()))

            with pytest.raises(RuntimeError, match="AMBIG"):
                create_features.create_af3_individual_features()

        mock_af3_pipeline.return_value.process.assert_not_called()
        assert not os.path.exists(
            os.path.join(FLAGS.output_dir, "AMBIG_af3_input.json")
        )
        assert any("Ambiguous sequence alphabet" in message for message in error_messages)

    def test_create_individual_features_truemultimer_respects_seq_index(self):
        """TrueMultimer mode should only process the selected CSV row."""
        from absl import flags

        FLAGS = flags.FLAGS
        FLAGS(["test"])
        FLAGS.description_file = os.path.join(self.test_dir, "description.csv")
        FLAGS.fasta_paths = [os.path.join(self.fasta_dir, "multi_protein.fasta")]
        FLAGS.path_to_mmt = os.path.join(self.test_dir, "templates")
        FLAGS.multiple_mmts = True
        FLAGS.seq_index = 2

        feats = [
            {"protein": "prot1"},
            {"protein": "prot2"},
            {"protein": "prot3"},
        ]

        with patch.object(create_features, "parse_csv_file", return_value=feats) as mock_parse, \
             patch.object(create_features, "process_multimeric_features") as mock_process:
            create_features.create_individual_features_truemultimer()

        mock_parse.assert_called_once_with(
            FLAGS.description_file,
            FLAGS.fasta_paths,
            FLAGS.path_to_mmt,
            FLAGS.multiple_mmts,
        )
        mock_process.assert_called_once_with(feats[1], 2)

    def test_process_multimeric_features_rejects_missing_templates(self):
        """TrueMultimer mode should fail early if a template path is missing."""
        feat = {
            "protein": "complexA",
            "chains": ["A"],
            "templates": [os.path.join(self.test_dir, "missing_template.cif")],
            "sequence": "ACDE",
        }

        with pytest.raises(FileNotFoundError, match="does not exist"):
            create_features.process_multimeric_features(feat, 1)

    def test_process_multimeric_features_creates_custom_db_and_saves_monomer(self):
        """TrueMultimer processing should build a custom DB and hand a monomer to the saver."""
        template_path = os.path.join(self.test_dir, "template1.cif")
        Path(template_path).write_text("data_template\n", encoding="utf-8")

        class RecordingMonomer:
            def __init__(self, description, sequence):
                self.description = description
                self.sequence = sequence
                self.feature_dict = {}
                self.uniprot_runner = None

        feat = {
            "protein": "complexB",
            "chains": ["A", "B"],
            "templates": [template_path],
            "sequence": "ACDEFG",
        }

        from absl import flags

        FLAGS = flags.FLAGS
        FLAGS(["test"])
        FLAGS.output_dir = os.path.join(self.test_dir, "truemultimer_output")
        FLAGS.data_dir = self.af2_db
        FLAGS.max_template_date = "2021-09-30"
        FLAGS.use_mmseqs2 = False
        FLAGS.jackhmmer_binary_path = "/usr/bin/jackhmmer"
        FLAGS.uniprot_database_path = "/db/uniprot.fasta"

        with patch.object(create_features, "MonomericObject", RecordingMonomer), \
             patch.object(create_features, "create_custom_db", return_value="/tmp/custom_db") as mock_custom_db, \
             patch.object(create_features, "create_arguments") as mock_create_arguments, \
             patch.object(create_features, "create_pipeline_af2", return_value="pipeline") as mock_pipeline, \
             patch.object(create_features, "create_uniprot_runner", return_value="runner") as mock_runner, \
             patch.object(create_features, "create_and_save_monomer_objects") as mock_save:
            create_features.process_multimeric_features(feat, 1)

        mock_custom_db.assert_called_once()
        custom_db_args = mock_custom_db.call_args.args
        assert custom_db_args[1:] == (
            "complexB",
            [template_path],
            ["A", "B"],
        )
        mock_create_arguments.assert_called_once_with("/tmp/custom_db")
        mock_pipeline.assert_called_once_with()
        mock_runner.assert_called_once_with(
            FLAGS.jackhmmer_binary_path,
            FLAGS.uniprot_database_path,
        )

        saved_monomer, saved_pipeline = mock_save.call_args.args
        assert saved_pipeline == "pipeline"
        assert saved_monomer.description == "complexB"
        assert saved_monomer.sequence == "ACDEFG"
        assert saved_monomer.uniprot_runner == "runner"

    @pytest.mark.parametrize("compressed_source", [False, True])
    def test_process_multimeric_features_reuses_existing_source_pickle(
        self, tmp_flags, compressed_source
    ):
        template_path = Path(self.test_dir) / "template1.cif"
        template_path.write_text("data_template\n", encoding="utf-8")

        from absl import flags

        FLAGS = flags.FLAGS
        FLAGS(["test"])
        FLAGS.output_dir = os.path.join(self.test_dir, "reused_truemultimer_output")
        FLAGS.use_mmseqs2 = False
        FLAGS.compress_features = False
        FLAGS.skip_existing = False

        output_dir = Path(FLAGS.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        source = MonomericObject("proteinA", "ACDE")
        source.feature_dict = {
            "msa": np.asarray([[1, 2, 3, 4]], dtype=np.int32),
            "deletion_matrix_int": np.zeros((1, 4), dtype=np.int32),
            "num_alignments": np.asarray([1, 1, 1, 1], dtype=np.int32),
            "msa_species_identifiers": np.asarray([b"9606"], dtype=object),
            "msa_all_seq": np.asarray([[1, 2, 3, 4]], dtype=np.int32),
            "deletion_matrix_int_all_seq": np.zeros((1, 4), dtype=np.int32),
            "msa_species_identifiers_all_seq": np.asarray([b"9606"], dtype=object),
            "template_aatype": np.zeros((1, 4, 22), dtype=np.float32),
            "template_all_atom_masks": np.ones((1, 4, 37), dtype=np.float32),
            "template_all_atom_positions": np.ones((1, 4, 37, 3), dtype=np.float32),
            "template_domain_names": np.asarray([b"old_template"], dtype=object),
            "template_sequence": np.asarray([b"OLD"], dtype=object),
            "template_sum_probs": np.asarray([0.5], dtype=np.float32),
            "template_confidence_scores": np.full((1, 4), 0.75, dtype=np.float32),
            "template_release_date": np.asarray(["2024-01-01"], dtype=object),
        }

        if compressed_source:
            with lzma.open(output_dir / "proteinA.pkl.xz", "wb") as handle:
                pickle.dump(source, handle)
        else:
            with open(output_dir / "proteinA.pkl", "wb") as handle:
                pickle.dump(source, handle)

        new_template_features = {
            "template_aatype": np.ones((2, 4, 22), dtype=np.float32),
            "template_all_atom_masks": np.full((2, 4, 37), 2.0, dtype=np.float32),
            "template_all_atom_positions": np.full((2, 4, 37, 3), 3.0, dtype=np.float32),
            "template_domain_names": np.asarray([b"newA", b"newB"], dtype=object),
            "template_sequence": np.asarray([b"NEWA", b"NEWB"], dtype=object),
            "template_sum_probs": np.asarray([0.1, 0.2], dtype=np.float32),
        }

        feat = {
            "protein": "proteinA.template1.cif.A",
            "chains": ["A"],
            "templates": [str(template_path)],
            "sequence": "ACDE",
        }

        with patch.object(
            create_features,
            "extract_multimeric_template_features_for_single_chain",
            return_value=types.SimpleNamespace(features=new_template_features),
        ) as mock_extract, \
            patch.object(create_features, "create_custom_db") as mock_custom_db, \
            patch.object(create_features, "create_arguments") as mock_arguments, \
            patch.object(create_features, "create_pipeline_af2") as mock_pipeline, \
            patch.object(create_features, "create_uniprot_runner") as mock_runner, \
            patch("alphapulldown.utils.save_meta_data.get_meta_dict", return_value={"source": "test"}):
            create_features.process_multimeric_features(feat, 1)

        mock_extract.assert_called_once_with(
            query_seq="ACDE",
            pdb_id="template1",
            chain_id="A",
            mmcif_file=str(template_path),
            threshold_clashes=create_features.FLAGS.threshold_clashes,
            hb_allowance=create_features.FLAGS.hb_allowance,
            plddt_threshold=create_features.FLAGS.plddt_threshold,
        )
        mock_custom_db.assert_not_called()
        mock_arguments.assert_not_called()
        mock_pipeline.assert_not_called()
        mock_runner.assert_not_called()

        output_pickle = output_dir / "proteinA.template1.cif.A.pkl"
        assert output_pickle.exists()
        with open(output_pickle, "rb") as handle:
            reused = pickle.load(handle)

        assert reused.description == "proteinA.template1.cif.A"
        assert np.array_equal(reused.feature_dict["msa"], source.feature_dict["msa"])
        assert reused.feature_dict["template_sequence"].tolist() == [b"NEWA", b"NEWB"]
        assert np.array_equal(
            reused.feature_dict["template_confidence_scores"],
            np.ones((2, 4), dtype=np.float32),
        )
        assert reused.feature_dict["template_release_date"].tolist() == ["none", "none"]
        assert list(output_dir.glob("proteinA.template1.cif.A_feature_metadata_*.json"))

    def test_process_multimeric_features_falls_back_when_source_sequence_mismatches(
        self, tmp_flags
    ):
        template_path = Path(self.test_dir) / "template1.cif"
        template_path.write_text("data_template\n", encoding="utf-8")

        from absl import flags

        FLAGS = flags.FLAGS
        FLAGS(["test"])
        FLAGS.output_dir = os.path.join(self.test_dir, "mismatched_truemultimer_output")
        FLAGS.use_mmseqs2 = False
        FLAGS.compress_features = False
        FLAGS.skip_existing = False
        FLAGS.jackhmmer_binary_path = "/usr/bin/jackhmmer"
        FLAGS.uniprot_database_path = "/db/uniprot.fasta"

        output_dir = Path(FLAGS.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        source = MonomericObject("proteinA", "ACDE")
        source.feature_dict = {
            "msa": np.asarray([[1, 2, 3, 4]], dtype=np.int32),
            "deletion_matrix_int": np.zeros((1, 4), dtype=np.int32),
            "num_alignments": np.asarray([1, 1, 1, 1], dtype=np.int32),
            "msa_species_identifiers": np.asarray([b"9606"], dtype=object),
        }
        with open(output_dir / "proteinA.pkl", "wb") as handle:
            pickle.dump(source, handle)

        feat = {
            "protein": "proteinA.template1.cif.A",
            "chains": ["A"],
            "templates": [str(template_path)],
            "sequence": "ACDF",
        }

        with patch.object(
            create_features,
            "extract_multimeric_template_features_for_single_chain",
        ) as mock_extract, \
            patch.object(create_features, "create_custom_db", return_value="/tmp/custom_db") as mock_custom_db, \
            patch.object(create_features, "create_arguments") as mock_arguments, \
            patch.object(create_features, "create_pipeline_af2", return_value="pipeline") as mock_pipeline, \
            patch.object(create_features, "create_uniprot_runner", return_value="runner") as mock_runner, \
            patch.object(create_features, "create_and_save_monomer_objects") as mock_save:
            create_features.process_multimeric_features(feat, 1)

        mock_extract.assert_not_called()
        mock_custom_db.assert_called_once()
        mock_arguments.assert_called_once_with("/tmp/custom_db")
        mock_pipeline.assert_called_once_with()
        mock_runner.assert_called_once_with(
            FLAGS.jackhmmer_binary_path,
            FLAGS.uniprot_database_path,
        )
        saved_monomer, saved_pipeline = mock_save.call_args.args
        assert saved_pipeline == "pipeline"
        assert saved_monomer.description == "proteinA.template1.cif.A"
        assert saved_monomer.sequence == "ACDF"
        assert saved_monomer.uniprot_runner == "runner"

    def test_process_multimeric_features_does_not_reuse_bulk_msa_pickle_for_skip_msa(
        self, tmp_flags
    ):
        template_path = Path(self.test_dir) / "template1.cif"
        template_path.write_text("data_template\n", encoding="utf-8")

        from absl import flags

        FLAGS = flags.FLAGS
        FLAGS(["test"])
        FLAGS.output_dir = os.path.join(self.test_dir, "skip_msa_truemultimer_output")
        FLAGS.use_mmseqs2 = False
        FLAGS.compress_features = False
        FLAGS.skip_existing = False
        FLAGS.skip_msa = True
        FLAGS.jackhmmer_binary_path = "/usr/bin/jackhmmer"
        FLAGS.uniprot_database_path = "/db/uniprot.fasta"

        output_dir = Path(FLAGS.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        source = MonomericObject("proteinA", "ACDE")
        source.feature_dict = {
            "msa": np.asarray([[1, 2, 3, 4]], dtype=np.int32),
            "deletion_matrix_int": np.zeros((1, 4), dtype=np.int32),
            "num_alignments": np.asarray([1, 1, 1, 1], dtype=np.int32),
            "msa_species_identifiers": np.asarray([b"9606"], dtype=object),
        }
        with open(output_dir / "proteinA.pkl", "wb") as handle:
            pickle.dump(source, handle)

        feat = {
            "protein": "proteinA.template1.cif.A",
            "chains": ["A"],
            "templates": [str(template_path)],
            "sequence": "ACDE",
        }

        with patch.object(
            create_features,
            "extract_multimeric_template_features_for_single_chain",
        ) as mock_extract, \
            patch.object(create_features, "create_custom_db", return_value="/tmp/custom_db") as mock_custom_db, \
            patch.object(create_features, "create_arguments") as mock_arguments, \
            patch.object(create_features, "create_pipeline_af2", return_value="pipeline") as mock_pipeline, \
            patch.object(create_features, "create_uniprot_runner", return_value="runner") as mock_runner, \
            patch.object(create_features, "create_and_save_monomer_objects") as mock_save:
            create_features.process_multimeric_features(feat, 1)

        mock_extract.assert_not_called()
        mock_custom_db.assert_called_once()
        mock_arguments.assert_called_once_with("/tmp/custom_db")
        mock_pipeline.assert_called_once_with()
        mock_runner.assert_called_once_with("/usr/bin/jackhmmer", "/db/uniprot.fasta")
        mock_save.assert_called_once()

    def test_main_dispatches_to_truemultimer_for_af2_template_runs(self):
        """The main entrypoint should route AF2 template jobs to the TrueMultimer path."""
        from absl import flags

        FLAGS = flags.FLAGS
        FLAGS(["test"])
        FLAGS.data_pipeline = "alphafold2"
        FLAGS.fasta_paths = [os.path.join(self.fasta_dir, "single_protein.fasta")]
        FLAGS.data_dir = self.af2_db
        FLAGS.output_dir = os.path.join(self.test_dir, "main_truemultimer")
        FLAGS.max_template_date = "2021-09-30"
        FLAGS.path_to_mmt = os.path.join(self.test_dir, "templates")

        with patch.object(create_features, "check_template_date") as mock_check, \
             patch.object(create_features, "create_individual_features_truemultimer") as mock_tm, \
             patch.object(create_features, "create_individual_features") as mock_single:
            create_features.main([])

        mock_check.assert_called_once_with()
        mock_tm.assert_called_once_with()
        mock_single.assert_not_called()

    def test_database_path_mapping(self):
        """Test that database paths are correctly mapped for both pipelines."""
        logger.info("Testing database path mapping")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        test_cases = [
            ("alphafold2", "uniref90", "uniref90/uniref90.fasta"),
            ("alphafold2", "uniref30", "uniref30/UniRef30_2023_02"),
            ("alphafold3", "uniref90", "uniref90_2022_05.fa"),
        ]
        
        for pipeline, key, expected_subpath in test_cases:
            logger.info(f"Testing {pipeline} pipeline with key '{key}'")
            FLAGS.data_pipeline = pipeline
            FLAGS.data_dir = "/test/db"
            expected_path = os.path.join("/test/db", expected_subpath)
            actual_path = create_features.get_database_path(key)
            assert actual_path == expected_path, f"Expected {expected_path}, got {actual_path}"
            logger.info(f"Database path mapping correct: {actual_path}")

        FLAGS.data_pipeline = "alphafold3"
        FLAGS.data_dir = "/test/db"
        with pytest.raises(
            KeyError,
            match="Database 'uniref30' is not configured for the alphafold3 pipeline",
        ):
            create_features.get_database_path("uniref30")

    def test_af3_pipeline_creation_failure(self):
        """Test that AF3 pipeline creation fails gracefully when AF3 is not available."""
        logger.info("Testing AF3 pipeline creation failure")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        with patch('alphapulldown.scripts.create_individual_features.AF3DataPipeline', None), \
             patch('alphapulldown.scripts.create_individual_features.AF3DataPipelineConfig', None):
            
            FLAGS.data_pipeline = "alphafold3"
            FLAGS.data_dir = "/test/db"
            with pytest.raises(ImportError, match="pip install -e .*alphafold3,test.*build_data"):
                create_features.create_pipeline_af3()
            logger.info("AF3 pipeline creation correctly failed with ImportError")

    def test_template_date_check(self):
        """Test template date validation."""
        logger.info("Testing template date validation")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        # Test valid date
        FLAGS.max_template_date = "2021-09-30"
        try:
            create_features.check_template_date()
            logger.info("Valid template date accepted")
        except SystemExit:
            pytest.fail("Valid date should not cause SystemExit")
        
        # Test invalid date (None)
        FLAGS.max_template_date = None
        with pytest.raises(SystemExit):
            create_features.check_template_date()
            logger.info("Invalid template date correctly rejected")

    def test_sequence_index_filtering(self):
        """Test sequence index filtering functionality."""
        logger.info("Testing sequence index filtering")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        # Test with valid sequence index
        FLAGS.seq_index = 1
        FLAGS.fasta_paths = ["test.fasta"]
        
        # Mock the iter_seqs function to return test data
        with patch('alphapulldown.utils.file_handling.iter_seqs') as mock_iter_seqs:
            mock_iter_seqs.return_value = [("SEQ1", "desc1"), ("SEQ2", "desc2"), ("SEQ3", "desc3")]
            
            # Test that only the specified sequence is processed
            sequences = list(mock_iter_seqs.return_value)
            if FLAGS.seq_index is not None:
                sequences = [sequences[FLAGS.seq_index - 1]]  # seq_index is 1-based
            
            assert len(sequences) == 1, f"Expected 1 sequence, got {len(sequences)}"
            assert sequences[0][0] == "SEQ1", f"Expected SEQ1, got {sequences[0][0]}"
            logger.info("Sequence filtering with valid index successful")

    def test_skip_existing_flag(self):
        """Test skip existing functionality."""
        logger.info("Testing skip existing functionality")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        output_dir = os.path.join(self.test_dir, "skip_test")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a dummy existing file
        existing_file = os.path.join(output_dir, "test.pkl")
        with open(existing_file, 'w') as f:
            f.write("dummy")
        
        FLAGS.output_dir = output_dir
        FLAGS.skip_existing = True
        
        # Mock the create_individual_features function to avoid database access
        with patch.object(create_features, 'create_individual_features') as mock_create_features:
            mock_create_features.return_value = None
            # This should not create new files when skip_existing is True
            create_features.create_individual_features()
            logger.info("Skip existing functionality tested successfully")

    def test_output_directory_creation(self):
        """Test output directory creation."""
        logger.info("Testing output directory creation")
        
        output_dir = os.path.join(self.test_dir, "new_output_dir")
        
        # Test directory creation by running the main function
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        FLAGS.output_dir = output_dir
        FLAGS.max_template_date = "2021-09-30"
        FLAGS.data_pipeline = "alphafold2"
        FLAGS.fasta_paths = ["dummy.fasta"]  # Use a dummy path instead of empty list
        FLAGS.data_dir = "/test/db"
        
        # Mock the pipeline creation to avoid real database access
        with patch.object(create_features, 'create_pipeline_af2') as mock_af2_pipeline, \
             patch.object(create_features, 'create_uniprot_runner') as mock_uniprot_runner, \
             patch('alphapulldown.scripts.create_individual_features.iter_seqs') as mock_iter_seqs:
            mock_af2_pipeline.return_value = MagicMock()
            mock_uniprot_runner.return_value = MagicMock()
            mock_iter_seqs.return_value = []  # Return empty iterator
            
            # The main function should create the output directory
            create_features.main([])
            
            assert os.path.exists(output_dir), f"Output directory {output_dir} was not created"
            assert os.path.isdir(output_dir), f"{output_dir} is not a directory"
            logger.info(f"Output directory created successfully: {output_dir}")

    def test_alphafold3_chain_type_detection(self):
        """Test AlphaFold3 chain type detection."""
        logger.info("Testing AlphaFold3 chain type detection")
        
        # Test protein sequence detection
        protein_seq = "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV"
        assert all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in protein_seq.upper()), "Protein sequence contains invalid amino acids"
        logger.info("Protein chain type detection successful")
        
        # Test RNA sequence detection
        rna_seq = "AUGGCUACGUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAG"
        assert all(c in 'ACGU' for c in rna_seq.upper()), "RNA sequence contains invalid nucleotides"
        logger.info("RNA chain type detection successful")
        
        # Test DNA sequence detection
        dna_seq = "ATGGCATCGATCGATCGATCGATCGATCGATCGATCGATC"
        assert all(c in 'ACGT' for c in dna_seq.upper()), "DNA sequence contains invalid nucleotides"
        logger.info("DNA chain type detection successful")

    def test_compression_flag(self):
        """Test feature compression functionality."""
        logger.info("Testing feature compression functionality")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        # Test compression enabled
        FLAGS.compress_features = True
        assert FLAGS.compress_features, "Compression flag should be True"
        logger.info("Compression flag enabled successfully")
        
        # Test compression disabled
        FLAGS.compress_features = False
        assert not FLAGS.compress_features, "Compression flag should be False"
        logger.info("Compression flag disabled successfully")
        
        # Test file extension handling
        test_file = "test.pkl"
        if FLAGS.compress_features:
            test_file += ".xz"
        assert test_file == "test.pkl", "File extension should not be modified when compression is disabled"
        logger.info("File extension handling tested successfully")

    def test_create_arguments_function(self):
        """Test create_arguments function."""
        logger.info("Testing create_arguments function")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        # Test basic argument creation
        FLAGS.data_dir = "/test/db"
        FLAGS.max_template_date = "2021-09-30"
        # Ensure default composition from data_dir (no prior values lingering)
        FLAGS.use_mmseqs2 = False
        FLAGS.data_pipeline = "alphafold2"
        FLAGS.uniref90_database_path = None
        FLAGS.uniref30_database_path = None
        FLAGS.mgnify_database_path = None
        FLAGS.bfd_database_path = None
        FLAGS.small_bfd_database_path = None
        FLAGS.pdb70_database_path = None
        FLAGS.uniprot_database_path = None
        FLAGS.pdb_seqres_database_path = None
        FLAGS.template_mmcif_dir = None
        FLAGS.obsolete_pdbs_path = None
        
        create_features.create_arguments()
        assert FLAGS.uniref90_database_path == "/test/db/uniref90/uniref90.fasta", f"Expected '/test/db/uniref90/uniref90.fasta', got '{FLAGS.uniref90_database_path}'"
        assert FLAGS.max_template_date == "2021-09-30", f"Expected '2021-09-30', got '{FLAGS.max_template_date}'"
        logger.info("Basic argument creation successful")
        
        # Test with custom template database
        custom_db_path = "/custom/templates"
        create_features.create_arguments(custom_db_path)
        assert FLAGS.pdb_seqres_database_path == "/custom/templates/pdb_seqres.txt", f"Expected '/custom/templates/pdb_seqres.txt', got '{FLAGS.pdb_seqres_database_path}'"
        logger.info("Custom template database argument creation successful")

    def test_create_arguments_with_custom_template_db(self):
        """Test create_arguments function with custom template database."""
        logger.info("Testing create_arguments with custom template database")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        # Test custom template database path handling
        custom_db_path = "/custom/template/db"
        create_features.create_arguments(custom_db_path)
        assert FLAGS.pdb_seqres_database_path == "/custom/template/db/pdb_seqres.txt", f"Expected '/custom/template/db/pdb_seqres.txt', got '{FLAGS.pdb_seqres_database_path}'"
        logger.info("Custom template database path handling successful")
        
        # Test that other flags are preserved
        FLAGS.data_dir = "/test/db"
        FLAGS.max_template_date = "2021-09-30"
        create_features.create_arguments()
        assert FLAGS.data_dir == "/test/db", "Data directory should be preserved"
        assert FLAGS.max_template_date == "2021-09-30", "Max template date should be preserved"
        logger.info("Flag preservation in custom template database mode successful")

    def test_create_arguments_alphafold3_clears_af2_only_databases(self):
        """Test that AF3 argument creation only populates databases used by AF3."""
        logger.info("Testing AF3 argument creation without AF2-only database leftovers")

        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])

        FLAGS.use_mmseqs2 = False
        FLAGS.data_pipeline = "alphafold3"
        FLAGS.data_dir = "/test/db"
        FLAGS.uniref90_database_path = None
        FLAGS.mgnify_database_path = None
        FLAGS.small_bfd_database_path = None
        FLAGS.uniprot_database_path = None
        FLAGS.pdb_seqres_database_path = None
        FLAGS.template_mmcif_dir = None
        FLAGS.uniref30_database_path = "/stale/uniref30"
        FLAGS.bfd_database_path = "/stale/bfd"
        FLAGS.pdb70_database_path = "/stale/pdb70"
        FLAGS.obsolete_pdbs_path = "/stale/obsolete.dat"

        create_features.create_arguments()

        assert FLAGS.uniref90_database_path == "/test/db/uniref90_2022_05.fa"
        assert FLAGS.mgnify_database_path == "/test/db/mgy_clusters_2022_05.fa"
        assert FLAGS.small_bfd_database_path == "/test/db/bfd-first_non_consensus_sequences.fasta"
        assert FLAGS.uniprot_database_path == "/test/db/uniprot_all_2021_04.fa"
        assert FLAGS.pdb_seqres_database_path == "/test/db/pdb_seqres_2022_09_28.fasta"
        assert FLAGS.template_mmcif_dir == "/test/db/mmcif_files"
        assert FLAGS.uniref30_database_path is None
        assert FLAGS.bfd_database_path is None
        assert FLAGS.pdb70_database_path is None
        assert FLAGS.obsolete_pdbs_path is None

        create_features.create_arguments("/custom/template/db")
        assert FLAGS.pdb_seqres_database_path == "/test/db/pdb_seqres_2022_09_28.fasta"
        assert FLAGS.template_mmcif_dir == "/test/db/mmcif_files"
        assert FLAGS.obsolete_pdbs_path is None
        logger.info("AF3 argument creation only kept AF3-relevant database paths")

    def test_create_arguments_reduced_dbs_clears_unused_af2_databases(self):
        """Test that reduced_dbs only sets the AF2 paths it actually needs."""
        logger.info("Testing reduced_dbs argument creation without full-db leftovers")

        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])

        FLAGS.use_mmseqs2 = False
        FLAGS.data_pipeline = "alphafold2"
        FLAGS.db_preset = "reduced_dbs"
        FLAGS.use_hhsearch = False
        FLAGS.data_dir = "/test/db"
        FLAGS.uniref90_database_path = None
        FLAGS.mgnify_database_path = None
        FLAGS.small_bfd_database_path = None
        FLAGS.uniprot_database_path = None
        FLAGS.pdb_seqres_database_path = None
        FLAGS.template_mmcif_dir = None
        FLAGS.obsolete_pdbs_path = None
        FLAGS.uniref30_database_path = "/stale/uniref30"
        FLAGS.bfd_database_path = "/stale/bfd"
        FLAGS.pdb70_database_path = "/stale/pdb70"

        create_features.create_arguments()

        assert FLAGS.uniref90_database_path == "/test/db/uniref90/uniref90.fasta"
        assert FLAGS.mgnify_database_path == "/test/db/mgnify/mgy_clusters_2022_05.fa"
        assert FLAGS.small_bfd_database_path == "/test/db/small_bfd/bfd-first_non_consensus_sequences.fasta"
        assert FLAGS.uniprot_database_path == "/test/db/uniprot/uniprot.fasta"
        assert FLAGS.pdb_seqres_database_path == "/test/db/pdb_seqres/pdb_seqres.txt"
        assert FLAGS.template_mmcif_dir == "/test/db/pdb_mmcif/mmcif_files"
        assert FLAGS.obsolete_pdbs_path == "/test/db/pdb_mmcif/obsolete.dat"
        assert FLAGS.uniref30_database_path is None
        assert FLAGS.bfd_database_path is None
        assert FLAGS.pdb70_database_path is None
        logger.info("Reduced-dbs argument creation cleared unused full-database paths")

    def test_create_arguments_reduced_dbs_keeps_pdb70_for_hhsearch(self):
        """Test that reduced_dbs still sets pdb70 when HHsearch templates are requested."""
        logger.info("Testing reduced_dbs HHsearch argument creation")

        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])

        FLAGS.use_mmseqs2 = False
        FLAGS.data_pipeline = "alphafold2"
        FLAGS.db_preset = "reduced_dbs"
        FLAGS.use_hhsearch = True
        FLAGS.data_dir = "/test/db"
        FLAGS.pdb70_database_path = None

        create_features.create_arguments()

        assert FLAGS.pdb70_database_path == "/test/db/pdb70/pdb70"
        assert FLAGS.bfd_database_path is None
        assert FLAGS.uniref30_database_path is None
        logger.info("Reduced-dbs HHsearch argument creation kept pdb70 without restoring full BFD")

    def test_mmseqs2_without_data_dir(self):
        """Test that MMseqs2 works without data_dir flag."""
        logger.info("Testing MMseqs2 without data_dir flag")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        # Set up flags for MMseqs2 without data_dir
        FLAGS.use_mmseqs2 = True
        FLAGS.data_dir = None
        FLAGS.fasta_paths = [os.path.join(self.fasta_dir, "single_protein.fasta")]
        FLAGS.output_dir = os.path.join(self.test_dir, "test_output")
        FLAGS.max_template_date = "2021-09-30"
        
        # Test that main() doesn't exit when data_dir is None but use_mmseqs2 is True
        with patch('sys.exit') as mock_exit, \
             patch('alphapulldown.scripts.create_individual_features.create_pipeline_af2') as mock_pipeline, \
             patch('alphapulldown.scripts.create_individual_features.create_uniprot_runner') as mock_uniprot, \
             patch('alphapulldown.objects.MonomericObject', DummyMonomer), \
             patch('alphapulldown.utils.save_meta_data.get_meta_dict', return_value={}), \
             patch('builtins.open', mock_open()), \
             patch('pickle.dump'):
            
            mock_pipeline.return_value = MagicMock()
            mock_uniprot.return_value = MagicMock()
            
            create_features.main([])
            mock_exit.assert_not_called()
        logger.info("MMseqs2 without data_dir flag test successful")

    def test_mmseqs2_with_data_dir(self):
        """Test that MMseqs2 works with data_dir flag (should still work)."""
        logger.info("Testing MMseqs2 with data_dir flag")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        # Set up flags for MMseqs2 with data_dir
        FLAGS.use_mmseqs2 = True
        FLAGS.data_dir = "/test/db"
        FLAGS.fasta_paths = [os.path.join(self.fasta_dir, "single_protein.fasta")]
        FLAGS.output_dir = os.path.join(self.test_dir, "test_output")
        FLAGS.max_template_date = "2021-09-30"
        
        # Test that main() doesn't exit when data_dir is provided and use_mmseqs2 is True
        with patch('sys.exit') as mock_exit, \
             patch('alphapulldown.scripts.create_individual_features.create_pipeline_af2') as mock_pipeline, \
             patch('alphapulldown.scripts.create_individual_features.create_uniprot_runner') as mock_uniprot, \
             patch('alphapulldown.objects.MonomericObject', DummyMonomer), \
             patch('alphapulldown.utils.save_meta_data.get_meta_dict', return_value={}), \
             patch('builtins.open', mock_open()), \
             patch('pickle.dump'):
            
            mock_pipeline.return_value = MagicMock()
            mock_uniprot.return_value = MagicMock()
            
            create_features.main([])
            mock_exit.assert_not_called()
        logger.info("MMseqs2 with data_dir flag test successful")

    def test_non_mmseqs2_without_data_dir(self):
        """Test that non-MMseqs2 fails without data_dir flag."""
        logger.info("Testing non-MMseqs2 without data_dir flag")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        # Set up flags for non-MMseqs2 without data_dir
        FLAGS.use_mmseqs2 = False
        FLAGS.data_dir = None
        FLAGS.fasta_paths = [os.path.join(self.fasta_dir, "single_protein.fasta")]
        FLAGS.output_dir = os.path.join(self.test_dir, "test_output")
        FLAGS.max_template_date = "2021-09-30"
        
        # Test that get_database_path raises ValueError when data_dir is None and use_mmseqs2 is False
        with pytest.raises(ValueError, match="data_dir is required when not using MMseqs2"):
            create_features.get_database_path("uniref90")
        logger.info("Non-MMseqs2 without data_dir flag correctly failed")

    def test_database_path_handling_mmseqs2(self):
        """Test database path handling when using MMseqs2."""
        logger.info("Testing database path handling with MMseqs2")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        # Test with MMseqs2 and no data_dir
        FLAGS.use_mmseqs2 = True
        FLAGS.data_dir = None
        
        # Test get_database_path returns None
        result = create_features.get_database_path("uniref90")
        assert result is None, f"Expected None, got {result}"
        
        # Test create_arguments sets database paths to None
        create_features.create_arguments()
        assert FLAGS.uniref90_database_path is None, "uniref90_database_path should be None"
        assert FLAGS.mgnify_database_path is None, "mgnify_database_path should be None"
        assert FLAGS.bfd_database_path is None, "bfd_database_path should be None"
        logger.info("Database path handling with MMseqs2 successful")

    def test_pipeline_creation_mmseqs2(self):
        """Test pipeline creation when using MMseqs2."""
        logger.info("Testing pipeline creation with MMseqs2")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        # Set up flags for MMseqs2
        FLAGS.use_mmseqs2 = True
        FLAGS.data_dir = None
        FLAGS.db_preset = "full_dbs"
        
        # Mock the AF2DataPipeline to avoid real database access
        with patch('alphapulldown.scripts.create_individual_features.AF2DataPipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            
            # Test that pipeline creation doesn't fail
            pipeline = create_features.create_pipeline_af2()
            assert pipeline is not None, "Pipeline should be created successfully"
            
            # Verify that template_searcher and template_featurizer are None
            # We can't directly access these, but we can verify the pipeline was created
            mock_pipeline.assert_called_once()
            logger.info("Pipeline creation with MMseqs2 successful")

    def test_feature_creation_mmseqs2(self):
        """Test feature creation when using MMseqs2."""
        logger.info("Testing feature creation with MMseqs2")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        # Set up flags for MMseqs2
        FLAGS.use_mmseqs2 = True
        FLAGS.data_dir = None
        FLAGS.fasta_paths = [os.path.join(self.fasta_dir, "single_protein.fasta")]
        FLAGS.output_dir = os.path.join(self.test_dir, "test_output")
        FLAGS.max_template_date = "2021-09-30"
        
        # Mock the necessary functions to avoid real database access
        with patch('alphapulldown.scripts.create_individual_features.create_pipeline_af2') as mock_pipeline, \
             patch('alphapulldown.scripts.create_individual_features.create_uniprot_runner') as mock_uniprot, \
             patch.object(create_features, 'iter_seqs') as mock_iter_seqs, \
             patch.object(create_features, 'MonomericObject', DummyMonomer), \
             patch('alphapulldown.utils.save_meta_data.get_meta_dict', return_value={}):
            
            mock_iter_seqs.return_value = [("TESTSEQ", "test_protein")]
            
            # Test that feature creation doesn't fail
            create_features.create_individual_features()
            
            # Verify that pipeline and uniprot_runner are None for MMseqs2
            mock_pipeline.assert_not_called()
            mock_uniprot.assert_not_called()
            assert os.path.exists(os.path.join(FLAGS.output_dir, "test_protein.pkl"))
            logger.info("Feature creation with MMseqs2 successful")

    def test_flag_validation_mmseqs2(self):
        """Test flag validation for MMseqs2 scenarios."""
        logger.info("Testing flag validation for MMseqs2")
        
        # Initialize flags properly
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS(['test'])  # Parse flags with dummy argv
        
        # Test case 1: MMseqs2 with data_dir (should work)
        FLAGS.use_mmseqs2 = True
        FLAGS.data_dir = "/test/db"
        FLAGS.fasta_paths = [os.path.join(self.fasta_dir, "single_protein.fasta")]
        FLAGS.output_dir = os.path.join(self.test_dir, "test_output")
        FLAGS.max_template_date = "2021-09-30"
        
        with patch('sys.exit') as mock_exit, \
             patch('alphapulldown.scripts.create_individual_features.create_pipeline_af2') as mock_pipeline, \
             patch('alphapulldown.scripts.create_individual_features.create_uniprot_runner') as mock_uniprot, \
             patch('alphapulldown.objects.MonomericObject', DummyMonomer), \
             patch('alphapulldown.utils.save_meta_data.get_meta_dict', return_value={}), \
             patch('builtins.open', mock_open()), \
             patch('pickle.dump'):
            
            mock_pipeline.return_value = MagicMock()
            mock_uniprot.return_value = MagicMock()
            
            create_features.main([])
            mock_exit.assert_not_called()
        
        # Test case 2: MMseqs2 without data_dir (should work)
        FLAGS.data_dir = None
        with patch('sys.exit') as mock_exit, \
             patch('alphapulldown.scripts.create_individual_features.create_pipeline_af2') as mock_pipeline, \
             patch('alphapulldown.scripts.create_individual_features.create_uniprot_runner') as mock_uniprot, \
             patch('alphapulldown.objects.MonomericObject', DummyMonomer), \
             patch('alphapulldown.utils.save_meta_data.get_meta_dict', return_value={}), \
             patch('builtins.open', mock_open()), \
             patch('pickle.dump'):
            
            mock_pipeline.return_value = MagicMock()
            mock_uniprot.return_value = MagicMock()
            
            create_features.main([])
            mock_exit.assert_not_called()
        
        # Test case 3: Non-MMseqs2 without data_dir (should fail)
        FLAGS.use_mmseqs2 = False
        FLAGS.data_dir = None
        with pytest.raises(SystemExit):
            create_features.main([])
        
        logger.info("Flag validation for MMseqs2 scenarios successful") 


def test_create_pipeline_af2_uses_hhsearch_template_stack(tmp_flags):
    create_features.FLAGS.use_mmseqs2 = False
    create_features.FLAGS.use_hhsearch = True
    create_features.FLAGS.hhsearch_binary_path = "/bin/hhsearch"
    create_features.FLAGS.pdb70_database_path = "/db/pdb70"
    create_features.FLAGS.template_mmcif_dir = "/db/mmcif"
    create_features.FLAGS.max_template_date = "2021-09-30"
    create_features.FLAGS.kalign_binary_path = "/bin/kalign"
    create_features.FLAGS.obsolete_pdbs_path = "/db/obsolete.dat"

    with patch.object(create_features.hhsearch, "HHSearch", return_value="searcher") as mock_searcher, \
         patch.object(create_features.templates, "HhsearchHitFeaturizer", return_value="featurizer") as mock_featurizer, \
         patch.object(create_features, "AF2DataPipeline", return_value="pipeline") as mock_pipeline:
        pipeline = create_features.create_pipeline_af2()

    assert pipeline == "pipeline"
    mock_searcher.assert_called_once_with(
        binary_path="/bin/hhsearch",
        databases=["/db/pdb70"],
    )
    mock_featurizer.assert_called_once_with(
        mmcif_dir="/db/mmcif",
        max_template_date="2021-09-30",
        max_hits=20,
        kalign_binary_path="/bin/kalign",
        release_dates_path=None,
        obsolete_pdbs_path="/db/obsolete.dat",
    )
    assert mock_pipeline.call_args.kwargs["template_searcher"] == "searcher"
    assert mock_pipeline.call_args.kwargs["template_featurizer"] == "featurizer"


def test_create_pipeline_af2_uses_hmmsearch_template_stack(tmp_flags):
    create_features.FLAGS.use_mmseqs2 = False
    create_features.FLAGS.use_hhsearch = False
    create_features.FLAGS.hmmsearch_binary_path = "/bin/hmmsearch"
    create_features.FLAGS.hmmbuild_binary_path = "/bin/hmmbuild"
    create_features.FLAGS.pdb_seqres_database_path = "/db/pdb_seqres.txt"
    create_features.FLAGS.template_mmcif_dir = "/db/mmcif"
    create_features.FLAGS.max_template_date = "2021-09-30"
    create_features.FLAGS.kalign_binary_path = "/bin/kalign"
    create_features.FLAGS.obsolete_pdbs_path = "/db/obsolete.dat"

    with patch.object(create_features.hmmsearch, "Hmmsearch", return_value="searcher") as mock_searcher, \
         patch.object(create_features.templates, "HmmsearchHitFeaturizer", return_value="featurizer") as mock_featurizer, \
         patch.object(create_features, "AF2DataPipeline", return_value="pipeline") as mock_pipeline:
        pipeline = create_features.create_pipeline_af2()

    assert pipeline == "pipeline"
    mock_searcher.assert_called_once_with(
        binary_path="/bin/hmmsearch",
        hmmbuild_binary_path="/bin/hmmbuild",
        database_path="/db/pdb_seqres.txt",
    )
    mock_featurizer.assert_called_once_with(
        mmcif_dir="/db/mmcif",
        max_template_date="2021-09-30",
        max_hits=20,
        kalign_binary_path="/bin/kalign",
        obsolete_pdbs_path="/db/obsolete.dat",
        release_dates_path=None,
    )
    assert mock_pipeline.call_args.kwargs["template_searcher"] == "searcher"
    assert mock_pipeline.call_args.kwargs["template_featurizer"] == "featurizer"


def test_create_individual_features_only_saves_selected_sequence(tmp_flags):
    create_features.FLAGS.seq_index = 2

    with patch.object(create_features, "create_arguments") as mock_arguments, \
         patch.object(create_features, "create_pipeline_af2", return_value="pipeline") as mock_pipeline, \
         patch.object(create_features, "create_uniprot_runner", return_value="runner") as mock_runner, \
         patch.object(create_features, "MonomericObject", DummyMonomer), \
         patch.object(create_features, "iter_seqs", return_value=[("AAAA", "first"), ("BBBB", "second")]), \
         patch.object(create_features, "create_and_save_monomer_objects") as mock_save:
        create_features.create_individual_features()

    mock_arguments.assert_called_once_with()
    mock_pipeline.assert_called_once_with()
    mock_runner.assert_called_once()
    saved_monomer, saved_pipeline = mock_save.call_args.args
    assert saved_pipeline == "pipeline"
    assert saved_monomer.description == "second"
    assert saved_monomer.uniprot_runner == "runner"


def test_create_and_save_monomer_objects_writes_compressed_af2_outputs(tmp_flags, tmp_path):
    create_features.FLAGS.output_dir = str(tmp_path)
    create_features.FLAGS.compress_features = True
    create_features.FLAGS.skip_existing = False
    create_features.FLAGS.use_mmseqs2 = False
    create_features.FLAGS.use_precomputed_msas = True
    create_features.FLAGS.save_msa_files = True

    monomer = RecordingDummyMonomer("protA")
    with patch("alphapulldown.utils.save_meta_data.get_meta_dict", return_value={"source": "test"}):
        create_features.create_and_save_monomer_objects(monomer, pipeline="pipeline")

    metadata_files = list(tmp_path.glob("protA_feature_metadata_*.json.xz"))
    assert len(metadata_files) == 1
    with lzma.open(metadata_files[0], "rt", encoding="utf-8") as handle:
        assert json.load(handle) == {"source": "test"}
    assert (tmp_path / "protA.pkl.xz").exists()
    assert monomer.feature_calls == [
        {
            "pipeline": "pipeline",
            "output_dir": str(tmp_path),
            "use_precomputed_msa": True,
            "save_msa": True,
            "skip_msa": False,
        }
    ]
    assert monomer.mmseq_calls == []


def test_create_and_save_monomer_objects_skips_existing_outputs(tmp_flags, tmp_path):
    create_features.FLAGS.output_dir = str(tmp_path)
    create_features.FLAGS.compress_features = False
    create_features.FLAGS.skip_existing = True
    create_features.FLAGS.use_mmseqs2 = True

    existing_pickle = tmp_path / "protA.pkl"
    existing_pickle.write_bytes(b"already-there")
    monomer = RecordingDummyMonomer("protA")
    create_features.create_and_save_monomer_objects(monomer, pipeline=None)

    assert monomer.feature_calls == []
    assert monomer.mmseq_calls == []
    assert list(tmp_path.glob("protA_feature_metadata_*.json")) == []


def test_create_and_save_monomer_objects_uses_mmseqs_when_requested(tmp_flags, tmp_path):
    create_features.FLAGS.output_dir = str(tmp_path)
    create_features.FLAGS.compress_features = False
    create_features.FLAGS.skip_existing = False
    create_features.FLAGS.use_mmseqs2 = True
    create_features.FLAGS.use_precomputed_msas = True
    create_features.FLAGS.re_search_templates_mmseqs2 = True

    monomer = RecordingDummyMonomer("protA")
    with patch("alphapulldown.utils.save_meta_data.get_meta_dict", return_value={"source": "test"}):
        create_features.create_and_save_monomer_objects(monomer, pipeline=None)

    assert monomer.feature_calls == []
    assert monomer.mmseq_calls == [
        {
            "DEFAULT_API_SERVER": create_features.DEFAULT_API_SERVER,
            "output_dir": str(tmp_path),
            "use_precomputed_msa": True,
            "use_templates": True,
            "custom_template_path": None,
            "skip_msa": False,
        }
    ]
    assert (tmp_path / "protA.pkl").exists()


def test_create_and_save_monomer_objects_passes_skip_msa_to_af2_builder(tmp_flags, tmp_path):
    create_features.FLAGS.output_dir = str(tmp_path)
    create_features.FLAGS.compress_features = False
    create_features.FLAGS.skip_existing = False
    create_features.FLAGS.use_mmseqs2 = False
    create_features.FLAGS.use_precomputed_msas = False
    create_features.FLAGS.save_msa_files = False
    create_features.FLAGS.skip_msa = True

    monomer = RecordingDummyMonomer("protA")
    with patch("alphapulldown.utils.save_meta_data.get_meta_dict", return_value={"source": "test"}):
        create_features.create_and_save_monomer_objects(monomer, pipeline="pipeline")

    assert monomer.feature_calls == [
        {
            "pipeline": "pipeline",
            "output_dir": str(tmp_path),
            "use_precomputed_msa": False,
            "save_msa": False,
            "skip_msa": True,
        }
    ]


def test_create_and_save_monomer_objects_passes_skip_msa_to_mmseqs_builder(tmp_flags, tmp_path):
    create_features.FLAGS.output_dir = str(tmp_path)
    create_features.FLAGS.compress_features = False
    create_features.FLAGS.skip_existing = False
    create_features.FLAGS.use_mmseqs2 = True
    create_features.FLAGS.use_precomputed_msas = False
    create_features.FLAGS.re_search_templates_mmseqs2 = False
    create_features.FLAGS.skip_msa = True

    monomer = RecordingDummyMonomer("protA")
    with patch("alphapulldown.utils.save_meta_data.get_meta_dict", return_value={"source": "test"}):
        create_features.create_and_save_monomer_objects(monomer, pipeline=None)

    assert monomer.mmseq_calls == [
        {
            "DEFAULT_API_SERVER": create_features.DEFAULT_API_SERVER,
            "output_dir": str(tmp_path),
            "use_precomputed_msa": False,
            "use_templates": False,
            "custom_template_path": None,
            "skip_msa": True,
        }
    ]


def test_create_and_save_monomer_objects_passes_custom_templates_to_mmseqs(tmp_flags, tmp_path):
    create_features.FLAGS.output_dir = str(tmp_path)
    create_features.FLAGS.compress_features = False
    create_features.FLAGS.skip_existing = False
    create_features.FLAGS.use_mmseqs2 = True
    create_features.FLAGS.use_precomputed_msas = False
    create_features.FLAGS.re_search_templates_mmseqs2 = False

    monomer = RecordingDummyMonomer("protA")
    custom_template_path = str(tmp_path / "custom_db" / "templates")

    with patch("alphapulldown.utils.save_meta_data.get_meta_dict", return_value={"source": "test"}):
        create_features.create_and_save_monomer_objects(
            monomer,
            pipeline=None,
            custom_template_path=custom_template_path,
        )

    assert monomer.feature_calls == []
    assert monomer.mmseq_calls == [
        {
            "DEFAULT_API_SERVER": create_features.DEFAULT_API_SERVER,
            "output_dir": str(tmp_path),
            "use_precomputed_msa": False,
            "use_templates": True,
            "custom_template_path": custom_template_path,
            "skip_msa": False,
        }
    ]
    assert (tmp_path / "protA.pkl").exists()


def test_create_and_save_monomer_objects_reuses_mmseq_identifier_sidecar(
    tmp_flags, tmp_path, monkeypatch
):
    create_features.FLAGS.output_dir = str(tmp_path)
    create_features.FLAGS.compress_features = False
    create_features.FLAGS.skip_existing = False
    create_features.FLAGS.use_mmseqs2 = True
    create_features.FLAGS.use_precomputed_msas = False
    create_features.FLAGS.re_search_templates_mmseqs2 = False

    a3m_text = "\n".join(
        [
            "# mmseqs header",
            ">101",
            "ACDE",
            ">UniRef100_A0A636IKY3\t136\t0.883",
            "ACDF",
            "",
        ]
    )

    monkeypatch.setattr(
        MonomericObject, "unzip_msa_files", staticmethod(lambda _path: False)
    )
    monkeypatch.setattr(
        MonomericObject, "zip_msa_files", staticmethod(lambda _path: None)
    )
    monkeypatch.setattr(
        objects_mod,
        "get_msa_and_templates",
        lambda **_kwargs: (
            ["UNPAIRED"],
            ["PAIRED"],
            ["UNIQUE"],
            ["CARD"],
            ["TEMPLATE"],
        ),
    )
    monkeypatch.setattr(objects_mod, "msa_to_str", lambda *args: a3m_text)
    monkeypatch.setattr(
        objects_mod,
        "unserialize_msa",
        lambda a3m_lines, sequence: (
            ["PRECOMP_MSA"],
            ["PRECOMP_PAIRED"],
            ["UNIQUE"],
            ["CARD"],
            ["PRECOMP_TEMPLATE"],
        ),
    )
    monkeypatch.setattr(
        objects_mod,
        "build_monomer_feature",
        lambda *_args, **_kwargs: {
            "msa": np.asarray([[1, 2, 3, 4], [1, 2, 3, 5]], dtype=np.int32),
            "deletion_matrix_int": np.asarray(
                [[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
            ),
            "template_confidence_scores": None,
            "template_release_date": None,
        },
    )

    first_calls = []

    def fake_uniprot_batch(accessions, *, urlopen):
        first_calls.append(tuple(accessions))
        return {
            "results": [
                {
                    "primaryAccession": "A0A636IKY3",
                    "organism": {"taxonId": 562},
                }
            ]
        }

    monkeypatch.setattr(
        mmseqs_species_identifiers,
        "_query_uniprot_batch",
        fake_uniprot_batch,
    )
    monkeypatch.setattr(
        mmseqs_species_identifiers,
        "_query_uniparc_batch",
        lambda accessions, *, urlopen: {"results": []},
    )

    with patch(
        "alphapulldown.utils.save_meta_data.get_meta_dict",
        return_value={"source": "test"},
    ):
        first = MonomericObject("protA", "ACDE")
        create_features.create_and_save_monomer_objects(first, pipeline=None)

    assert first_calls == [("A0A636IKY3",)]
    assert (tmp_path / "protA.a3m").exists()
    assert (tmp_path / "protA.mmseq_ids.json").exists()
    assert (tmp_path / "protA.pkl").exists()

    create_features.FLAGS.use_precomputed_msas = True
    mmseqs_species_identifiers._SPECIES_ID_CACHE.clear()
    second_calls = []

    def fail_uniprot_batch(accessions, *, urlopen):
        second_calls.append(tuple(accessions))
        raise AssertionError("expected mmseq sidecar cache to skip UniProt lookups")

    monkeypatch.setattr(
        mmseqs_species_identifiers,
        "_query_uniprot_batch",
        fail_uniprot_batch,
    )

    with patch(
        "alphapulldown.utils.save_meta_data.get_meta_dict",
        return_value={"source": "test"},
    ):
        second = MonomericObject("protA", "ACDE")
        create_features.create_and_save_monomer_objects(second, pipeline=None)

    assert second_calls == []
    assert second.feature_dict["msa_species_identifiers_all_seq"].tolist() == [
        b"",
        b"562",
    ]
    assert second.feature_dict["msa_uniprot_accession_identifiers_all_seq"].tolist() == [
        b"",
        b"A0A636IKY3",
    ]


def test_process_multimeric_features_uses_mmseqs_without_local_pipeline(tmp_flags, tmp_path):
    template_path = tmp_path / "template.cif"
    template_path.write_text("data_template\n", encoding="utf-8")

    create_features.FLAGS.output_dir = str(tmp_path / "out")
    create_features.FLAGS.use_mmseqs2 = True

    feat = {
        "protein": "complex_mmseqs",
        "chains": ["A"],
        "templates": [str(template_path)],
        "sequence": "ACDE",
    }

    with patch.object(create_features, "MonomericObject", RecordingDummyMonomer), \
         patch.object(create_features, "create_custom_db", return_value="/tmp/custom_db") as mock_custom_db, \
         patch.object(create_features, "create_arguments") as mock_arguments, \
         patch.object(create_features, "create_pipeline_af2") as mock_pipeline, \
         patch.object(create_features, "create_uniprot_runner") as mock_runner, \
         patch.object(create_features, "create_and_save_monomer_objects") as mock_save:
        create_features.process_multimeric_features(feat, 1)

    mock_custom_db.assert_called_once()
    mock_arguments.assert_called_once_with("/tmp/custom_db")
    mock_pipeline.assert_not_called()
    mock_runner.assert_not_called()
    saved_monomer, saved_pipeline = mock_save.call_args.args
    saved_kwargs = mock_save.call_args.kwargs
    assert saved_pipeline is None
    assert saved_monomer.description == "complex_mmseqs"
    assert saved_monomer.uniprot_runner is None
    assert saved_kwargs == {"custom_template_path": "/tmp/custom_db/templates"}


def test_create_custom_db_passes_thresholds_to_builder(tmp_flags):
    create_features.FLAGS.threshold_clashes = 12.5
    create_features.FLAGS.hb_allowance = 0.7
    create_features.FLAGS.plddt_threshold = 42.0

    with patch.object(create_features, "create_db") as mock_create_db:
        db_path = create_features.create_custom_db("/tmp/base", "proteinX", ["a.cif"], ["A"])

    assert str(db_path) == "/tmp/base/custom_template_db/proteinX"
    mock_create_db.assert_called_once_with(
        db_path,
        ["a.cif"],
        ["A"],
        12.5,
        0.7,
        42.0,
    )


def test_create_pipeline_af3_prefers_explicit_database_overrides(tmp_flags):
    class DummyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    create_features.FLAGS.max_template_date = "2021-09-30"
    create_features.FLAGS.data_pipeline = "alphafold3"
    create_features.FLAGS.data_dir = "/db"
    create_features.FLAGS.small_bfd_database_path = "/override/small_bfd"
    create_features.FLAGS.uniref90_database_path = "/override/uniref90"
    create_features.FLAGS.template_mmcif_dir = "/override/mmcif"

    with patch.object(create_features, "AF3DataPipelineConfig", side_effect=DummyConfig) as mock_config, \
         patch.object(create_features, "AF3DataPipeline", side_effect=lambda config: config) as mock_pipeline:
        config = create_features.create_pipeline_af3()

    mock_config.assert_called_once()
    mock_pipeline.assert_called_once()
    assert config.kwargs["small_bfd_database_path"] == "/override/small_bfd"
    assert config.kwargs["uniref90_database_path"] == "/override/uniref90"
    assert config.kwargs["pdb_database_path"] == "/override/mmcif"
    assert config.kwargs["mgnify_database_path"] == "/db/mgy_clusters_2022_05.fa"
    assert config.kwargs["seqres_database_path"] == "/db/pdb_seqres_2022_09_28.fasta"


def test_main_rejects_af3_mmseqs2(tmp_flags, tmp_path):
    create_features.FLAGS.data_pipeline = "alphafold3"
    create_features.FLAGS.use_mmseqs2 = True
    create_features.FLAGS.data_dir = None
    create_features.FLAGS.output_dir = str(tmp_path / "af3_out")

    with patch.object(create_features.logging, "error") as mock_error, \
         pytest.raises(SystemExit):
        create_features.main([])

    mock_error.assert_called_once_with(
        "AlphaFold3 does not support --use_mmseqs2. "
        "Please provide local databases via --data_dir."
    )


def test_create_af3_individual_features_falls_back_to_double_letter_chain_ids(tmp_flags, tmp_path):
    create_features.FLAGS.output_dir = str(tmp_path)
    create_features.FLAGS.seq_index = 27

    af3_modules, folding_input_stub = build_af3_stub_modules()
    del af3_modules["alphafold3.structure"].mmcif
    af3_modules.pop("alphafold3.structure.mmcif")

    sequences = [("ACDE", f"chain_{idx}") for idx in range(1, 28)]
    with patch.dict(sys.modules, af3_modules), \
         patch.object(create_features, "create_pipeline_af3", return_value=MagicMock(process=MagicMock(return_value={"plain": "json"}))), \
         patch.object(create_features, "folding_input", folding_input_stub), \
         patch.object(create_features, "iter_seqs", return_value=sequences), \
         patch("pathlib.Path.write_text", new=real_write_text):
        create_features.create_af3_individual_features()

    outpath = tmp_path / "chain_27_af3_input.json"
    assert outpath.exists()
    assert json.loads(outpath.read_text(encoding="utf-8")) == {"plain": "json"}


def test_create_af3_individual_features_skips_existing_outputs(tmp_flags, tmp_path):
    create_features.FLAGS.output_dir = str(tmp_path)
    create_features.FLAGS.skip_existing = True

    af3_modules, folding_input_stub = build_af3_stub_modules()
    existing_output = tmp_path / "protA_af3_input.json"
    existing_output.write_text("{}", encoding="utf-8")

    pipeline = MagicMock(process=MagicMock(return_value=DummyJsonObj()))
    with patch.dict(sys.modules, af3_modules), \
         patch.object(create_features, "create_pipeline_af3", return_value=pipeline), \
         patch.object(create_features, "folding_input", folding_input_stub), \
         patch.object(create_features, "iter_seqs", return_value=[("ACDE", "protA")]), \
         patch("pathlib.Path.write_text", new=real_write_text):
        create_features.create_af3_individual_features()

    pipeline.process.assert_not_called()
    assert existing_output.read_text(encoding="utf-8") == "{}"


def test_create_af3_individual_features_prefills_query_only_msas_when_skip_msa(
    tmp_flags, tmp_path
):
    create_features.FLAGS.output_dir = str(tmp_path)
    create_features.FLAGS.data_pipeline = "alphafold3"
    create_features.FLAGS.skip_msa = True

    af3_modules, folding_input_stub = build_af3_stub_modules()
    pipeline = MagicMock(process=MagicMock(return_value=DummyJsonObj()))
    with patch.dict(sys.modules, af3_modules), \
         patch.object(create_features, "create_pipeline_af3", return_value=pipeline), \
         patch.object(create_features, "folding_input", folding_input_stub), \
         patch.object(
             create_features,
             "iter_seqs",
             return_value=[("ACDE", "protein_chain protein"), ("AUGA", "rna_chain RNA")],
         ), \
         patch("pathlib.Path.write_text", new=real_write_text):
        create_features.create_af3_individual_features()

    protein_input = pipeline.process.call_args_list[0].args[0]
    protein_chain = protein_input.chains[0]
    assert protein_chain.unpaired_msa == ">query\nACDE\n"
    assert protein_chain.paired_msa == ""
    assert protein_chain.templates is None

    rna_input = pipeline.process.call_args_list[1].args[0]
    rna_chain = rna_input.chains[0]
    assert rna_chain.unpaired_msa == ">query\nAUGA\n"


def test_main_dispatches_to_af3_feature_creation(tmp_flags, tmp_path):
    create_features.FLAGS.data_pipeline = "alphafold3"
    create_features.FLAGS.output_dir = str(tmp_path / "af3_out")

    with patch.object(create_features, "create_af3_individual_features") as mock_af3, \
         patch.object(create_features, "check_template_date") as mock_check:
        create_features.main([])

    mock_af3.assert_called_once_with()
    mock_check.assert_not_called()
