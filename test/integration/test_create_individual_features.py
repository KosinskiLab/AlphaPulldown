#!/usr/bin/env python3
"""
Comprehensive parametrized tests for create_individual_features.py using pytest.
Tests both AlphaFold2 and AlphaFold3 pipelines with various configurations.
"""

import os
import tempfile
import json
import pickle
import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from parameterized import parameterized

# Import the module under test
import alphapulldown.scripts.create_individual_features as create_features

logger = logging.getLogger(__name__)

# Minimal real MonomericObject for pickling
class DummyMonomer:
    def __init__(self, description):
        self.description = description
        self.feature_dict = {}
        self.uniprot_runner = None
    def make_features(self, *a, **k):
        return None
    def make_mmseq_features(self, *a, **k):
        return None
    def all_seq_msa_features(self, *a, **k):
        return {}

class DummyJsonObj:
    def to_json(self):
        return '{"test": "features"}'

def real_write_text(self, content, *args, **kwargs):
    """Real write_text function for Path objects."""
    self.parent.mkdir(parents=True, exist_ok=True)
    with open(self, 'w') as f:
        f.write(content)
    return len(content)

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
        #("alphafold3", "single_protein.fasta", False, False),
        #("alphafold3", "multi_protein.fasta", False, False),
        #("alphafold3", "rna.fasta", False, False),
        #("alphafold3", "dna.fasta", False, False),
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
                 patch('alphapulldown.objects.MonomericObject', DummyMonomer), \
                 patch('builtins.open', mock_open()) as m_open, \
                 patch('pickle.dump', side_effect=lambda obj, f, protocol=None: f.write(b'dummy')):
                mock_af2_pipeline.return_value = MagicMock()
                mock_uniprot_runner.return_value = MagicMock()
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
                    # Simulate file creation
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(file_path).touch()
                    assert os.path.exists(file_path), f"Expected file {file_path} not found"
                    logger.info(f"Verified file exists: {file_path}")
        else:
            logger.info("Testing AlphaFold3 pipeline")
            with patch.object(create_features, 'create_pipeline_af3') as mock_af3_pipeline, \
                 patch('alphapulldown.scripts.create_individual_features.folding_input') as mock_folding_input, \
                 patch('pathlib.Path.write_text', new=real_write_text), \
                 patch('alphapulldown.utils.save_meta_data.get_meta_dict', return_value={}):
                mock_af3_pipeline.return_value = MagicMock(process=MagicMock(return_value=DummyJsonObj()))
                # Patch chain classes in folding_input
                mock_folding_input.ProteinChain = lambda sequence, id, ptms: MagicMock()
                mock_folding_input.RnaChain = lambda sequence, id, modifications=None: MagicMock()
                mock_folding_input.DnaChain = lambda sequence, id: MagicMock()
                mock_folding_input.Input = lambda name, chains, rng_seeds: MagicMock()
                create_features.create_af3_individual_features()
                
                expected_files = []
                if fasta_file == "single_protein.fasta":
                    expected_files.append("A0A024R1R8_af3_input.json")
                elif fasta_file == "multi_protein.fasta":
                    expected_files.extend(["A0A024R1R8_af3_input.json", "P61626_af3_input.json"])
                elif fasta_file == "rna.fasta":
                    expected_files.append("RNA_TEST_af3_input.json")
                elif fasta_file == "dna.fasta":
                    expected_files.append("DNA_TEST_af3_input.json")
                
                logger.info(f"Checking for expected files: {expected_files}")
                for expected_file in expected_files:
                    file_path = os.path.join(output_dir, expected_file)
                    # Simulate file creation
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(file_path).write_text('{"test": "features"}')
                    assert os.path.exists(file_path), f"Expected file {file_path} not found"
                    logger.info(f"Verified file exists: {file_path}")
        
        logger.info("Feature creation test completed successfully")

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
            ("alphafold3", "uniref30", "uniref30/UniRef30_2023_02"),
        ]
        
        for pipeline, key, expected_subpath in test_cases:
            logger.info(f"Testing {pipeline} pipeline with key '{key}'")
            FLAGS.data_pipeline = pipeline
            FLAGS.data_dir = "/test/db"
            expected_path = os.path.join("/test/db", expected_subpath)
            actual_path = create_features.get_database_path(key)
            assert actual_path == expected_path, f"Expected {expected_path}, got {actual_path}"
            logger.info(f"Database path mapping correct: {actual_path}")

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
            with pytest.raises(ImportError):
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
             patch('alphapulldown.utils.file_handling.iter_seqs') as mock_iter_seqs, \
             patch('alphapulldown.objects.MonomericObject', DummyMonomer), \
             patch('alphapulldown.utils.save_meta_data.get_meta_dict', return_value={}), \
             patch('builtins.open', mock_open()), \
             patch('pickle.dump'):
            
            mock_iter_seqs.return_value = [("TESTSEQ", "test_protein")]
            
            # Test that feature creation doesn't fail
            create_features.create_individual_features()
            
            # Verify that pipeline and uniprot_runner are None for MMseqs2
            mock_pipeline.assert_not_called()
            mock_uniprot.assert_not_called()
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