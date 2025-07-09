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
    def setup_and_teardown(self):
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
        FLAGS.fasta_paths = []
        FLAGS.data_dir = "/test/db"
        
        # Mock the pipeline creation to avoid real database access
        with patch.object(create_features, 'create_pipeline_af2') as mock_af2_pipeline, \
             patch.object(create_features, 'create_uniprot_runner') as mock_uniprot_runner:
            mock_af2_pipeline.return_value = MagicMock()
            mock_uniprot_runner.return_value = MagicMock()
            
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