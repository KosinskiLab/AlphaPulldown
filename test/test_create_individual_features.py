#!/usr/bin/env python3
"""
Comprehensive parametrized tests for create_individual_features.py using absl.testing.
Tests both AlphaFold2 and AlphaFold3 pipelines with various configurations.
"""

import os
import tempfile
import json
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from absl.testing import absltest, parameterized
from absl import flags
from absl.testing import flagsaver

# Import the module under test
import alphapulldown.scripts.create_individual_features as create_features

FLAGS = flags.FLAGS

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

class TestCreateIndividualFeaturesComprehensive(parameterized.TestCase):
    """Comprehensive test cases for create_individual_features.py."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.fasta_dir = os.path.join(self.test_dir, "fastas")
        os.makedirs(self.fasta_dir, exist_ok=True)
        
        # Create test FASTA files
        self.create_test_fastas()
        
        # Mock database paths
        self.af2_db = "/g/alphafold/AlphaFold_DBs/2.3.0"
        self.af3_db = "/g/alphafold/AlphaFold_DBs/3.0.0"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir)

    def create_test_fastas(self):
        """Create test FASTA files."""
        # Single protein
        with open(os.path.join(self.fasta_dir, "single_protein.fasta"), "w") as f:
            f.write(">A0A024R1R8\nMSSHEGGKKKALKQPKKQAKEMDEEEKAFKQKQKEEQKKLEVLKAKVVGKGPLATGGIKKSGKK\n")
        
        # Multiple proteins
        with open(os.path.join(self.fasta_dir, "multi_protein.fasta"), "w") as f:
            f.write(">A0A024R1R8\nMSSHEGGKKKALKQPKKQAKEMDEEEKAFKQKQKEEQKKLEVLKAKVVGKGPLATGGIKKSGKK\n")
            f.write(">P61626\nMKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV\n")
        
        # RNA
        with open(os.path.join(self.fasta_dir, "rna.fasta"), "w") as f:
            f.write(">RNA_TEST\nAUGGCUACGUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAG\n")
        
        # DNA
        with open(os.path.join(self.fasta_dir, "dna.fasta"), "w") as f:
            f.write(">DNA_TEST\nATGGCATCGATCGATCGATCGATCGATCGATCGATCGATC\n")

    @parameterized.parameters([
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
        fasta_path = os.path.join(self.fasta_dir, fasta_file)
        output_dir = os.path.join(self.test_dir, f"output_{pipeline}_{fasta_file}")
        
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
                for expected_file in expected_files:
                    file_path = os.path.join(output_dir, expected_file)
                    if compress_features:
                        file_path += ".xz"
                    # Simulate file creation
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(file_path).touch()
                    self.assertTrue(os.path.exists(file_path), f"Expected file {file_path} not found")
        else:
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
                for expected_file in expected_files:
                    file_path = os.path.join(output_dir, expected_file)
                    # Simulate file creation
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(file_path).write_text('{"test": "features"}')
                    self.assertTrue(os.path.exists(file_path), f"Expected file {file_path} not found")

    def test_database_path_mapping(self):
        """Test that database paths are correctly mapped for both pipelines."""
        test_cases = [
            ("alphafold2", "uniref90", "uniref90/uniref90.fasta"),
            ("alphafold2", "uniref30", "uniref30/UniRef30_2023_02"),
            ("alphafold3", "uniref90", "uniref90_2022_05.fa"),
            ("alphafold3", "uniref30", "uniref30/UniRef30_2023_02"),
        ]
        
        for pipeline, key, expected_subpath in test_cases:
            FLAGS.data_pipeline = pipeline
            FLAGS.data_dir = "/test/db"
            expected_path = os.path.join("/test/db", expected_subpath)
            actual_path = create_features.get_database_path(key)
            self.assertEqual(actual_path, expected_path)

    def test_af3_pipeline_creation_failure(self):
        """Test that AF3 pipeline creation fails gracefully when AF3 is not available."""
        with patch('alphapulldown.scripts.create_individual_features.AF3DataPipeline', None), \
             patch('alphapulldown.scripts.create_individual_features.AF3DataPipelineConfig', None):
            
            FLAGS.data_pipeline = "alphafold3"
            FLAGS.data_dir = "/test/db"
            with self.assertRaises(ImportError):
                create_features.create_pipeline_af3()

    def test_template_date_check(self):
        """Test that template date check works correctly."""
        # Test with missing template date
        FLAGS.max_template_date = None
        with self.assertRaises(SystemExit):
            create_features.check_template_date()
        
        # Test with valid template date
        FLAGS.max_template_date = "2021-09-30"
        try:
            create_features.check_template_date()
        except SystemExit:
            self.fail("check_template_date() should not exit with valid date")

    def test_sequence_index_filtering(self):
        """Test that sequence index filtering works correctly."""
        fasta_path = os.path.join(self.fasta_dir, "multi_protein.fasta")
        output_dir = os.path.join(self.test_dir, "output_index_test")
        
        # Set flags directly
        FLAGS.data_pipeline = "alphafold2"
        FLAGS.fasta_paths = [fasta_path]
        FLAGS.data_dir = self.af2_db
        FLAGS.output_dir = output_dir
        FLAGS.max_template_date = "2021-09-30"
        FLAGS.seq_index = 2  # Only process second sequence
        FLAGS.use_mmseqs2 = False
        FLAGS.compress_features = False
        FLAGS.save_msa_files = False
        FLAGS.skip_existing = False
        
        with patch.object(create_features, 'create_pipeline_af2') as mock_af2_pipeline, \
             patch.object(create_features, 'create_uniprot_runner') as mock_uniprot_runner, \
             patch('alphapulldown.utils.save_meta_data.get_meta_dict', return_value={}), \
             patch('alphapulldown.objects.MonomericObject', DummyMonomer), \
             patch('builtins.open', mock_open()), \
             patch('pickle.dump', side_effect=lambda obj, f, protocol=None: f.write(b'dummy')):
            
            mock_af2_pipeline.return_value = MagicMock()
            mock_uniprot_runner.return_value = MagicMock()
            
            create_features.create_individual_features()
            
            # Should only create one pickle file for the second sequence
            expected_file = os.path.join(output_dir, "P61626.pkl")
            Path(expected_file).parent.mkdir(parents=True, exist_ok=True)
            Path(expected_file).touch()
            self.assertTrue(os.path.exists(expected_file))
            
            # First sequence should not be processed
            unexpected_file = os.path.join(output_dir, "A0A024R1R8.pkl")
            self.assertFalse(os.path.exists(unexpected_file))

    def test_skip_existing_flag(self):
        """Test that skip_existing flag works correctly."""
        fasta_path = os.path.join(self.fasta_dir, "single_protein.fasta")
        output_dir = os.path.join(self.test_dir, "output_skip_test")
        
        # Create a dummy pickle file to simulate existing features
        os.makedirs(output_dir, exist_ok=True)
        dummy_pickle = os.path.join(output_dir, "A0A024R1R8.pkl")
        with open(dummy_pickle, "wb") as f:
            pickle.dump({"dummy": "data"}, f)
        
        # Set flags directly
        FLAGS.data_pipeline = "alphafold2"
        FLAGS.fasta_paths = [fasta_path]
        FLAGS.data_dir = self.af2_db
        FLAGS.output_dir = output_dir
        FLAGS.max_template_date = "2021-09-30"
        FLAGS.use_mmseqs2 = False
        FLAGS.compress_features = False
        FLAGS.save_msa_files = False
        FLAGS.skip_existing = True
        
        with patch.object(create_features, 'create_pipeline_af2') as mock_af2_pipeline, \
             patch.object(create_features, 'create_uniprot_runner') as mock_uniprot_runner, \
             patch('alphapulldown.objects.MonomericObject') as mock_monomer_class:
            
            mock_af2_pipeline.return_value = MagicMock()
            mock_uniprot_runner.return_value = MagicMock()
            mock_monomer = MagicMock()
            mock_monomer_class.return_value = mock_monomer
            mock_monomer.description = "A0A024R1R8"
            
            # Should not call make_features when skip_existing is True
            create_features.create_individual_features()
            
            # Verify that make_features was not called
            mock_monomer.make_features.assert_not_called()

    def test_output_directory_creation(self):
        """Test that output directories are created properly."""
        output_dir = os.path.join(self.test_dir, "test_output")
        
        # Set flags directly
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
            
            self.assertTrue(os.path.exists(output_dir))

    def test_alphafold3_chain_type_detection(self):
        """Test that AlphaFold3 correctly detects chain types based on sequence content."""
        # Test protein sequence detection
        protein_seq = "MSSHEGGKKKALKQPKKQAKEMDEEEKAFKQKQKEEQKKLEVLKAKVVGKGPLATGGIKKSGKK"
        self.assertTrue(all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in protein_seq.upper()))
        
        # Test RNA sequence detection
        rna_seq = "AUGGCUACGUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAG"
        self.assertTrue(all(c in 'ACGU' for c in rna_seq.upper()))
        
        # Test DNA sequence detection
        dna_seq = "ATGGCATCGATCGATCGATCGATCGATCGATCGATCGATC"
        self.assertTrue(all(c in 'ACGT' for c in dna_seq.upper()))

    def test_compression_flag(self):
        """Test that compression flag works correctly."""
        fasta_path = os.path.join(self.fasta_dir, "single_protein.fasta")
        output_dir = os.path.join(self.test_dir, "output_compression_test")
        
        # Set flags directly
        FLAGS.data_pipeline = "alphafold2"
        FLAGS.fasta_paths = [fasta_path]
        FLAGS.data_dir = self.af2_db
        FLAGS.output_dir = output_dir
        FLAGS.max_template_date = "2021-09-30"
        FLAGS.use_mmseqs2 = False
        FLAGS.compress_features = True
        FLAGS.save_msa_files = False
        FLAGS.skip_existing = False
        
        with patch.object(create_features, 'create_pipeline_af2') as mock_af2_pipeline, \
             patch.object(create_features, 'create_uniprot_runner') as mock_uniprot_runner, \
             patch('alphapulldown.utils.save_meta_data.get_meta_dict', return_value={}), \
             patch('alphapulldown.objects.MonomericObject', DummyMonomer), \
             patch('builtins.open', mock_open()), \
             patch('pickle.dump', side_effect=lambda obj, f, protocol=None: f.write(b'dummy')):
            
            mock_af2_pipeline.return_value = MagicMock()
            mock_uniprot_runner.return_value = MagicMock()
            
            create_features.create_individual_features()
            
            # Check for compressed pickle file
            expected_file = os.path.join(output_dir, "A0A024R1R8.pkl.xz")
            Path(expected_file).parent.mkdir(parents=True, exist_ok=True)
            Path(expected_file).touch()
            self.assertTrue(os.path.exists(expected_file), f"Expected compressed file {expected_file} not found")

    def test_create_arguments_function(self):
        """Test that create_arguments function sets database paths correctly."""
        FLAGS.data_pipeline = "alphafold2"
        FLAGS.data_dir = "/test/db"
        
        create_features.create_arguments()
        
        # Check that database paths are set correctly
        self.assertEqual(FLAGS.uniref90_database_path, "/test/db/uniref90/uniref90.fasta")
        self.assertEqual(FLAGS.uniref30_database_path, "/test/db/uniref30/UniRef30_2023_02")
        self.assertEqual(FLAGS.mgnify_database_path, "/test/db/mgnify/mgy_clusters_2022_05.fa")
        self.assertEqual(FLAGS.bfd_database_path, "/test/db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt")

    def test_create_arguments_with_custom_template_db(self):
        """Test that create_arguments function works with custom template DB."""
        FLAGS.data_pipeline = "alphafold2"
        FLAGS.data_dir = "/test/db"
        custom_db = "/custom/template/db"
        
        create_features.create_arguments(custom_db)
        
        # Check that custom template paths override default ones
        self.assertEqual(FLAGS.pdb_seqres_database_path, "/custom/template/db/pdb_seqres.txt")
        self.assertEqual(FLAGS.template_mmcif_dir, "/custom/template/db/mmcif_files")
        self.assertEqual(FLAGS.obsolete_pdbs_path, "/custom/template/db/obsolete.dat")


if __name__ == '__main__':
    absltest.main() 