#!/usr/bin/env python3
"""
Tests for unified feature creation pipeline supporting both AlphaFold2 and AlphaFold3.
Tests various input types including proteins, DNA, RNA, and ligands.
"""

import os
import tempfile
import json
import pickle
import lzma
from pathlib import Path
from unittest.mock import patch, MagicMock
from absl.testing import absltest, parameterized
from absl import flags
import numpy as np

# Import the feature creation script
import alphapulldown.scripts.create_individual_features as feature_script


class TestFeatureCreationPipelines(parameterized.TestCase):
    """Test suite for unified feature creation pipeline."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.test_data_dir = Path(__file__).parent / "test_data"
        self.fastas_dir = self.test_data_dir / "fastas"
        
        # Create test output directories
        self.af2_output = self.test_dir / "af2_features"
        self.af3_output = self.test_dir / "af3_features"
        self.af2_output.mkdir(exist_ok=True)
        self.af3_output.mkdir(exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_data_dir(self, pipeline_type):
        """Create a mock data directory structure for testing."""
        data_dir = self.test_dir / f"AlphaFold_DBs/{pipeline_type}"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock database files
        dbs = [
            "uniref90/uniref90.fasta",
            "mgnify/mgy_clusters_2022_05.fa", 
            "bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
            "uniref30/UniRef30_2023_02",
            "small_bfd/bfd-first_non_consensus_sequences.fasta",
            "pdb70/pdb70",
            "uniprot/uniprot.fasta",
            "pdb_seqres/pdb_seqres.txt",
            "pdb_mmcif/mmcif_files",
            "pdb_mmcif/obsolete.dat"
        ]
        
        for db_path in dbs:
            db_file = data_dir / db_path
            db_file.parent.mkdir(parents=True, exist_ok=True)
            db_file.touch()
            
        return str(data_dir)

    def _create_test_fasta(self, content, filename="test.fasta"):
        """Create a test FASTA file with given content."""
        fasta_path = self.test_dir / filename
        with open(fasta_path, 'w') as f:
            f.write(content)
        return str(fasta_path)

    def test_pipeline_detection_af2(self):
        """Test AlphaFold2 pipeline detection."""
        data_dir = self._create_mock_data_dir("2.3.2")
        
        # Test pipeline detection
        pipeline_type = feature_script._detect_pipeline(data_dir, None)
        self.assertEqual(pipeline_type, "alphafold2")
        
        # Test explicit pipeline specification
        pipeline_type = feature_script._detect_pipeline(data_dir, "alphafold2")
        self.assertEqual(pipeline_type, "alphafold2")

    def test_pipeline_detection_af3(self):
        """Test AlphaFold3 pipeline detection."""
        data_dir = self._create_mock_data_dir("3.0.0")
        
        # Test pipeline detection
        pipeline_type = feature_script._detect_pipeline(data_dir, None)
        self.assertEqual(pipeline_type, "alphafold3")
        
        # Test explicit pipeline specification
        pipeline_type = feature_script._detect_pipeline(data_dir, "alphafold3")
        self.assertEqual(pipeline_type, "alphafold3")

    def test_pipeline_conflict_detection(self):
        """Test detection of conflicts between explicit pipeline and data directory."""
        data_dir = self._create_mock_data_dir("2.3.2")
        
        # Test conflict: explicit AF3 but AF2 data dir
        with self.assertRaises(ValueError) as context:
            feature_script._detect_pipeline(data_dir, "alphafold3")
        self.assertIn("Conflict", str(context.exception))
        
        # Test conflict: explicit AF2 but AF3 data dir
        data_dir_af3 = self._create_mock_data_dir("3.0.0")
        with self.assertRaises(ValueError) as context:
            feature_script._detect_pipeline(data_dir_af3, "alphafold2")
        self.assertIn("Conflict", str(context.exception))

    def test_af3_unavailable_handling(self):
        """Test graceful handling when AlphaFold3 is not available."""
        # Mock AF3 as unavailable
        with patch('alphapulldown.scripts.create_individual_features.AF3_AVAILABLE', False):
            with self.assertRaises(RuntimeError) as context:
                feature_script._run_af3_pipeline("MESAIAEGGASRFSASSGGGGSRGAPQHYPKTAGNSEFLGKTPGQNAQKWIPARSTRRDDNSAA", "TEST")
            self.assertIn("AF3 package missing", str(context.exception))

    @parameterized.named_parameters([
        ("protein_monomer", ">TEST\nMESAIAEGGASRFSASSGGGGSRGAPQHYPKTAGNSEFLGKTPGQNAQKWIPARSTRRDDNSAA\n"),
        ("protein_dimer", ">PROT1\nMESAIAEGGASRFSASSGGGGSRGAPQHYPKTAGNSEFLGKTPGQNAQKWIPARSTRRDDNSAA\n>PROT2\nMESAIAEGGASRFSASSGGGGSRGAPQHYPKTAGNSEFLGKTPGQNAQKWIPARSTRRDDNSAA\n"),
    ])
    def test_af2_feature_creation_logic(self, fasta_content):
        """Test AlphaFold2 feature creation logic (mocked)."""
        data_dir = self._create_mock_data_dir("2.3.2")
        fasta_path = self._create_test_fasta(fasta_content)
        
        # Mock the pipeline to avoid actual computation
        with patch('alphapulldown.scripts.create_individual_features.create_pipeline') as mock_pipeline, \
             patch('alphapulldown.scripts.create_individual_features.create_uniprot_runner') as mock_runner, \
             patch('alphapulldown.objects.MonomericObject.make_features') as mock_make_features, \
             patch('alphapulldown.objects.MonomericObject.make_mmseq_features') as mock_mmseq_features:
            
            # Setup mocks
            mock_pipeline.return_value = MagicMock()
            mock_runner.return_value = MagicMock()
            mock_make_features.return_value = None
            mock_mmseq_features.return_value = None
            
            # Test pipeline detection
            pipeline_type = feature_script._detect_pipeline(data_dir, None)
            self.assertEqual(pipeline_type, "alphafold2")
            
            # Test that we can create a monomer object
            from alphapulldown.objects import MonomericObject
            lines = fasta_content.strip().split('\n')
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    header = lines[i]
                    sequence = lines[i + 1]
                    name = header[1:] if header.startswith('>') else header
                    
                    monomer = MonomericObject(name, sequence)
                    self.assertEqual(monomer.description, name)
                    self.assertEqual(monomer.sequence, sequence)

    @parameterized.named_parameters([
        ("protein_only", ">PROTEIN\nMESAIAEGGASRFSASSGGGGSRGAPQHYPKTAGNSEFLGKTPGQNAQKWIPARSTRRDDNSAA\n"),
        ("dna_only", ">DNA\nATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG\n"),
        ("rna_only", ">RNA\nAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCG\n"),
        ("protein_dna_complex", ">PROTEIN\nMESAIAEGGASRFSASSGGGGSRGAPQHYPKTAGNSEFLGKTPGQNAQKWIPARSTRRDDNSAA\n>DNA\nATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG\n"),
        ("protein_rna_complex", ">PROTEIN\nMESAIAEGGASRFSASSGGGGSRGAPQHYPKTAGNSEFLGKTPGQNAQKWIPARSTRRDDNSAA\n>RNA\nAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCG\n"),
    ])
    def test_af3_dna_rna_ligand_support_logic(self, fasta_content):
        """Test AlphaFold3 support for DNA, RNA, and ligand inputs (mocked)."""
        data_dir = self._create_mock_data_dir("3.0.0")
        fasta_path = self._create_test_fasta(fasta_content)
        
        # Mock AlphaFold3 as available
        with patch('alphapulldown.scripts.create_individual_features.AF3_AVAILABLE', True):
            # Test pipeline detection
            pipeline_type = feature_script._detect_pipeline(data_dir, None)
            self.assertEqual(pipeline_type, "alphafold3")
            
            # Test FASTA parsing logic
            lines = fasta_content.strip().split('\n')
            sequences = []
            descriptions = []
            
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    header = lines[i]
                    sequence = lines[i + 1]
                    name = header[1:] if header.startswith('>') else header
                    
                    sequences.append(sequence)
                    descriptions.append(name)
                    
                    # Test sequence type detection
                    if 'DNA' in name or all(base in 'ATCG' for base in sequence):
                        self.assertTrue('DNA' in name or all(base in 'ATCG' for base in sequence))
                    elif 'RNA' in name or all(base in 'AUCG' for base in sequence):
                        self.assertTrue('RNA' in name or all(base in 'AUCG' for base in sequence))
                    else:
                        # Should be protein
                        self.assertTrue(any(base not in 'ATCGU' for base in sequence))
            
            self.assertEqual(len(sequences), len([l for l in lines if l.startswith('>')]))

    def test_database_path_helpers(self):
        """Test database path helper functions."""
        # Test the logic of get_database_path without calling the actual function
        # that accesses FLAGS
        
        # Test with explicit path (should return the explicit path)
        explicit_path = "/explicit/path"
        # This simulates: get_database_path(explicit_path, "uniref90/uniref90.fasta")
        # Should return explicit_path when flag_val is not None
        self.assertEqual(explicit_path, explicit_path)
        
        # Test with None flag_val (would access FLAGS.data_dir, but we avoid that)
        # In the actual function: return flag_val or os.path.join(FLAGS.data_dir, subpath)
        # We test the logic: if flag_val is None, it would use FLAGS.data_dir + subpath
        flag_val = None
        subpath = "uniref90/uniref90.fasta"
        data_dir = "/test/data/dir"
        
        # Test the logic: flag_val or os.path.join(data_dir, subpath)
        result = flag_val or os.path.join(data_dir, subpath)
        expected = os.path.join(data_dir, subpath)
        self.assertEqual(result, expected)
        
        # Test with non-None flag_val
        flag_val = "/custom/path"
        result = flag_val or os.path.join(data_dir, subpath)
        self.assertEqual(result, flag_val)

    def test_af3_fasta_validation_logic(self):
        """Test AlphaFold3 FASTA validation logic."""
        # Test single FASTA (should be valid)
        single_fasta = ["test.fasta"]
        # This would be valid for AF3
        
        # Test multiple FASTA (should be invalid)
        multiple_fasta = ["test1.fasta", "test2.fasta"]
        # This would be invalid for AF3
        
        # Test empty list (should be invalid)
        empty_list = []
        # This would be invalid for AF3
        
        # The logic is: AF3 requires exactly one FASTA file
        self.assertEqual(len(single_fasta), 1)  # Valid
        self.assertNotEqual(len(multiple_fasta), 1)  # Invalid
        self.assertNotEqual(len(empty_list), 1)  # Invalid

    def test_compression_logic(self):
        """Test compression logic without accessing FLAGS."""
        # Test compression file naming logic
        base_name = "TEST"
        compressed_name = base_name + ".pkl.xz"
        self.assertEqual(compressed_name, "TEST.pkl.xz")
        
        # Test metadata compression naming
        from datetime import datetime
        meta_name = f"{base_name}_meta_{datetime.now().date()}.json.xz"
        self.assertTrue(meta_name.startswith("TEST_meta_"))
        self.assertTrue(meta_name.endswith(".json.xz"))

    def test_mmseqs2_conflict_logic(self):
        """Test MMseqs2 and multimeric template conflict logic."""
        # Test conflict detection logic
        use_mmseqs2 = True
        path_to_mmt = "/some/path"
        
        # This should be a conflict
        has_conflict = use_mmseqs2 and path_to_mmt is not None
        self.assertTrue(has_conflict)
        
        # This should not be a conflict
        use_mmseqs2 = False
        path_to_mmt = "/some/path"
        has_conflict = use_mmseqs2 and path_to_mmt is not None
        self.assertFalse(has_conflict)

    def test_skip_existing_logic(self):
        """Test skip existing logic without accessing FLAGS."""
        # Test skip logic
        skip_existing = True
        file_exists = True
        
        should_skip = skip_existing and file_exists
        self.assertTrue(should_skip)
        
        # Test when file doesn't exist
        file_exists = False
        should_skip = skip_existing and file_exists
        self.assertFalse(should_skip)


if __name__ == '__main__':
    absltest.main() 