import os
import shutil
import pickle
import tempfile
import subprocess
from pathlib import Path
import lzma
import json
import pytest
import logging

import numpy as np
from parameterized import parameterized

from alphapulldown.utils.remove_clashes_low_plddt import extract_seqs

logger = logging.getLogger(__name__)

class TestCreateIndividualFeaturesWithTemplates:

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup test environment and cleanup after each test"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.TEST_DATA_DIR = Path(self.temp_dir.name)
        
        # Copy test data files to the temporary directory
        original_test_data_dir = Path(__file__).parent / "test_data"
        shutil.copytree(original_test_data_dir, self.TEST_DATA_DIR, dirs_exist_ok=True)
        
        # Create necessary directories
        (self.TEST_DATA_DIR / 'features').mkdir(parents=True, exist_ok=True)
        (self.TEST_DATA_DIR / 'templates').mkdir(parents=True, exist_ok=True)
        
        # Ensure fastas directory exists and has the required files
        fastas_dir = self.TEST_DATA_DIR / 'fastas'
        fastas_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Test setup complete. Using temp directory: {self.TEST_DATA_DIR}")
        logger.info(f"FASTA files available: {list(fastas_dir.glob('*.fasta'))}")
        
        yield
        
        # Cleanup
        self.temp_dir.cleanup()
        logger.info("Test cleanup complete")

    def create_mock_file(self, file_path):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).touch(exist_ok=True)

    def mock_databases(self):
        root_dir = self.TEST_DATA_DIR
        logger.info("Creating mock database files")

        self.create_mock_file(root_dir / 'uniref90/uniref90.fasta')
        self.create_mock_file(root_dir / 'mgnify/mgy_clusters_2022_05.fa')
        self.create_mock_file(root_dir / 'uniprot/uniprot.fasta')
        with open(root_dir / 'uniprot' / 'uniprot.fasta', 'w') as f:
            f.write(">dummy_uniprot\nAAAAAAAAAAAAAAAAAAA\n")
        self.create_mock_file(root_dir / 'bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt_hhm.ffindex')
        self.create_mock_file(root_dir / 'uniref30/UniRef30_2021_03_hhm.ffindex')
        self.create_mock_file(root_dir / 'uniref30/UniRef30_2023_02_hhm.ffindex')
        self.create_mock_file(root_dir / 'pdb70/pdb70_hhm.ffindex')

        hhblits_root = root_dir / 'bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'
        hhblits_files = ['_a3m.ffdata', '_a3m.ffindex', '_cs219.ffdata', '_cs219.ffindex', '_hhmm.ffdata', '_hhmm.ffindex']
        for file in hhblits_files:
            self.create_mock_file(hhblits_root / file)

        uniclust_db_root = root_dir / 'uniclust30/uniclust30_2018_08/uniclust30_2018_08'
        uniclust_db_files = ['_a3m_db', '_a3m.ffdata', '_a3m.ffindex', '.cs219', '_cs219.ffdata', '_cs219.ffindex',
                             '_hhm_db', '_hhm.ffdata', '_hhm.ffindex']
        for suffix in uniclust_db_files:
            self.create_mock_file(f"{uniclust_db_root}{suffix}")

        uniref_db_root = root_dir / 'uniref30/UniRef30_2023_02'
        uniref_db_files = ['_a3m.ffdata', '_a3m.ffindex', '_hmm.ffdata', '_hmm.ffindex', '_cs.ffdata', '_cs.ffindex']
        for suffix in uniref_db_files:
            self.create_mock_file(f"{uniref_db_root}{suffix}")
        
        logger.info("Mock database files created successfully")

    def run_features_generation(self, file_name, chain_id, file_extension, use_mmseqs2, compress_features=False):
        (self.TEST_DATA_DIR / 'features').mkdir(parents=True, exist_ok=True)
        (self.TEST_DATA_DIR / 'templates').mkdir(parents=True, exist_ok=True)
        self.mock_databases()

        logger.info(f"Running features generation for {file_name}_{chain_id}.{file_extension}")

        with open(f"{self.TEST_DATA_DIR}/description.csv", 'w') as desc_file:
            desc_file.write(f">{file_name}_{chain_id}, {file_name}.{file_extension}, {chain_id}\n")

        fasta_path = f"{self.TEST_DATA_DIR}/fastas/{file_name}_{chain_id}.fasta"
        assert Path(fasta_path).exists(), f"FASTA file not found: {fasta_path}"
        
        # Remove .pkl, .pkl.xz, .json, .json.xz files from the features directory
        for ext in ['*.pkl', '*.pkl.xz', '*.json', '*.json.xz']:
            for f in self.TEST_DATA_DIR.glob(f'features/{ext}'):
                f.unlink()

        cmd = [
            'create_individual_features.py',
            '--use_precomputed_msas', 'True',
            '--save_msa_files', 'True',
            '--skip_existing', 'True',
            '--data_dir', f"{self.TEST_DATA_DIR}",
            '--max_template_date', '3021-01-01',
            '--threshold_clashes', '1000',
            '--hb_allowance', '0.4',
            '--plddt_threshold', '0',
            '--fasta_paths', fasta_path,
            '--path_to_mmt', f"{self.TEST_DATA_DIR}/templates",
            '--description_file', f"{self.TEST_DATA_DIR}/description.csv",
            '--output_dir', f"{self.TEST_DATA_DIR}/features",
        ]
        if use_mmseqs2:
            cmd.extend(['--use_mmseqs2', 'True'])
        if compress_features:
            cmd.extend(['--compress_features', 'True'])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Features generation completed successfully")

    def validate_generated_features(self, pkl_path, json_path, file_name, file_extension, chain_id, compress_features):
        logger.info(f"Validating generated features for {file_name}_{chain_id}")
        
        # Validate that the expected output files exist
        assert json_path.exists(), f"Metadata JSON file was not created: {json_path}"
        assert pkl_path.exists(), f"Pickle file was not created: {pkl_path}"
        logger.info(f"Output files verified: {pkl_path}, {json_path}")

        # Validate the contents of the PKL file
        if compress_features:
            with lzma.open(pkl_path, 'rb') as f:
                monomeric_object = pickle.load(f)
        else:
            with open(pkl_path, 'rb') as f:
                monomeric_object = pickle.load(f)

        assert hasattr(monomeric_object, 'feature_dict'), "Loaded object does not have 'feature_dict' attribute."
        features = monomeric_object.feature_dict

        # Validate that the expected sequences and atom coordinates are present and valid
        temp_sequence = features['template_sequence'][0].decode('utf-8')
        target_sequence = features['sequence'][0].decode('utf-8')
        atom_coords = features['template_all_atom_positions'][0]
        template_path = self.TEST_DATA_DIR / 'templates' / f'{file_name}.{file_extension}'
        
        # Check that template sequence is not empty
        assert len(temp_sequence) > 0, "Template sequence is empty"
        # Check that the atom coordinates are not all 0
        assert (atom_coords.any()) > 0, "All atom coordinates are zero"

        atom_seq, seqres_seq = extract_seqs(template_path, chain_id)
        
        logger.info(f"Target sequence: {target_sequence} (length: {len(target_sequence)})")
        logger.info(f"Template sequence: {temp_sequence} (length: {len(temp_sequence)})")
        if seqres_seq:
            logger.info(f"SEQRES sequence: {seqres_seq} (length: {len(seqres_seq)})")
        logger.info(f"Atom sequence: {atom_seq} (length: {len(atom_seq)})")
        
        # Check that atoms for not missing residues are not all 0
        residue_has_nonzero_coords = []
        for number, (s, a) in enumerate(zip(atom_seq, atom_coords)):
            # no coordinates for missing residues
            if s == 'X':
                assert np.all(a == 0), f"Missing residue {number} has non-zero coordinates"
                residue_has_nonzero_coords.append(False)
            else:
                non_zero = np.any(a != 0)
                residue_has_nonzero_coords.append(non_zero)
                if non_zero:
                    if seqres_seq:
                        seqres = seqres_seq[number]
                    else:
                        seqres = None
                    if seqres:
                        assert (s in seqres_seq), f"Residue {s} not found in SEQRES"
                    # first 4 coordinates are non zero
                    assert np.any(a[:4] != 0), f"First 4 coordinates are zero for residue {number}"
        
        logger.info(f"Feature validation completed successfully. {sum(residue_has_nonzero_coords)} residues have coordinates")

    @parameterized.expand([
        (True, '3L4Q', 'A', 'cif'),
        (False, '3L4Q', 'A', 'cif'),
    ])
    def test_compress_features_flag(self, compress_features, file_name, chain_id, file_extension):
        """Test feature compression functionality"""
        logger.info(f"Testing compress_features_flag: compress={compress_features}, file={file_name}_{chain_id}.{file_extension}")
        
        self.run_features_generation(file_name, chain_id, file_extension, use_mmseqs2=False, compress_features=compress_features)

        json_pattern = f'{file_name}_{chain_id}.{file_name}.{file_extension}.{chain_id}_feature_metadata_*.json'
        if compress_features:
            json_pattern += '.xz'
        metadata_files = list((self.TEST_DATA_DIR / 'features').glob(json_pattern))
        assert len(metadata_files) > 0, "Metadata JSON file was not created."
        json_path = metadata_files[0]

        pkl_filename = f'{file_name}_{chain_id}.{file_name}.{file_extension}.{chain_id}.pkl'
        if compress_features:
            pkl_filename += '.xz'
        pkl_path = self.TEST_DATA_DIR / 'features' / pkl_filename

        self.validate_generated_features(pkl_path, json_path, file_name, file_extension, chain_id, compress_features)

        # Clean up
        pkl_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        logger.info("Test cleanup completed")

    @parameterized.expand([
        ('3L4Q', 'A', 'cif', False),
        ('3L4Q', 'C', 'pdb', False),
        ('RANdom_name1_.7-1_0', 'B', 'pdb', False),
        ('RANdom_name1_.7-1_0', 'C', 'pdb', False),
        ('GAPPY_PDB', 'B', 'pdb', False),
        ('hetatoms', 'A', 'pdb', False),
    ])
    def test_run_features_generation(self, file_name, chain_id, file_extension, use_mmseqs2):
        """Test feature generation for various template types"""
        logger.info(f"Testing features generation: {file_name}_{chain_id}.{file_extension}, mmseqs2={use_mmseqs2}")
        
        self.run_features_generation(file_name, chain_id, file_extension, use_mmseqs2)

        # Determine the output paths for validation
        pkl_filename = f'{file_name}_{chain_id}.{file_name}.{file_extension}.{chain_id}.pkl'
        pkl_path = self.TEST_DATA_DIR / 'features' / pkl_filename
        json_pattern = f'{file_name}_{chain_id}.{file_name}.{file_extension}.{chain_id}_feature_metadata_*.json'
        metadata_files = list((self.TEST_DATA_DIR / 'features').glob(json_pattern))
        assert len(metadata_files) > 0, "Metadata JSON file was not created."
        json_path = metadata_files[0]

        self.validate_generated_features(pkl_path, json_path, file_name, file_extension, chain_id, compress_features=False)

        # Clean up
        pkl_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        logger.info("Test cleanup completed")

    @pytest.mark.skip(reason="use_mmseqs2 must not be set when running with --path_to_mmts")
    def test_6a_mmseqs2(self):
        """Test mmseqs2 functionality (currently skipped)"""
        logger.info("Skipping mmseqs2 test as it's not compatible with --path_to_mmts")
        self.run_features_generation('3L4Q', 'A', 'cif', True)
