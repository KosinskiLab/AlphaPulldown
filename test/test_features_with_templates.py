import os
import sys
import shutil
import pickle
import tempfile
import subprocess
from pathlib import Path
import lzma
import json

import numpy as np
from absl.testing import absltest, parameterized

from alphapulldown.utils.remove_clashes_low_plddt import extract_seqs


class TestCreateIndividualFeaturesWithTemplates(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.TEST_DATA_DIR = Path(self.temp_dir.name)
        # copy test data into temp area
        original_test_data_dir = Path(__file__).parent / "test_data"
        shutil.copytree(original_test_data_dir, self.TEST_DATA_DIR, dirs_exist_ok=True)
        # ensure required dirs exist
        (self.TEST_DATA_DIR / 'features').mkdir(parents=True, exist_ok=True)
        (self.TEST_DATA_DIR / 'templates').mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_mock_file(self, file_path: Path):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).touch(exist_ok=True)

    def mock_databases(self):
        root_dir = self.TEST_DATA_DIR

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

    def run_features_generation(self, file_name, chain_id, file_extension, use_mmseqs2, compress_features=False):
        (self.TEST_DATA_DIR / 'features').mkdir(parents=True, exist_ok=True)
        (self.TEST_DATA_DIR / 'templates').mkdir(parents=True, exist_ok=True)
        self.mock_databases()

        with open(self.TEST_DATA_DIR / "description.csv", 'w') as desc_file:
            desc_file.write(f">{file_name}_{chain_id}, {file_name}.{file_extension}, {chain_id}\n")

        assert (self.TEST_DATA_DIR / f"fastas/{file_name}_{chain_id}.fasta").exists()

        # clean previous outputs
        for ext in ('*.pkl', '*.pkl.xz', '*.json', '*.json.xz'):
            for f in (self.TEST_DATA_DIR / 'features').glob(ext):
                f.unlink()

        # run module directly (avoid console-script version pin)
        cmd = [
            sys.executable, "-m", "alphapulldown.scripts.create_individual_features",
            "--use_precomputed_msas", "True",
            "--save_msa_files", "True",
            "--skip_existing", "True",
            "--max_template_date", "3021-01-01",
            "--threshold_clashes", "1000",
            "--hb_allowance", "0.4",
            "--plddt_threshold", "0",
            "--fasta_paths", str(self.TEST_DATA_DIR / f"fastas/{file_name}_{chain_id}.fasta"),
            "--path_to_mmt", str(self.TEST_DATA_DIR / "templates"),
            "--description_file", str(self.TEST_DATA_DIR / "description.csv"),
            "--output_dir", str(self.TEST_DATA_DIR / "features"),
        ]
        # only pass data_dir when not using mmseqs2 (to match the test name/intent)
        if not use_mmseqs2:
            cmd += ["--data_dir", str(self.TEST_DATA_DIR)]
        else:
            cmd += ["--use_mmseqs2", "True"]
        if compress_features:
            cmd += ["--compress_features", "True"]

        # capture output for easier debugging
        res = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            # surface helpful context if the module errors
            raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)

    def validate_generated_features(self, pkl_path, json_path, file_name, file_extension, chain_id, compress_features):
        self.assertTrue(json_path.exists(), f"Metadata JSON file was not created: {json_path}")
        self.assertTrue(pkl_path.exists(), f"Pickle file was not created: {pkl_path}")

        if compress_features:
            with lzma.open(pkl_path, 'rb') as f:
                monomeric_object = pickle.load(f)
        else:
            with open(pkl_path, 'rb') as f:
                monomeric_object = pickle.load(f)

        self.assertTrue(hasattr(monomeric_object, 'feature_dict'), "Loaded object does not have 'feature_dict' attribute.")
        features = monomeric_object.feature_dict

        temp_sequence = features['template_sequence'][0].decode('utf-8')
        target_sequence = features['sequence'][0].decode('utf-8')
        atom_coords = features['template_all_atom_positions'][0]
        template_path = self.TEST_DATA_DIR / 'templates' / f'{file_name}.{file_extension}'
        assert len(temp_sequence) > 0
        assert (atom_coords.any()) > 0

        atom_seq, seqres_seq = extract_seqs(template_path, chain_id)
        print(f"target sequence: {target_sequence}")
        print(len(target_sequence))
        print(f"template sequence: {temp_sequence}")
        print(len(temp_sequence))
        print(f"seq-seqres: {seqres_seq}")
        if seqres_seq:
            print(len(seqres_seq))
        print(f"seq-atom: {atom_seq}")
        print(len(atom_seq))

        residue_has_nonzero_coords = []
        for number, (s, a) in enumerate(zip(atom_seq, atom_coords)):
            if s == 'X':
                assert np.all(a == 0)
                residue_has_nonzero_coords.append(False)
            else:
                non_zero = np.any(a != 0)
                residue_has_nonzero_coords.append(non_zero)
                if non_zero:
                    seqres = seqres_seq[number] if seqres_seq else None
                    if seqres:
                        assert (s in seqres_seq)
                    assert np.any(a[:4] != 0)

    @parameterized.parameters(
        {'compress_features': True, 'file_name': '3L4Q', 'chain_id': 'A', 'file_extension': 'cif'},
        {'compress_features': False, 'file_name': '3L4Q', 'chain_id': 'A', 'file_extension': 'cif'},
    )
    def test_compress_features_flag(self, compress_features, file_name, chain_id, file_extension):
        self.run_features_generation(file_name, chain_id, file_extension, use_mmseqs2=False, compress_features=compress_features)

        json_pattern = f'{file_name}_{chain_id}.{file_name}.{file_extension}.{chain_id}_feature_metadata_*.json'
        if compress_features:
            json_pattern += '.xz'
        metadata_files = list((self.TEST_DATA_DIR / 'features').glob(json_pattern))
        self.assertTrue(len(metadata_files) > 0, "Metadata JSON file was not created.")
        json_path = metadata_files[0]

        pkl_filename = f'{file_name}_{chain_id}.{file_name}.{file_extension}.{chain_id}.pkl'
        if compress_features:
            pkl_filename += '.xz'
        pkl_path = self.TEST_DATA_DIR / 'features' / pkl_filename

        self.validate_generated_features(pkl_path, json_path, file_name, file_extension, chain_id, compress_features)

        # cleanup
        pkl_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)

    @parameterized.parameters(
        {'file_name': '3L4Q', 'chain_id': 'A', 'file_extension': 'cif', 'use_mmseqs2': False},
        {'file_name': '3L4Q', 'chain_id': 'C', 'file_extension': 'pdb', 'use_mmseqs2': False},
        {'file_name': 'RANdom_name1_.7-1_0', 'chain_id': 'B', 'file_extension': 'pdb', 'use_mmseqs2': False},
        {'file_name': 'RANdom_name1_.7-1_0', 'chain_id': 'C', 'file_extension': 'pdb', 'use_mmseqs2': False},
        {'file_name': 'GAPPY_PDB', 'chain_id': 'B', 'file_extension': 'pdb', 'use_mmseqs2': False},
        {'file_name': 'hetatoms', 'chain_id': 'A', 'file_extension': 'pdb', 'use_mmseqs2': False},
    )
    def test_run_features_generation(self, file_name, chain_id, file_extension, use_mmseqs2):
        self.run_features_generation(file_name, chain_id, file_extension, use_mmseqs2)

        pkl_filename = f'{file_name}_{chain_id}.{file_name}.{file_extension}.{chain_id}.pkl'
        pkl_path = self.TEST_DATA_DIR / 'features' / pkl_filename
        json_pattern = f'{file_name}_{chain_id}.{file_name}.{file_extension}.{chain_id}_feature_metadata_*.json'
        metadata_files = list((self.TEST_DATA_DIR / 'features').glob(json_pattern))
        self.assertTrue(len(metadata_files) > 0, "Metadata JSON file was not created.")
        json_path = metadata_files[0]

        self.validate_generated_features(pkl_path, json_path, file_name, file_extension, chain_id, compress_features=False)

        # cleanup
        pkl_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)

    def test_mmseqs2_without_data_dir(self):
        """Test that MMseqs2 works without data_dir flag."""
        self.run_features_generation('3L4Q', 'A', 'cif', True)

    @absltest.skip("use_mmseqs2 must not be set when running with --path_to_mmts")
    def test_6a_mmseqs2(self):
        self.run_features_generation('3L4Q', 'A', 'cif', True)


if __name__ == '__main__':
    absltest.main()
