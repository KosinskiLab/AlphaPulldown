#!/usr/bin/env python
"""
Functional Alphapulldown tests for AlphaLink (parameterised).

The script is identical for Slurm and workstation users – only the
wrapper decides *how* each case is executed.
"""
from __future__ import annotations
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import shutil
import pickle
import json
import re
from typing import Dict, List, Tuple, Any

from absl.testing import absltest, parameterized

import alphapulldown
from alphapulldown.utils.create_combinations import process_files


# --------------------------------------------------------------------------- #
#                       configuration / environment guards                    #
# --------------------------------------------------------------------------- #
# Point to the AlphaLink weights directory
ALPHALINK_WEIGHTS_DIR = os.getenv(
    "ALPHALINK_WEIGHTS_DIR",
    "/scratch/AlphaFold_DBs/alphalink_weights"   #  default for EMBL cluster
)
ALPHALINK_WEIGHTS_FILE = os.path.join(ALPHALINK_WEIGHTS_DIR, "AlphaLink-Multimer_SDA_v3.pt")
if not os.path.exists(ALPHALINK_WEIGHTS_FILE):
    absltest.skip("set $ALPHALINK_WEIGHTS_DIR to run AlphaLink functional tests")


# --------------------------------------------------------------------------- #
#                       common helper mix-in / assertions                     #
# --------------------------------------------------------------------------- #
class _TestBase(parameterized.TestCase):
    use_temp_dir = True  # Class variable to control directory behavior - default to True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Create a base directory for all test outputs
        if cls.use_temp_dir:
            cls.base_output_dir = Path(tempfile.mkdtemp(prefix="alphalink_test_"))
        else:
            cls.base_output_dir = Path("test/test_data/predictions/alphalink_backend")
            if cls.base_output_dir.exists():
                try:
                    shutil.rmtree(cls.base_output_dir)
                except (PermissionError, OSError) as e:
                    # If we can't remove the directory due to permissions, just warn and continue
                    print(f"Warning: Could not remove existing output directory {cls.base_output_dir}: {e}")
            cls.base_output_dir.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        super().setUp()

        # directories inside the repo (relative to this file)
        this_dir = Path(__file__).resolve().parent
        self.test_data_dir = this_dir / "test_data"
        self.test_fastas_dir = self.test_data_dir / "fastas"
        self.test_features_dir = this_dir / "test_data" / "features"
        self.test_protein_lists_dir = this_dir / "test_data" / "protein_lists"
        self.test_templates_dir = this_dir / "test_data" / "templates"
        self.test_modelling_dir = this_dir / "test_data" / "predictions"
        self.test_crosslinks_dir = this_dir / "alphalink"

        # Create a unique output directory for this test
        test_name = self._testMethodName
        self.output_dir = self.base_output_dir / test_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # paths to alphapulldown CLI scripts
        apd_path = Path(alphapulldown.__path__[0])
        self.script_multimer = apd_path / "scripts" / "run_multimer_jobs.py"
        self.script_single = apd_path / "scripts" / "run_structure_prediction.py"

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        # Clean up all test outputs after all tests are done
        if cls.use_temp_dir and cls.base_output_dir.exists():
            try:
                shutil.rmtree(cls.base_output_dir)
            except (PermissionError, OSError) as e:
                # If we can't remove the temp directory, just warn
                print(f"Warning: Could not remove temporary directory {cls.base_output_dir}: {e}")
                # Try to remove individual files that we can
                try:
                    for item in cls.base_output_dir.rglob("*"):
                        if item.is_file():
                            try:
                                item.unlink()
                            except (PermissionError, OSError):
                                pass  # Skip files we can't remove
                except Exception:
                    pass  # Ignore any errors during cleanup

    def _get_sequence_from_pkl(self, protein_name: str) -> str:
        """Extract sequence from a PKL file."""
        pkl_path = self.test_features_dir / f"{protein_name}.pkl"
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                monomeric_object = pickle.load(f)
            
            if hasattr(monomeric_object, 'feature_dict'):
                sequence = monomeric_object.feature_dict.get('sequence', [])
                if len(sequence) > 0:
                    return sequence[0].decode('utf-8')
        return None

    def _get_sequence_from_fasta(self, protein_name: str) -> str:
        """Extract sequence from a FASTA file with case-insensitive search."""
        fasta_path = self.test_fastas_dir / f"{protein_name}.fasta"
        if not fasta_path.exists():
            # Try case-insensitive search
            for fasta_file in self.test_fastas_dir.glob("*.fasta"):
                if fasta_file.stem.lower() == protein_name.lower():
                    fasta_path = fasta_file
                    break
        
        if fasta_path.exists():
            with open(fasta_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    return lines[1].strip()
        return None

    def _extract_expected_sequences(self, protein_list: str) -> List[Tuple[str, str]]:
        """
        Extract expected sequences from input files based on test case name.
        
        Args:
            protein_list: Name of the protein list file
            
        Returns:
            List of tuples (chain_id, sequence) for expected chains
        """
        expected_sequences = []
        
        # Read the protein list file
        protein_list_path = self.test_protein_lists_dir / protein_list
        with open(protein_list_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Extract test case name from filename
        test_case = protein_list.replace('.txt', '')
        
        for line in lines:
            if ";" in line:
                # Multiple proteins separated by semicolons
                sequences = self._process_mixed_line(line)
            else:
                # Single protein
                sequences = self._process_single_protein_line(line)
            
            expected_sequences.extend(sequences)
        
        return expected_sequences

    def _process_mixed_line(self, line: str) -> List[Tuple[str, str]]:
        """Process a line with multiple proteins/features separated by semicolons."""
        if ";" not in line:
            return []
        
        sequences = []
        parts = line.split(";")
        
        for i, part in enumerate(parts):
            part = part.strip()
            
            # Check if this part contains chopped protein format (commas and dashes)
            if "," in part and any("-" in token for token in part.split(",")):
                # This is a chopped protein format: "PROTEIN,regions"
                tokens = [x.strip() for x in part.split(",")]
                protein_name = tokens[0]
                regions = []
                for region_str in tokens[1:]:
                    if "-" in region_str:
                        s, e = region_str.split("-")
                        regions.append((int(s), int(e)))
                
                # Get chopped sequence
                sequence = self._get_chopped_sequence(protein_name, regions)
                if sequence:
                    chain_id = chr(ord('A') + i)
                    sequences.append((chain_id, sequence))
            else:
                # Regular protein name
                protein_name = part
                sequence = self._get_sequence_for_protein(protein_name)
                if sequence:
                    chain_id = chr(ord('A') + i)
                    sequences.append((chain_id, sequence))
        
        return sequences

    def _process_single_protein_line(self, line: str) -> List[Tuple[str, str]]:
        """Process a line with a single protein."""
        part = line.strip()
        
        # Handle homo-oligomer format: "PROTEIN,N" where N is the number of copies
        if "," in part:
            tokens = [x.strip() for x in part.split(",")]
            protein_name = tokens[0]
            
            # Check if second token is a number (homo-oligomer format)
            try:
                num_copies = int(tokens[1])
                # Check if there are additional tokens that look like regions (contain "-")
                if len(tokens) > 2 and any("-" in token for token in tokens[2:]):
                    # This is a chopped protein format: "PROTEIN,N,regions"
                    regions = []
                    for region_str in tokens[2:]:
                        if "-" in region_str:
                            s, e = region_str.split("-")
                            regions.append((int(s), int(e)))
                    
                    # Get chopped sequence
                    sequence = self._get_chopped_sequence(protein_name, regions)
                    if sequence:
                        # Return multiple copies of the chopped sequence
                        return [(chr(ord('A') + i), sequence) for i in range(num_copies)]
                else:
                    # Regular homo-oligomer format
                    sequence = self._get_sequence_for_protein(protein_name)
                    if sequence:
                        # Return multiple copies of the same sequence
                        return [(chr(ord('A') + i), sequence) for i in range(num_copies)]
            except ValueError:
                # If not a number, treat as regular protein name
                protein_name = part
                sequence = self._get_sequence_for_protein(protein_name)
                if sequence:
                    return [('A', sequence)]
        else:
            protein_name = part
            sequence = self._get_sequence_for_protein(protein_name)
            if sequence:
                return [('A', sequence)]
        
        return []

    def _get_chopped_sequence(self, protein_name: str, regions: list) -> str:
        """Get chopped sequence from a protein with specified regions."""
        # Get the full sequence from PKL or FASTA
        full_sequence = self._get_sequence_for_protein(protein_name)
        if not full_sequence:
            return None
        
        chopped_sequence = ""
        for start, end in regions:
            # Convert to 0-based indexing
            start_idx = start - 1
            end_idx = end  # end is exclusive
            chopped_sequence += full_sequence[start_idx:end_idx]
        return chopped_sequence

    def _get_sequence_for_protein(self, protein_name: str, chain_id: str = 'A') -> str:
        """Get sequence for a single protein, trying PKL first, then FASTA."""
        # Try PKL file first
        sequence = self._get_sequence_from_pkl(protein_name)
        if sequence:
            return sequence
        
        # Fallback to FASTA
        sequence = self._get_sequence_from_fasta(protein_name)
        if sequence:
            return sequence
        
        return None

    def _check_chain_counts_and_sequences(self, protein_list: str):
        """
        Check that the predicted PDB files have the correct number of chains
        and that the sequences match the expected input sequences.
        
        Args:
            protein_list: Name of the protein list file
        """
        # Get expected sequences from input files
        expected_sequences = self._extract_expected_sequences(protein_list)
        
        print(f"\nExpected sequences: {expected_sequences}")
        
        # Find the predicted PDB files (should be in the output directory)
        pdb_files = list(self.output_dir.glob("ranked_*.pdb"))
        if not pdb_files:
            self.fail("No predicted PDB files found")
        
        # Use the first PDB file (should be the best ranked one)
        pdb_path = pdb_files[0]
        print(f"Checking PDB file: {pdb_path}")
        
        # Extract chains and sequences from the PDB file
        actual_chains_and_sequences = self._extract_pdb_chains_and_sequences(pdb_path)
        
        print(f"Actual chains and sequences: {actual_chains_and_sequences}")
        
        # Check that the number of chains matches
        self.assertEqual(
            len(actual_chains_and_sequences), 
            len(expected_sequences),
            f"Expected {len(expected_sequences)} chains, but found {len(actual_chains_and_sequences)}"
        )
        
        # For AlphaLink cases, check exact sequence matches
        actual_sequences = [seq for _, seq in actual_chains_and_sequences]
        expected_sequences_only = [seq for _, seq in expected_sequences]
        
        # Sort sequences for comparison (since chain order might vary)
        actual_sequences.sort()
        expected_sequences_only.sort()
        
        self.assertEqual(
            actual_sequences,
            expected_sequences_only,
            f"Sequences don't match. Expected: {expected_sequences_only}, Actual: {actual_sequences}"
        )

    def _extract_pdb_chains_and_sequences(self, pdb_path: Path) -> List[Tuple[str, str]]:
        """
        Extract chain IDs and sequences from a PDB file using Biopython.
        
        Args:
            pdb_path: Path to the PDB file
            
        Returns:
            List of tuples (chain_id, sequence) for chains in the PDB file
        """
        chains_and_sequences = []
        
        try:
            from Bio.PDB import PDBParser
            
            # Parse the PDB file
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("model", str(pdb_path))
            
            # Get the first model (should be the only one for AlphaLink)
            model = structure[0]
            
            # Extract sequences for each chain
            for chain in model:
                chain_id = chain.id
                
                # Get residues in order
                residues = list(chain.get_residues())
                residues.sort(key=lambda r: r.id[1])  # Sort by residue number
                
                # Extract protein sequence
                sequence = ""
                for residue in residues:
                    hetfield, resseq, icode = residue.id
                    res_name = residue.resname
                    
                    if hetfield == " ":
                        # Standard residue (protein)
                        if res_name in self._protein_letters_3to1:
                            sequence += self._protein_letters_3to1[res_name]
                        else:
                            sequence += "X"  # Unknown residue
                
                if sequence:  # Only add if we have a sequence
                    chains_and_sequences.append((chain_id, sequence))
                    
        except ImportError:
            # Fallback to regex parsing if Biopython is not available
            print("Warning: Biopython not available, using regex parsing")
            chains_and_sequences = self._extract_pdb_chains_and_sequences_regex(pdb_path)
        except Exception as e:
            print(f"Error parsing PDB with Biopython: {e}")
            # Fallback to regex parsing
            chains_and_sequences = self._extract_pdb_chains_and_sequences_regex(pdb_path)
        
        return chains_and_sequences

    @property
    def _protein_letters_3to1(self):
        """Protein three-letter to one-letter code mapping using Bio.Data.PDBData."""
        try:
            from Bio.Data.PDBData import protein_letters_3to1_extended
            return protein_letters_3to1_extended
        except ImportError:
            # Fallback if PDBData is not available
            from Bio.Data.IUPACData import protein_letters_3to1
            return {**protein_letters_3to1, 'UNK': 'X'}

    def _extract_pdb_chains_and_sequences_regex(self, pdb_path: Path) -> List[Tuple[str, str]]:
        """
        Fallback method to extract chain IDs and sequences from a PDB file using regex.
        
        Args:
            pdb_path: Path to the PDB file
            
        Returns:
            List of tuples (chain_id, sequence) for chains in the PDB file
        """
        chains_and_sequences = []
        
        with open(pdb_path, 'r') as f:
            pdb_content = f.read()
        
        # Extract unique chain IDs from ATOM records
        # Format: ATOM/HETATM serial atom_name res_name chain_id res_seq
        atom_pattern = r'^(ATOM|HETATM)\s+\d+\s+\w+\s+(\w{3})\s+([A-Z])\s+(\d+)'
        atom_matches = re.findall(atom_pattern, pdb_content, re.MULTILINE)
        
        # Group residues by chain_id
        chain_sequences = {}
        for _, res_name, chain_id, res_seq in atom_matches:
            if chain_id not in chain_sequences:
                chain_sequences[chain_id] = {}
            chain_sequences[chain_id][int(res_seq)] = res_name
        
        # Convert three-letter codes to one-letter sequences for each chain
        try:
            # Use comprehensive dictionaries from PDBData
            three_to_one = self._protein_letters_3to1
        except ImportError:
            # Fallback if PDBData is not available
            from Bio.Data.IUPACData import protein_letters_3to1
            three_to_one = {**protein_letters_3to1, 'UNK': 'X'}
        
        # Build sequences for each chain
        for chain_id, residues in chain_sequences.items():
            # Sort by residue number
            sorted_residues = sorted(residues.items())
            sequence = ''.join([three_to_one.get(res[1], 'X') for res in sorted_residues])
            
            if sequence:  # Only add if we have a sequence
                chains_and_sequences.append((chain_id, sequence))
        
        return chains_and_sequences

    # ---------------- assertions reused by all subclasses ----------------- #
    def _runCommonTests(self, res: subprocess.CompletedProcess):
        print(res.stdout)
        print(res.stderr)
        self.assertEqual(res.returncode, 0, "sub-process failed")

        # Check if output directory exists (in case prediction was skipped)
        if not self.output_dir.exists():
            print(f"Output directory {self.output_dir} does not exist. This may be because the prediction was skipped due to resume functionality.")
            # If the prediction was skipped, we should still have some output files in the parent directory
            parent_dir = self.output_dir.parent
            if parent_dir.exists():
                files = list(parent_dir.iterdir())
                print(f"contents of {parent_dir}: {[f.name for f in files]}")
                
                # Check if there are any AlphaLink2 model files in the parent directory
                alphalink_model_files = [f for f in files if f.name.startswith("AlphaLink2_model_") and f.name.endswith(".pdb")]
                if alphalink_model_files:
                    print("Found AlphaLink2 model files in parent directory. Prediction was likely skipped due to resume functionality.")
                    return
                else:
                    self.fail(f"No output directory found at {self.output_dir} and no AlphaLink2 model files found in parent directory {parent_dir}")
            else:
                self.fail(f"Neither output directory {self.output_dir} nor parent directory {parent_dir} exist")

        # Look in the parent directory for output files
        files = list(self.output_dir.iterdir())
        print(f"contents of {self.output_dir}: {[f.name for f in files]}")

        # Check for AlphaLink output files
        # 1. Main output files
        self.assertIn("ranking_debug.json", {f.name for f in files})
        
        # 2. Check for ranked PDB files
        ranked_pdb_files = [f for f in files if f.name.startswith("ranked_") and f.name.endswith(".pdb")]
        self.assertTrue(len(ranked_pdb_files) > 0, "No ranked PDB files found")
        
        # 3. Check for AlphaLink2 model files
        alphalink_model_files = [f for f in files if f.name.startswith("AlphaLink2_model_") and f.name.endswith(".pdb")]
        self.assertTrue(len(alphalink_model_files) > 0, "No AlphaLink2 model PDB files found")
        
        # 4. Check for PAE files
        pae_files = [f for f in files if f.name.startswith("pae_AlphaLink2_model_") and f.name.endswith(".json")]
        self.assertTrue(len(pae_files) > 0, "No PAE JSON files found")
        
        # 5. Check for PAE plots
        pae_plot_files = [f for f in files if f.name.startswith("AlphaLink2_model_") and f.name.endswith("_pae.png")]
        self.assertTrue(len(pae_plot_files) > 0, "No PAE plot files found")

        # 6. Verify ranking debug JSON
        with open(self.output_dir / "ranking_debug.json") as f:
            ranking_data = json.load(f)
            self.assertIn("iptm+ptm", ranking_data)
            self.assertIn("order", ranking_data)
            self.assertTrue(len(ranking_data["iptm+ptm"]) > 0, "No ranking scores found")
            self.assertTrue(len(ranking_data["order"]) > 0, "No model order found")
            
            # Verify that the number of models matches
            self.assertEqual(
                len(ranking_data["iptm+ptm"]), 
                len(ranking_data["order"]),
                "Number of scores and model names should match"
            )
            
            print(f"✓ Verified ranking_debug.json with {len(ranking_data['iptm+ptm'])} models")

    # convenience builder
    def _args(self, *, plist, script):
        # Determine mode from protein list name
        if "homooligomer" in plist:
            mode = "homo-oligomer"
        else:
            mode = "custom"
            
        if script == "run_structure_prediction.py":
            # Format from run_multimer_jobs.py input to run_structure_prediction.py input
            buffer = io.StringIO()
            _ = process_files(
                input_files=[str(self.test_protein_lists_dir / plist)],
                output_path=buffer,
                exclude_permutations = True
            )
            buffer.seek(0)
            formatted_input_lines = [x.strip().replace(",", ":").replace(";", "+") for x in buffer.readlines() if x.strip()]
            # Use the first non-empty line as the input string
            formatted_input = formatted_input_lines[0] if formatted_input_lines else ""
            args = [
                sys.executable,
                str(self.script_single),
                f"--input={formatted_input}",
                f"--output_directory={self.output_dir}",
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_directory={ALPHALINK_WEIGHTS_FILE}",
                f"--features_directory={self.test_features_dir}",
                "--fold_backend=alphalink",
                f"--crosslinks={self.test_crosslinks_dir}/example_crosslink.pkl.gz",
            ]
            
            return args
        elif script == "run_multimer_jobs.py":
            args = [
                sys.executable,
                str(self.script_multimer),
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_dir={ALPHALINK_WEIGHTS_DIR}",
                f"--monomer_objects_dir={self.test_features_dir}",
                "--job_index=1",
                f"--output_path={self.output_dir}",
                f"--mode={mode}",
                "--use_alphalink=True",
                f"--alphalink_weight={ALPHALINK_WEIGHTS_FILE}",
                f"--crosslinks={self.test_crosslinks_dir}/example_crosslink.pkl.gz",
                (
                    "--oligomer_state_file"
                    if mode == "homo-oligomer"
                    else "--protein_lists"
                ) + f"={self.test_protein_lists_dir / plist}",
            ]
            return args


# --------------------------------------------------------------------------- #
#                        parameterised "run mode" tests                       #
# --------------------------------------------------------------------------- #
class TestAlphaLinkRunModes(_TestBase):
    @parameterized.named_parameters(
        dict(testcase_name="monomer", protein_list="test_monomer.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="dimer", protein_list="test_dimer.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="trimer", protein_list="test_trimer.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="homo_oligomer", protein_list="test_homooligomer.txt", mode="homo-oligomer", script="run_multimer_jobs.py"),
        dict(testcase_name="chopped_dimer", protein_list="test_dimer_chopped.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="long_name", protein_list="test_long_name.txt", mode="custom", script="run_structure_prediction.py"),
    )
    def test_(self, protein_list, mode, script):
        # Create environment with GPU settings
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        # Set environment variables globally to prevent threading conflicts
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        os.environ["NUMEXPR_NUM_THREADS"] = "4"
        
        # JAX/TensorFlow specific settings
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        
        # Allow TensorFlow to manage its own threading
        os.environ["TF_NUM_INTEROP_THREADS"] = "4"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ["JAX_ENABLE_X64"] = "false"
        
        # Disable problematic XLA optimizations
        os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=0"
        os.environ["JAX_FLASH_ATTENTION_IMPL"] = "xla"
        
        # Additional environment variables for the subprocess
        env.update({
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4", 
            "NUMEXPR_NUM_THREADS": "4",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "TF_NUM_INTEROP_THREADS": "4",
            "TF_NUM_INTRAOP_THREADS": "4",
            "JAX_PLATFORM_NAME": "gpu",
            "JAX_ENABLE_X64": "false",
            "XLA_FLAGS": "--xla_gpu_force_compilation_parallelism=0",
            "JAX_FLASH_ATTENTION_IMPL": "xla"
        })
        
        # Debug output
        print("\nEnvironment variables:")
        print(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES')}")
        print(f"PYTORCH_CUDA_ALLOC_CONF: {env.get('PYTORCH_CUDA_ALLOC_CONF')}")
        
        # Check GPU availability
        try:
            import torch
            print("\nPyTorch GPU devices:")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current device: {torch.cuda.current_device()}")
        except Exception as e:
            print(f"\nError checking PyTorch GPU: {e}")
        
        res = subprocess.run(
            self._args(plist=protein_list, script=script),
            capture_output=True,
            text=True,
            env=env
        )
        self._runCommonTests(res)
        
        # Check chain counts and sequences
        self._check_chain_counts_and_sequences(protein_list)


# --------------------------------------------------------------------------- #
#                        parameterised "run mode" tests (no crosslinks)        #
# --------------------------------------------------------------------------- #
class TestAlphaLinkRunModesNoCrosslinks(_TestBase):
    @parameterized.named_parameters(
        dict(testcase_name="monomer_no_xl", protein_list="test_monomer.txt", mode="custom", script="run_multimer_jobs.py"),
        dict(testcase_name="dimer_no_xl", protein_list="test_dimer.txt", mode="custom", script="run_multimer_jobs.py"),
    )
    def test_(self, protein_list, mode, script):
        # Create environment with GPU settings
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        # Set environment variables globally to prevent threading conflicts
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        os.environ["NUMEXPR_NUM_THREADS"] = "4"
        
        # JAX/TensorFlow specific settings
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        
        # Allow TensorFlow to manage its own threading
        os.environ["TF_NUM_INTEROP_THREADS"] = "4"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ["JAX_ENABLE_X64"] = "false"
        
        # Disable problematic XLA optimizations
        os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=0"
        os.environ["JAX_FLASH_ATTENTION_IMPL"] = "xla"
        
        # Additional environment variables for the subprocess
        env.update({
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4", 
            "NUMEXPR_NUM_THREADS": "4",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "TF_NUM_INTEROP_THREADS": "4",
            "TF_NUM_INTRAOP_THREADS": "4",
            "JAX_PLATFORM_NAME": "gpu",
            "JAX_ENABLE_X64": "false",
            "XLA_FLAGS": "--xla_gpu_force_compilation_parallelism=0",
            "JAX_FLASH_ATTENTION_IMPL": "xla"
        })
        
        # Debug output
        print("\nEnvironment variables:")
        print(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES')}")
        print(f"PYTORCH_CUDA_ALLOC_CONF: {env.get('PYTORCH_CUDA_ALLOC_CONF')}")
        
        # Check GPU availability
        try:
            import torch
            print("\nPyTorch GPU devices:")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current device: {torch.cuda.current_device()}")
        except Exception as e:
            print(f"\nError checking PyTorch GPU: {e}")
        
        res = subprocess.run(
            self._args_no_crosslinks(plist=protein_list, script=script),
            capture_output=True,
            text=True,
            env=env
        )
        self._runCommonTests(res)
        
        # Check chain counts and sequences
        self._check_chain_counts_and_sequences(protein_list)

    def _args_no_crosslinks(self, *, plist, script):
        """Generate arguments for tests without crosslinks."""
        # Determine mode from protein list name
        if "homooligomer" in plist:
            mode = "homo-oligomer"
        else:
            mode = "custom"
            
        if script == "run_structure_prediction.py":
            # Format from run_multimer_jobs.py input to run_structure_prediction.py input
            buffer = io.StringIO()
            _ = process_files(
                input_files=[str(self.test_protein_lists_dir / plist)],
                output_path=buffer,
                exclude_permutations = True
            )
            buffer.seek(0)
            formatted_input_lines = [x.strip().replace(",", ":").replace(";", "+") for x in buffer.readlines() if x.strip()]
            # Use the first non-empty line as the input string
            formatted_input = formatted_input_lines[0] if formatted_input_lines else ""
            args = [
                sys.executable,
                str(self.script_single),
                f"--input={formatted_input}",
                f"--output_directory={self.output_dir}",
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_directory={ALPHALINK_WEIGHTS_FILE}",
                f"--features_directory={self.test_features_dir}",
                "--fold_backend=alphalink",
                # No crosslinks parameter
            ]
            
            return args
        elif script == "run_multimer_jobs.py":
            args = [
                sys.executable,
                str(self.script_multimer),
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_dir={ALPHALINK_WEIGHTS_DIR}",
                f"--monomer_objects_dir={self.test_features_dir}",
                "--job_index=1",
                f"--output_path={self.output_dir}",
                f"--mode={mode}",
                "--use_alphalink=True",
                f"--alphalink_weight={ALPHALINK_WEIGHTS_FILE}",
                # No crosslinks parameter
                (
                    "--oligomer_state_file"
                    if mode == "homo-oligomer"
                    else "--protein_lists"
                ) + f"={self.test_protein_lists_dir / plist}",
            ]
            return args


# --------------------------------------------------------------------------- #
def _parse_test_args():
    """Parse test-specific arguments that work with both absltest and pytest."""
    # Check for --use-temp-dir in sys.argv or environment variable
    use_temp_dir = '--use-temp-dir' in sys.argv or os.getenv('USE_TEMP_DIR', '').lower() in ('1', 'true', 'yes')
    
    # Remove the argument from sys.argv if present to avoid conflicts
    while '--use-temp-dir' in sys.argv:
        sys.argv.remove('--use-temp-dir')
    
    return use_temp_dir

# Parse arguments at module level to work with both absltest and pytest
_TestBase.use_temp_dir = _parse_test_args()

if __name__ == "__main__":
    absltest.main() 