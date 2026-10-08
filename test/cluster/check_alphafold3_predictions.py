#!/usr/bin/env python
"""
Functional Alphapulldown tests for AlphaFold3 (parameterised).

The script is identical for Slurm and workstation users – only the
wrapper decides *how* each case is executed.
"""
from __future__ import annotations
import os
import subprocess
import time
import sys
import tempfile
import importlib.util
from pathlib import Path
import shutil
import json
import numpy as np
import re
import unittest
from typing import Dict, List, Tuple, Any, Union

from absl.testing import absltest, parameterized

import alphapulldown
from alphafold3.common import folding_input
from alphafold3.structure import mmcif as af3_mmcif
from alphapulldown.utils.feature_metadata import (
    AF3_METADATA_MARKER,
    extract_metadata_from_af3_json,
)
from alphapulldown_input_parser import generate_fold_specifications

# The sibling helper module; pytest puts this directory on sys.path, a direct load may not.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import af3_gpu_checks  # noqa: E402

# Test data helpers shared with test/integration/test_af3_input_preparation.py.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from af3_fixtures import (  # noqa: E402
    Af3TestData,
    _a3m_query_sequence,
    _load_feature_dict,
    _load_feature_metadata,
    _load_json_payload,
    _metadata_bool,
    _non_empty_identifier_count,
    _protein_entries_from_af3_input,
)


# --------------------------------------------------------------------------- #
#                       configuration / environment guards                    #
# --------------------------------------------------------------------------- #
# AF3 model weights used by inference tests.
DATA_DIR = os.getenv(
    "ALPHAFOLD_DATA_DIR",
    "/g/kosinski/dima/alphafold3_weights/"   #  default for EMBL cluster
)
AF3_DATABASE_DIR = Path(
    os.getenv("ALPHAFOLD3_DATABASE_DIR", "/g/alphafold/AlphaFold_DBs/3.0.0")
)
if not os.path.exists(DATA_DIR):
    absltest.skip("set $ALPHAFOLD_DATA_DIR to run Alphafold functional tests")
REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "test"


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


# --------------------------------------------------------------------------- #
#                       common helper mix-in / assertions                     #
# --------------------------------------------------------------------------- #
class _TestBase(Af3TestData, parameterized.TestCase):
    use_temp_dir = True  # Class variable to control directory behavior - default to True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        skip_reason = _gpu_functional_test_skip_reason()
        if skip_reason:
            raise unittest.SkipTest(skip_reason)
        # Create a base directory for all test outputs
        if cls.use_temp_dir:
            cls.base_output_dir = Path(tempfile.mkdtemp(prefix="af3_test_"))
        else:
            cls.base_output_dir = Path("test/test_data/predictions/af3_backend")
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
        self.test_data_dir = TEST_ROOT / "test_data"
        self.test_fastas_dir = self.test_data_dir / "fastas"
        self.test_features_dir = self.test_data_dir / "features"
        self.test_protein_lists_dir = self.test_data_dir / "protein_lists"
        self.test_templates_dir = self.test_data_dir / "templates"
        self.test_modelling_dir = self.test_data_dir / "predictions"

        # Create a unique output directory for this test
        test_name = self._testMethodName
        self.output_dir = self.base_output_dir / test_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # paths to alphapulldown CLI scripts
        apd_path = Path(alphapulldown.__path__[0])
        self.script_multimer = apd_path / "scripts" / "run_multimer_jobs.py"
        self.script_single = apd_path / "scripts" / "run_structure_prediction.py"
        self.script_create_features = (
            apd_path / "scripts" / "create_individual_features.py"
        )

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

    def _chain_id_from_index(self, index: int) -> str:
        """Mirror AF3's reverse-spreadsheet chain ID progression."""
        if index < 26:
            return chr(ord('A') + index)
        first_char = chr(ord('A') + (index // 26) - 1)
        second_char = chr(ord('A') + (index % 26))
        return first_char + second_char

    def _process_homo_oligomer_line(self, line: str) -> List[Tuple[str, str]]:
        """Process a homo-oligomer line (format: 'PROTEIN,number')."""
        if "," not in line:
            return []
        
        parts = line.split(",")
        protein_name = parts[0].strip()
        num_copies = int(parts[1].strip())
        
        sequence = self._get_sequence_for_protein(protein_name)
        if not sequence:
            return []
        
        sequences = []
        for i in range(num_copies):
            chain_id = chr(ord('A') + i)
            sequences.append((chain_id, sequence))
        
        return sequences

    def _process_mixed_line(self, line: str) -> List[Tuple[str, str]]:
        """Process a line with multiple proteins/features separated by semicolons."""
        if ";" not in line:
            return []
        
        sequences = []
        parts = line.split(";")
        
        for i, part in enumerate(parts):
            part = part.strip()
            
            if part.endswith('.json'):
                # JSON input
                json_sequences = self._get_sequence_from_json(part)
                for chain_id, sequence in json_sequences:
                    if chain_id == 'A':  # Use default chain ID if not specified
                        chain_id = chr(ord('A') + i)
                    sequences.append((chain_id, sequence))
            else:
                # Protein input (handle chopped proteins)
                if "," in part:
                    # Extract protein name before first comma
                    protein_name = part.split(",")[0].strip()
                else:
                    protein_name = part
                
                sequence = self._get_sequence_for_protein(protein_name)
                if sequence:
                    chain_id = chr(ord('A') + i)
                    sequences.append((chain_id, sequence))
        
        return sequences

    def _process_single_protein_line(self, line: str) -> List[Tuple[str, str]]:
        """Process a line with a single protein."""
        part = line.strip()
        
        if part.endswith('.json'):
            # JSON input
            return self._get_sequence_from_json(part)
        else:
            # Protein input (handle chopped proteins)
            if "," in part:
                # Extract protein name before first comma
                protein_name = part.split(",")[0].strip()
            else:
                protein_name = part
            
            sequence = self._get_sequence_for_protein(protein_name)
            if sequence:
                return [('A', sequence)]
        
        return []

    def _process_homo_oligomer_chopped_line(self, line: str) -> List[Tuple[str, str]]:
        """Process a homo-oligomer of chopped proteins (format: 'PROTEIN,number,regions')."""
        if "," not in line:
            return []
        
        parts = line.split(",")
        if len(parts) < 3:
            return []
        
        protein_name = parts[0].strip()
        num_copies = int(parts[1].strip())
        
        # Parse regions (everything after the number of copies)
        regions = []
        for region_str in parts[2:]:
            if "-" in region_str:
                s, e = region_str.split("-")
                regions.append((int(s), int(e)))

        # AF3 cannot represent immediately repeated author residue IDs at a
        # region boundary (e.g. 6-7 followed by 7-8). Collapse only that shared
        # boundary residue while keeping the explicit region naming unchanged.
        normalized_regions = []
        for start, end in regions:
            if normalized_regions and start == normalized_regions[-1][1]:
                start += 1
            if start <= end:
                normalized_regions.append((start, end))

        region_sequences = self._get_region_sequences(protein_name, normalized_regions)
        if not region_sequences:
            return []
        
        concatenated_sequence = "".join(region_sequences)
        sequences = []
        for copy_index in range(num_copies):
            chain_id = self._chain_id_from_index(copy_index)
            sequences.append((chain_id, concatenated_sequence))
        
        return sequences

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
            match test_case:
                case "test_homooligomer":
                    # Homo-oligomer format: "PROTEIN,number"
                    sequences = self._process_homo_oligomer_line(line)
                
                case "test_monomer":
                    # Single protein
                    sequences = self._process_single_protein_line(line)
                
                case "test_dimer" | "test_trimer" | "test_truemultimer":
                    # Multiple proteins separated by semicolons
                    sequences = self._process_mixed_line(line)
                
                case "test_dimer_chopped":
                    # Chopped proteins (comma-separated ranges)
                    sequences = self._process_chopped_protein_line(line)
                
                case "test_long_name":
                    # Homo-oligomer of chopped proteins: "PROTEIN,number,regions"
                    sequences = self._process_homo_oligomer_chopped_line(line)
                
                case "test_monomer_with_rna" | "test_monomer_with_dna" | "test_monomer_with_ligand":
                    # Mixed inputs (protein + JSON)
                    sequences = self._process_mixed_line(line)
                
                case "test_protein_with_ptms":
                    # JSON-only input
                    sequences = self._process_single_protein_line(line)
                
                case "test_multi_seeds_samples":
                    # Test case for multiple seeds and diffusion samples (chopped protein)
                    sequences = self._process_chopped_protein_line(line)
                
                case _:
                    # Default case: try to process as mixed line
                    sequences = self._process_mixed_line(line)
            
            expected_sequences.extend(sequences)
        
        return expected_sequences

    def _process_chopped_protein_line(self, line: str) -> List[Tuple[str, str]]:
        """Process a line with chopped proteins (comma-separated ranges)."""

        def parse_protein_and_regions(part: str):
            # Example: A0A075B6L2,1-10,2-5,3-12
            tokens = [x.strip() for x in part.split(",")]
            protein_name = tokens[0]
            regions = []
            for region_str in tokens[1:]:
                if "-" in region_str:
                    s, e = region_str.split("-")
                    regions.append((int(s), int(e)))
            return protein_name, regions

        if ";" in line:
            # Multiple chopped proteins
            sequences = []
            parts = line.split(";")
            for part in parts:
                part = part.strip()
                if "," in part:
                    protein_name, regions = parse_protein_and_regions(part)
                    region_sequences = self._get_region_sequences(protein_name, regions)
                    if not region_sequences:
                        continue
                    chain_id = self._chain_id_from_index(len(sequences))
                    sequences.append((chain_id, "".join(region_sequences)))
                else:
                    protein_name = part
                    sequence = self._get_sequence_for_protein(protein_name)
                    if not sequence:
                        continue
                    chain_id = self._chain_id_from_index(len(sequences))
                    sequences.append((chain_id, sequence))
            return sequences
        else:
            # Single chopped protein
            part = line.strip()
            if "," in part:
                protein_name, regions = parse_protein_and_regions(part)
                region_sequences = self._get_region_sequences(protein_name, regions)
                if region_sequences:
                    return [('A', "".join(region_sequences))]
            else:
                protein_name = part
                sequence = self._get_sequence_for_protein(protein_name)
            if sequence:
                return [('A', sequence)]
        return []

    def _extract_cif_chains_and_sequences(self, cif_path: Path) -> List[Tuple[str, str]]:
        """
        Extract chain IDs and sequences from a CIF file.
        
        Args:
            cif_path: Path to the CIF file
            
        Returns:
            List of tuples (chain_id, sequence) for chains in the CIF file
        """
        chains_and_sequences = []
        
        try:
            from alphafold3.cpp import cif_dict

            with open(cif_path, "rt") as handle:
                cif = cif_dict.from_string(handle.read())

            sequences_by_chain = {}

            if "_pdbx_poly_seq_scheme.asym_id" in cif:
                asym_ids = cif.get_array("_pdbx_poly_seq_scheme.asym_id", dtype=object)
                mon_ids = cif.get_array("_pdbx_poly_seq_scheme.mon_id", dtype=object)

                for chain_id, mon_id in zip(asym_ids, mon_ids, strict=True):
                    sequence = sequences_by_chain.setdefault(chain_id, "")
                    if mon_id in self._protein_letters_3to1:
                        sequence += self._protein_letters_3to1[mon_id]
                    elif mon_id in self._dna_letters_3to1:
                        sequence += self._dna_letters_3to1[mon_id]
                    elif mon_id in self._rna_letters_3to1:
                        sequence += self._rna_letters_3to1[mon_id]
                    elif mon_id + "  " in self._rna_letters_3to1:
                        sequence += self._rna_letters_3to1[mon_id + "  "]
                    elif mon_id + " " in self._dna_letters_3to1:
                        sequence += self._dna_letters_3to1[mon_id + " "]
                    elif mon_id == "HYS":
                        sequence += "H"
                    elif mon_id == "2MG":
                        sequence += "G"
                    else:
                        sequence += "X"
                    sequences_by_chain[chain_id] = sequence

            for scheme_prefix in ("_pdbx_nonpoly_scheme", "_pdbx_branch_scheme"):
                asym_key = f"{scheme_prefix}.asym_id"
                mon_key = f"{scheme_prefix}.mon_id"
                if asym_key not in cif or mon_key not in cif:
                    continue
                asym_ids = cif.get_array(asym_key, dtype=object)
                mon_ids = cif.get_array(mon_key, dtype=object)
                for chain_id, mon_id in zip(asym_ids, mon_ids, strict=True):
                    if mon_id in {"HOH", "DOD"}:
                        continue
                    sequence = sequences_by_chain.setdefault(chain_id, "")
                    ligand_codes = [] if not sequence else sequence.split("+")
                    ligand_codes.append(mon_id if mon_id in self._ligand_ccd_codes else "UNKNOWN")
                    sequences_by_chain[chain_id] = "+".join(ligand_codes)

            chain_order = (
                list(cif.get_array("_struct_asym.id", dtype=object))
                if "_struct_asym.id" in cif
                else list(sequences_by_chain.keys())
            )
            for chain_id in chain_order:
                sequence = sequences_by_chain.get(chain_id)
                if sequence:
                    chains_and_sequences.append((chain_id, sequence))
            if chains_and_sequences:
                return chains_and_sequences
        except ImportError:
            pass
        except Exception as e:
            print(f"Error parsing CIF with AF3 cif_dict: {e}")

        try:
            from Bio.PDB import MMCIFParser
            
            # Parse the CIF file
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("model", str(cif_path))
            
            # Get the first model (should be the only one for AlphaFold3)
            model = structure[0]
            
            # Extract sequences for each chain
            for chain in model:
                chain_id = chain.id
                
                # Keep the residue order from the file instead of sorting by
                # residue number so discontinuous numbering remains testable.
                residues = list(chain.get_residues())
                
                # Separate standard residues from HETATM records
                standard_residues = []
                hetatm_residues = []
                
                for residue in residues:
                    hetfield, resseq, icode = residue.id
                    res_name = residue.resname
                    
                    if hetfield == " ":
                        # Standard residue (protein, DNA, RNA)
                        standard_residues.append((resseq, res_name))
                    elif hetfield != "W":  # Skip water molecules
                        # HETATM record (ligand or PTM)
                        hetatm_residues.append((resseq, res_name))
                
                # Check if this chain contains any HETATM records (ligands)
                has_ligand_hetatm = any(res_name in self._ligand_ccd_codes for _, res_name in hetatm_residues)
                
                if has_ligand_hetatm:
                    # This is a ligand chain - extract HETATM residues
                    ligand_codes = []
                    for _, res_name in hetatm_residues:
                        if res_name in self._ligand_ccd_codes:
                            ligand_codes.append(res_name)
                        else:
                            ligand_codes.append("UNKNOWN")
                    
                    if ligand_codes:
                        # Join multiple ligand codes if present (e.g., ["ATP", "MG"] -> "ATP+MG")
                        sequence = '+'.join(ligand_codes)
                    else:
                        sequence = "UNKNOWN_LIGAND"
                else:
                    # This is a polymer chain (protein, DNA, RNA) - extract base sequence
                    sequence = ""
                    unknown_residues = []
                    
                    for _, res_name in standard_residues:
                        # Try protein first
                        if res_name in self._protein_letters_3to1:
                            sequence += self._protein_letters_3to1[res_name]
                        # Try DNA
                        elif res_name in self._dna_letters_3to1:
                            sequence += self._dna_letters_3to1[res_name]
                        # Try RNA
                        elif res_name in self._rna_letters_3to1:
                            sequence += self._rna_letters_3to1[res_name]
                        # Try RNA with spaces (PDBData format)
                        elif res_name + "  " in self._rna_letters_3to1:
                            sequence += self._rna_letters_3to1[res_name + "  "]
                        # Try DNA with spaces (PDBData format)
                        elif res_name + " " in self._dna_letters_3to1:
                            sequence += self._dna_letters_3to1[res_name + " "]
                        else:
                            sequence += "X"  # Unknown residue
                            unknown_residues.append(res_name)
                    
                    # Debug: print unknown residues
                    if unknown_residues:
                        print(f"Warning: Unknown residues in chain {chain_id}: {set(unknown_residues)}")
                    
                    # Apply PTMs from HETATM records if present
                    if hetatm_residues and sequence:
                        sequence = self._apply_ptms_from_hetatm(sequence, hetatm_residues)
                
                if sequence:  # Only add if we have a sequence
                    chains_and_sequences.append((chain_id, sequence))
                    
        except ImportError:
            # Fallback to regex parsing if Biopython is not available
            print("Warning: Biopython not available, using regex parsing")
            chains_and_sequences = self._extract_cif_chains_and_sequences_regex(cif_path)
        except Exception as e:
            print(f"Error parsing CIF with Biopython: {e}")
            # Fallback to regex parsing
            chains_and_sequences = self._extract_cif_chains_and_sequences_regex(cif_path)
        
        return chains_and_sequences

    def _extract_cif_chain_residue_numbers(self, cif_path: Path) -> List[Tuple[str, List[Union[int, str]]]]:
        """Extract author-facing residue numbers for each polymer chain from a CIF file."""
        try:
            from alphafold3.cpp import cif_dict

            with open(cif_path, "rt") as handle:
                cif = cif_dict.from_string(handle.read())

            asym_ids = cif.get_array("_pdbx_poly_seq_scheme.asym_id", dtype=object)
            auth_seq_nums = cif.get_array(
                "_pdbx_poly_seq_scheme.auth_seq_num", dtype=object
            )
            ins_codes = cif.get_array(
                "_pdbx_poly_seq_scheme.pdb_ins_code", dtype=object
            )

            chain_residue_numbers = []
            chain_to_numbers = {}
            for chain_id, auth_seq_num, ins_code in zip(
                asym_ids,
                auth_seq_nums,
                ins_codes,
                strict=True,
            ):
                residue_numbers = chain_to_numbers.setdefault(chain_id, [])
                ins_code = str(ins_code)
                auth_seq_num = int(auth_seq_num)
                if ins_code in {".", "?"}:
                    residue_numbers.append(auth_seq_num)
                else:
                    residue_numbers.append(f"{auth_seq_num}{ins_code}")

            for chain_id, residue_numbers in chain_to_numbers.items():
                if residue_numbers:
                    chain_residue_numbers.append((chain_id, residue_numbers))
            return chain_residue_numbers
        except Exception as exc:
            self.fail(f"Failed to extract CIF residue numbers from {cif_path}: {exc}")

    def _apply_ptms_from_hetatm(self, sequence: str, hetatm_residues: List[Tuple[int, str]]) -> str:
        """
        Apply PTMs from HETATM records to the protein sequence.
        
        Args:
            sequence: Base protein sequence
            hetatm_residues: List of (residue_number, residue_name) tuples from HETATM records
            
        Returns:
            Modified sequence with PTMs applied
        """
        # Convert to list for easier modification
        seq_list = list(sequence)
        
        for resseq, res_name in hetatm_residues:
            ptm_position = resseq - 1  # Convert to 0-based indexing
            
            if ptm_position < len(seq_list):
                if res_name == "HYS":
                    # N-terminal histidine modification - replace N-terminal methionine with HYS
                    if ptm_position == 0 and seq_list[0] == 'M':
                        # Replace M with H (histidine) - HYS is the CCD code, but we use H for sequence
                        seq_list[0] = 'H'
                elif res_name == "2MG":
                    # 2-methylguanosine modification - replace G with modified G
                    # For simplicity, we'll keep it as G since the exact representation may vary
                    pass
                # Add more PTM types as needed
                else:
                    print(f"Warning: Unknown PTM type '{res_name}' at position {ptm_position + 1}")
        
        return ''.join(seq_list)

    @property
    def _dna_letters_3to1(self):
        """DNA three-letter to one-letter code mapping using Bio.Data.PDBData."""
        try:
            from Bio.Data.PDBData import nucleic_letters_3to1_extended
            return nucleic_letters_3to1_extended
        except ImportError:
            # Fallback if PDBData is not available
            return {
                'DA': 'A',   # deoxyadenosine
                'DT': 'T',   # deoxythymidine
                'DG': 'G',   # deoxyguanosine
                'DC': 'C',   # deoxycytidine
            }

    @property
    def _rna_letters_3to1(self):
        """RNA three-letter to one-letter code mapping using Bio.Data.PDBData."""
        try:
            from Bio.Data.PDBData import nucleic_letters_3to1_extended
            return nucleic_letters_3to1_extended
        except ImportError:
            # Fallback if PDBData is not available
            return {
                'A': 'A',   # adenosine
                'U': 'U',   # uridine
                'G': 'G',   # guanosine
                'C': 'C',   # cytidine
            }

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

    @property
    def _ligand_ccd_codes(self):
        """Common ligand CCD codes that might appear in CIF files."""
        return {
            'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'GMP', 'CTP', 'CDP', 'CMP',
            'UTP', 'UDP', 'UMP', 'NAD', 'NADH', 'FAD', 'FADH2', 'COA',
            'HEM', 'MG', 'CA', 'ZN', 'FE', 'CU', 'MN', 'K', 'NA', 'CL',
            'SO4', 'PO4', 'NO3', 'CO3', 'HCO3', 'OH', 'H2O', 'DMS', 'EDO',
            'GOL', 'PEG', 'PEO', 'MPD', 'BME', 'DTT', 'TCEP', 'GSH', 'GSSG'
        }

    def _extract_cif_chains_and_sequences_regex(self, cif_path: Path) -> List[Tuple[str, str]]:
        """
        Fallback method to extract chain IDs and sequences from a CIF file using regex.
        
        Args:
            cif_path: Path to the CIF file
            
        Returns:
            List of tuples (chain_id, sequence) for chains in the CIF file
        """
        chains_and_sequences = []
        
        with open(cif_path, 'r') as f:
            cif_content = f.read()
        
        # Extract unique chain IDs from _struct_asym table
        # Format: chain_id entity_id (e.g., "A 1")
        struct_asym_pattern = r'([A-Z]+)\s+(\d+)'
        struct_asym_matches = re.findall(struct_asym_pattern, cif_content)
        
        # Create mapping of entity_id to chain_ids
        entity_to_chains = {}
        for chain_id, entity_id in struct_asym_matches:
            entity_id = int(entity_id)
            if entity_id not in entity_to_chains:
                entity_to_chains[entity_id] = []
            entity_to_chains[entity_id].append(chain_id)
        
        # Extract sequences for each entity from _entity_poly_seq table
        # Format: entity_id num mon_id (e.g., "1 n MET 1" or "2 n DA 1")
        entity_poly_seq_pattern = r'(\d+)\s+n\s+([A-Z]{2,3})\s+(\d+)'
        entity_poly_seq_matches = re.findall(entity_poly_seq_pattern, cif_content)
        
        # Group residues by entity_id
        entity_sequences = {}
        for entity_id, mon_id, num in entity_poly_seq_matches:
            entity_id = int(entity_id)
            if entity_id not in entity_sequences:
                entity_sequences[entity_id] = []
            entity_sequences[entity_id].append((int(num), mon_id))
        
        # Extract ligand information from _pdbx_nonpoly_scheme entries
        # Look for single entries (not loops) with format:
        # _pdbx_nonpoly_scheme.asym_id L
        # _pdbx_nonpoly_scheme.mon_id ATP
        nonpoly_asym_pattern = r'_pdbx_nonpoly_scheme\.asym_id\s+([A-Z]+)'
        nonpoly_mon_pattern = r'_pdbx_nonpoly_scheme\.mon_id\s+([A-Z0-9]+)'
        
        nonpoly_asym_matches = re.findall(nonpoly_asym_pattern, cif_content)
        nonpoly_mon_matches = re.findall(nonpoly_mon_pattern, cif_content)
        
        # Create ligand chains directly
        for asym_id, mon_id in zip(nonpoly_asym_matches, nonpoly_mon_matches):
            chains_and_sequences.append((asym_id, mon_id))
        
        # Convert three-letter codes to one-letter sequences for polymer entities
        try:
            # Use comprehensive dictionaries from PDBData
            three_to_one = {}
            three_to_one.update(self._protein_letters_3to1)
            three_to_one.update(self._dna_letters_3to1)
            three_to_one.update(self._rna_letters_3to1)
        except ImportError:
            # Fallback if PDBData is not available
            from Bio.Data.IUPACData import protein_letters_3to1
            three_to_one = {**protein_letters_3to1, 'UNK': 'X'}
            # Add DNA and RNA mappings
            three_to_one.update(self._dna_letters_3to1)
            three_to_one.update(self._rna_letters_3to1)
        
        # Build sequences for each polymer entity
        for entity_id, residues in entity_sequences.items():
            # Sort by residue number
            residues.sort(key=lambda x: x[0])
            sequence = ''.join([three_to_one.get(res[1], 'X') for res in residues])
            
            # Get chain IDs for this entity - only add one entry per chain
            if entity_id in entity_to_chains:
                for chain_id in entity_to_chains[entity_id]:
                    # Check if we already have this chain_id to avoid duplicates
                    if not any(existing_chain_id == chain_id for existing_chain_id, _ in chains_and_sequences):
                        chains_and_sequences.append((chain_id, sequence))
        
        return chains_and_sequences

    def _assert_exact_chain_mapping(
        self,
        expected_sequences: List[Tuple[str, str]],
        actual_chains_and_sequences: List[Tuple[str, str]],
        *,
        context: str,
    ) -> None:
        """Assert an exact chain-id to sequence mapping, independent of file order."""
        expected_dict = dict(expected_sequences)
        actual_dict = dict(actual_chains_and_sequences)

        self.assertLen(
            expected_dict,
            len(expected_sequences),
            f"{context}: expected chain IDs must be unique",
        )
        self.assertLen(
            actual_dict,
            len(actual_chains_and_sequences),
            f"{context}: actual chain IDs must be unique",
        )

        print(f"Expected exact chain mapping for {context}: {expected_dict}")
        print(f"Actual exact chain mapping for {context}: {actual_dict}")

        self.assertEqual(
            actual_dict,
            expected_dict,
            f"{context}: exact chain mapping mismatch",
        )

    def _requires_exact_chain_mapping(self, protein_list: str) -> bool:
        """Cases where inference must preserve the explicit input chain IDs."""
        return protein_list in {
            "test_monomer_with_rna.txt",
            "test_monomer_with_dna.txt",
            "test_monomer_with_ligand.txt",
            "test_protein_with_ptms.txt",
        }

    def _check_chain_counts_and_sequences(self, protein_list: str):
        """
        Check that the predicted CIF files have the correct number of chains
        and that the sequences match the expected input sequences.
        
        Args:
            protein_list: Name of the protein list file
        """
        # Get expected sequences from input files
        expected_sequences = self._extract_expected_sequences(protein_list)
        
        print(f"\nExpected sequences: {expected_sequences}")
        
        # Find the predicted CIF file (should be in the output directory)
        result_dir = self._resolve_single_af3_result_dir()
        cif_files = list(result_dir.glob("*_model.cif"))
        if not cif_files:
            self.fail("No predicted CIF files found")
        
        # Use the first CIF file (should be the best ranked one)
        cif_path = cif_files[0]
        print(f"Checking CIF file: {cif_path}")
        
        # Extract chains and sequences from the CIF file
        actual_chains_and_sequences = self._extract_cif_chains_and_sequences(cif_path)
        
        print(f"Actual chains and sequences: {actual_chains_and_sequences}")
        
        # Check that the number of chains matches
        self.assertEqual(
            len(actual_chains_and_sequences), 
            len(expected_sequences),
            f"Expected {len(expected_sequences)} chains, but found {len(actual_chains_and_sequences)}"
        )

        if self._requires_exact_chain_mapping(protein_list):
            self._assert_exact_chain_mapping(
                expected_sequences,
                actual_chains_and_sequences,
                context=protein_list,
            )
            return

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

    def _make_af3_test_env(self) -> Dict[str, str]:
        flash_impl = self._af3_flash_attention_impl()
        env = os.environ.copy()
        env["XLA_FLAGS"] = os.getenv(
            "AF3_TEST_XLA_FLAGS",
            "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter "
            "--xla_gpu_force_compilation_parallelism=0",
        )
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        env["XLA_CLIENT_MEM_FRACTION"] = "0.95"
        env["JAX_FLASH_ATTENTION_IMPL"] = flash_impl
        if "XLA_PYTHON_CLIENT_MEM_FRACTION" in env:
            del env["XLA_PYTHON_CLIENT_MEM_FRACTION"]
        return env

    def _af3_flash_attention_impl(self) -> str:
        return os.getenv("AF3_TEST_FLASH_ATTENTION_IMPL", "xla")

    @staticmethod
    def _jax_compilation_cache_args(cache_dir: Path) -> list[str]:
        """Return the optional persistent-cache flag for functional tests.

        Some XLA/JAX combinations cannot safely populate the per-fusion cache on
        a cluster scratch filesystem. Keep the normal cached path while making
        it possible to isolate that environment failure without weakening the
        actual inference and ModelCIF assertions.
        """
        disabled = os.getenv(
            "AF3_TEST_DISABLE_JAX_COMPILATION_CACHE", ""
        ).strip().lower()
        if disabled in {"1", "true", "yes"}:
            return []
        return [f"--jax_compilation_cache_dir={cache_dir}"]

    def _require_af3_functional_environment(self) -> None:
        if not os.path.exists(DATA_DIR):
            self.skipTest(
                f"AF3 functional tests require ALPHAFOLD_DATA_DIR; missing path: {DATA_DIR}"
            )

    def _assert_af3_outputs_present(self, output_dir: Path) -> None:
        files = list(output_dir.iterdir())
        print(f"contents of {output_dir}: {[f.name for f in files]}")

        self.assertIn("TERMS_OF_USE.md", {f.name for f in files})
        ranking_files = [
            path
            for path in files
            if path.name == "ranking_scores.csv"
            or path.name.endswith("_ranking_scores.csv")
        ]
        self.assertLen(
            ranking_files,
            1,
            f"Expected one ranking-scores CSV in {output_dir}",
        )

        conf_files = [f for f in files if f.name.endswith("_confidences.json")]
        summary_conf_files = [f for f in files if f.name.endswith("_summary_confidences.json")]
        model_files = [f for f in files if f.name.endswith("_model.cif")]

        self.assertTrue(len(conf_files) > 0, f"No confidences.json files found in {output_dir}")
        self.assertTrue(len(summary_conf_files) > 0, f"No summary_confidences.json files found in {output_dir}")
        self.assertTrue(len(model_files) > 0, f"No model.cif files found in {output_dir}")
        for path in summary_conf_files:
            self.assertEqual(
                af3_gpu_checks.chain_id_problems(json.loads(path.read_text())), [], path
            )

        sample_dirs = [
            f for f in files if f.is_dir() and f.name.startswith("seed-") and "sample-" in f.name
        ]

        for sample_dir in sample_dirs:
            sample_files = list(sample_dir.iterdir())
            sample_names = {path.name for path in sample_files}
            self.assertTrue(
                any(
                    name == "confidences.json"
                    or (
                        name.endswith("_confidences.json")
                        and not name.endswith("_summary_confidences.json")
                    )
                    for name in sample_names
                ),
                sample_dir,
            )
            self.assertTrue(
                any(
                    name == "model.cif" or name.endswith("_model.cif")
                    for name in sample_names
                ),
                sample_dir,
            )
            self.assertTrue(
                any(
                    name == "summary_confidences.json"
                    or name.endswith("_summary_confidences.json")
                    for name in sample_names
                ),
                sample_dir,
            )

        with ranking_files[0].open() as f:
            lines = f.readlines()
            self.assertTrue(len(lines) > 1, "ranking_scores.csv should have header and data")
            self.assertEqual(len(lines[0].strip().split(",")), 3, "ranking_scores.csv should have 3 columns")

            seeds_in_csv = {ln.strip().split(",")[0] for ln in lines[1:] if ln.strip()}

            def _seed_from_dirname(name: str) -> str:
                try:
                    part = name.split("seed-")[1]
                    return part.split("_")[0]
                except Exception:
                    return ""

            sample_dirs_for_this_run = [d for d in sample_dirs if _seed_from_dirname(d.name) in seeds_in_csv]
            expected_sample_dirs = len(lines) - 1
            self.assertEqual(
                len(sample_dirs_for_this_run), expected_sample_dirs,
                f"Expected {expected_sample_dirs} sample directories, found {len(sample_dirs_for_this_run)}"
            )

            for i, line in enumerate(lines[1:], 1):
                parts = line.strip().split(",")
                self.assertEqual(len(parts), 3, f"Line {i+1} should have 3 columns: seed,sample,ranking_score")
                try:
                    int(parts[0])
                    int(parts[1])
                    float(parts[2])
                except ValueError:
                    self.fail(f"Line {i+1} has invalid format: {line.strip()}")

            print(f"✓ Verified ranking_scores.csv has correct format with {len(lines)-1} entries")

    def _resolve_single_af3_result_dir(self) -> Path:
        """Return the actual AF3 result directory for single-job tests."""
        if (self.output_dir / "ranking_scores.csv").exists():
            return self.output_dir

        candidate_dirs = [
            path
            for path in self.output_dir.iterdir()
            if path.is_dir() and (path / "ranking_scores.csv").exists()
        ]
        if len(candidate_dirs) == 1:
            print(f"Resolved nested AF3 result dir: {candidate_dirs[0]}")
            return candidate_dirs[0]

        return self.output_dir

    # ---------------- assertions reused by all subclasses ----------------- #
    def _runCommonTests(self, res: subprocess.CompletedProcess):
        print(res.stdout)
        print(res.stderr)
        self.assertEqual(res.returncode, 0, "sub-process failed")

        self._assert_af3_outputs_present(self._resolve_single_af3_result_dir())

    # convenience builder
    def _args(self, *, plist, script):
        flash_impl = self._af3_flash_attention_impl()
        # Determine mode from protein list name
        if "homooligomer" in plist:
            mode = "homo-oligomer"
        else:
            mode = "custom"
            
        if script == "run_structure_prediction.py":
            # Format from run_multimer_jobs.py input to run_structure_prediction.py input
            specifications = generate_fold_specifications(
                input_files=[str(self.test_protein_lists_dir / plist)],
                delimiter="+",
                exclude_permutations=True,
            )
            formatted_input_lines = [
                spec.replace(",", ":").replace(";", "+")
                for spec in specifications
                if spec.strip()
            ]
            formatted_input = formatted_input_lines[0] if formatted_input_lines else ""
            args = [
                sys.executable,
                str(self.script_single),
                f"--input={formatted_input}",
                f"--output_directory={self.output_dir}",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={self.test_features_dir}",
                "--fold_backend=alphafold3",
                f"--flash_attention_implementation={flash_impl}",
            ]
            
            # Add special arguments for multi_seeds_samples test
            if "multi_seeds_samples" in plist:
                args.extend([
                    "--num_seeds=3",
                    "--num_diffusion_samples=4",
                ])
            
            return args
        elif script == "run_multimer_jobs.py":
            args = [
                sys.executable,
                str(self.script_multimer),
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_dir={DATA_DIR}",
                f"--monomer_objects_dir={self.test_features_dir}",
                "--job_index=1",
                f"--output_path={self.output_dir}",
                f"--mode={mode}",
                "--oligomer_state_file"
                if mode == "homo-oligomer"
                else "--protein_lists"
                + f"={self.test_protein_lists_dir / plist}",
                # Ensure AF3 backend and keep runtime small
                "--fold_backend=alphafold3",
                f"--flash_attention_implementation={flash_impl}",
                "--num_diffusion_samples=1",
            ]
            return args


# --------------------------------------------------------------------------- #
#                      backend-only AF3 preparation tests                      #
# --------------------------------------------------------------------------- #
class _BackendOnlyTestBase(_TestBase):
    """Backend-only AF3 preparation tests that do not run model inference."""

    @classmethod
    def setUpClass(cls):
        parameterized.TestCase.setUpClass()
        if cls.use_temp_dir:
            cls.base_output_dir = Path(tempfile.mkdtemp(prefix="af3_backend_test_"))
        else:
            cls.base_output_dir = Path("test/test_data/predictions/af3_backend")
            if cls.base_output_dir.exists():
                try:
                    shutil.rmtree(cls.base_output_dir)
                except (PermissionError, OSError) as e:
                    print(
                        "Warning: Could not remove existing output directory "
                        f"{cls.base_output_dir}: {e}"
                    )
            cls.base_output_dir.mkdir(parents=True, exist_ok=True)


class TestAlphaFold3BackendRegressions(_BackendOnlyTestBase):
    """AF3 input-construction regressions; these tests do not assert end-to-end ipTM quality."""

    ISSUE_588_IDS = ("A0ABD7FQG0", "P18004")

    def _require_issue_588_mmseqs_environment(self) -> None:
        skip_reason = _mmseqs_functional_test_skip_reason()
        if skip_reason:
            self.skipTest(skip_reason)
        for protein_id in self.ISSUE_588_IDS:
            fasta_path = self.test_data_dir / "fastas" / f"{protein_id}.fasta"
            self.assertTrue(
                fasta_path.is_file(),
                f"Missing FASTA fixture {fasta_path}",
            )

    def _generate_issue_588_precomputed_mmseq_features(self, env: Dict[str, str]) -> Path:
        source_dir = self.output_dir / "issue_588_mmseq_source_features"
        precomputed_dir = self.output_dir / "issue_588_mmseq_precomputed_features"
        source_dir.mkdir(parents=True, exist_ok=True)
        precomputed_dir.mkdir(parents=True, exist_ok=True)
        fasta_paths = ",".join(
            str(self.test_data_dir / "fastas" / f"{protein_id}.fasta")
            for protein_id in self.ISSUE_588_IDS
        )

        source_res = subprocess.run(
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
            ],
            capture_output=True,
            text=True,
            env=env,
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

        precomputed_res = subprocess.run(
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
            ],
            capture_output=True,
            text=True,
            env=env,
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

    def test_issue_588_precomputed_mmseqs_msas_preserve_af3_species_pairing(self):
        """Precomputed MMseq A3Ms should preserve recovered identifiers and AF3 species pairing."""
        from alphapulldown.folding_backend.alphafold3_backend import process_fold_input

        self._require_issue_588_mmseqs_environment()
        env = os.environ.copy()
        feature_dir = self._generate_issue_588_precomputed_mmseq_features(env)

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

        fold_input_obj = self._prepare_fold_input(
            fold_spec="A0ABD7FQG0+P18004",
            feature_dir=feature_dir,
            debug_msas=True,
        )
        job_name = fold_input_obj.sanitised_name()
        summary_path = self.output_dir / f"{job_name}_af2_to_af3_translation_summary.json"
        self.assertTrue(summary_path.is_file(), f"Missing translation summary {summary_path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(
            summary["translation_modes"],
            ["af3_species_pairing_from_af2_individual_msas"],
        )
        self.assertTrue(summary["paired_rows_valid"])
        self.assertTrue(summary["unpaired_rows_valid"])
        self.assertLen(summary["chains"], 2)
        for chain_summary in summary["chains"]:
            self.assertGreater(chain_summary["paired_msa_row_count"], 0)
            self.assertGreater(chain_summary["unpaired_msa_row_count"], 0)
            self.assertGreater(chain_summary["paired_species_identifier_count"], 0)

        process_fold_input(
            fold_input=fold_input_obj,
            model_runner=None,
            output_dir=str(self.output_dir),
            buckets=(512,),
        )
        input_json = self.output_dir / f"{job_name}_data.json"
        written = json.loads(input_json.read_text(encoding="utf-8"))
        protein_entries = _protein_entries_from_af3_input(written)
        self.assertLen(protein_entries, 2)
        for protein_entry in protein_entries:
            self.assertEqual(
                _a3m_query_sequence(protein_entry["pairedMsa"]),
                protein_entry["sequence"],
            )
            self.assertEqual(
                _a3m_query_sequence(protein_entry["unpairedMsa"]),
                protein_entry["sequence"],
            )

class TestAlphaFold3MmseqsIssue588Inference(_TestBase):
    """Opt-in AF3 end-to-end smoke test for freshly regenerated mmseq AF2 features."""

    ISSUE_588_IDS = ("A0ABD7FQG0", "P18004")

    def _require_mmseqs_functional_environment(self) -> None:
        self._require_af3_functional_environment()
        skip_reason = _mmseqs_functional_test_skip_reason()
        if skip_reason:
            self.skipTest(skip_reason)
        for protein_id in self.ISSUE_588_IDS:
            fasta_path = self.test_data_dir / "fastas" / f"{protein_id}.fasta"
            self.assertTrue(
                fasta_path.is_file(),
                f"Missing FASTA fixture {fasta_path}",
            )

    def _generate_issue_588_mmseq_features(self, env: Dict[str, str]) -> Path:
        feature_dir = self.output_dir / "issue_588_mmseq_features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        fasta_paths = ",".join(
            str(self.test_data_dir / "fastas" / f"{protein_id}.fasta")
            for protein_id in self.ISSUE_588_IDS
        )
        res = subprocess.run(
            [
                sys.executable,
                str(self.script_create_features),
                f"--fasta_paths={fasta_paths}",
                f"--output_dir={feature_dir}",
                f"--data_dir={DATA_DIR}",
                "--max_template_date=2024-05-02",
                "--use_mmseqs2=True",
                "--data_pipeline=alphafold2",
                "--save_msa_files=True",
                "--compress_features=True",
                "--skip_existing=False",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(
            res.returncode,
            0,
            f"MMseqs feature generation failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
        )
        return feature_dir

    def test_issue_588_mmseqs_af2_features_enable_af3_species_pairing_inference(self):
        self._require_mmseqs_functional_environment()
        env = self._make_af3_test_env()
        feature_dir = self._generate_issue_588_mmseq_features(env)

        for protein_id in self.ISSUE_588_IDS:
            self.assertTrue(
                (feature_dir / f"{protein_id}.a3m").is_file(),
                f"Expected MMseq A3M {feature_dir / f'{protein_id}.a3m'} to be created.",
            )
            self.assertTrue(
                (feature_dir / f"{protein_id}.pkl.xz").is_file(),
                f"Expected compressed feature pickle {feature_dir / f'{protein_id}.pkl.xz'} to be created.",
            )
            feature_dict = _load_feature_dict(feature_dir / f"{protein_id}.pkl.xz")
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

        flash_impl = self._af3_flash_attention_impl()
        res = subprocess.run(
            [
                sys.executable,
                str(self.script_single),
                "--input=A0ABD7FQG0+P18004",
                f"--output_directory={self.output_dir}",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={feature_dir}",
                "--fold_backend=alphafold3",
                f"--flash_attention_implementation={flash_impl}",
                "--num_diffusion_samples=1",
                "--random_seed=42",
                "--debug_msas",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self._runCommonTests(res)

        result_dir = self._resolve_single_af3_result_dir()
        summary_paths = sorted(
            result_dir.glob("*_af2_to_af3_translation_summary.json")
        )
        self.assertLen(summary_paths, 1)
        summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
        self.assertEqual(
            summary["translation_modes"],
            ["af3_species_pairing_from_af2_individual_msas"],
        )
        self.assertTrue(summary["paired_rows_valid"])
        self.assertTrue(summary["unpaired_rows_valid"])
        for chain_summary in summary["chains"]:
            self.assertGreater(chain_summary["paired_msa_row_count"], 0)
            self.assertGreater(chain_summary["unpaired_msa_row_count"], 0)
            self.assertGreater(chain_summary["paired_species_identifier_count"], 0)

        confidence_files = sorted(result_dir.glob("*_summary_confidences.json"))
        self.assertLen(confidence_files, 1)
        confidence_payload = json.loads(
            confidence_files[0].read_text(encoding="utf-8")
        )
        self.assertIn("iptm", confidence_payload)
        self.assertGreater(
            confidence_payload["iptm"],
            0.6,
            f"Expected AF3 ipTM > 0.6, got {confidence_payload['iptm']}",
        )

    def test_issue_588_mmseqs_af2_features_enable_af3_species_pairing_trimer_inference(self):
        """AF3 should accept trimer jobs built from AF2/mmseqs2 pkl features and report effective pairing."""
        self._require_mmseqs_functional_environment()
        env = self._make_af3_test_env()
        feature_dir = self._generate_issue_588_mmseq_features(env)

        flash_impl = self._af3_flash_attention_impl()
        res = subprocess.run(
            [
                sys.executable,
                str(self.script_single),
                "--input=A0ABD7FQG0+P18004+A0ABD7FQG0",
                f"--output_directory={self.output_dir}",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={feature_dir}",
                "--fold_backend=alphafold3",
                f"--flash_attention_implementation={flash_impl}",
                "--num_diffusion_samples=1",
                "--random_seed=42",
                "--debug_msas",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self._runCommonTests(res)

        result_dir = self._resolve_single_af3_result_dir()
        summary_paths = sorted(
            result_dir.glob("*_af2_to_af3_translation_summary.json")
        )
        self.assertLen(summary_paths, 1)
        summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
        self.assertEqual(
            summary["translation_modes"],
            ["af3_species_pairing_from_af2_individual_msas"],
        )
        self.assertTrue(summary["paired_rows_valid"])
        self.assertTrue(summary["unpaired_rows_valid"])
        self.assertGreater(summary["translated_paired_input_row_count"], 0)
        self.assertGreater(summary["paired_row_count"], 0)
        self.assertGreaterEqual(
            summary["translated_paired_input_row_count"],
            summary["paired_row_count"],
        )
        histogram = summary["effective_paired_row_histogram_by_num_chains"]
        self.assertTrue(histogram)
        self.assertGreaterEqual(max(int(key) for key in histogram), 2)
        self.assertLen(summary["chains"], 3)
        for chain_summary in summary["chains"]:
            self.assertGreater(chain_summary["paired_msa_row_count"], 0)
            self.assertGreater(chain_summary["unpaired_msa_row_count"], 0)
            self.assertGreater(chain_summary["effective_paired_msa_row_count"], 0)

        input_json_paths = sorted(result_dir.glob("*_data.json"))
        self.assertLen(input_json_paths, 1)
        written = json.loads(input_json_paths[0].read_text(encoding="utf-8"))
        protein_entries = _protein_entries_from_af3_input(written)
        self.assertLen(protein_entries, 2)
        all_chain_ids = []
        for protein_entry in protein_entries:
            entry_ids = protein_entry["id"]
            if isinstance(entry_ids, str):
                entry_ids = [entry_ids]
            all_chain_ids.extend(entry_ids)
            self.assertEqual(
                _a3m_query_sequence(protein_entry["pairedMsa"]),
                protein_entry["sequence"],
            )
            self.assertEqual(
                _a3m_query_sequence(protein_entry["unpairedMsa"]),
                protein_entry["sequence"],
            )
        self.assertCountEqual(all_chain_ids, ["A", "B", "C"])


class TestAlphaFold3MetadataEndToEnd(_TestBase):
    """Real databases/GPU coverage for AF2, AF3, and mixed provenance paths."""

    PROTEIN_ID = "A0A024R1R8"
    REQUIRED_AF3_DATABASES = (
        "bfd-first_non_consensus_sequences.fasta",
        "mgy_clusters_2022_05.fa",
        "uniprot_all_2021_04.fa",
        "uniref90_2022_05.fa",
        "pdb_seqres_2022_09_28.fasta",
        "mmcif_files",
    )

    def _require_metadata_e2e_environment(self) -> None:
        self._require_af3_functional_environment()
        missing_databases = [
            str(AF3_DATABASE_DIR / relative_path)
            for relative_path in self.REQUIRED_AF3_DATABASES
            if not (AF3_DATABASE_DIR / relative_path).exists()
        ]
        if missing_databases:
            self.skipTest(
                "AF3 metadata end-to-end test requires the AF3 databases; "
                f"missing: {missing_databases}"
            )
        if importlib.util.find_spec("modelcif") is None:
            # A test-only dependency: the AF3 runtime image leaves it out.
            self.skipTest(
                "AF3 metadata end-to-end test reads the ModelCIF output with "
                "modelcif; install modelcif>=1.6 alongside the tests"
            )

        model_files = list(Path(DATA_DIR).glob("af3.bin*"))
        self.assertTrue(model_files, f"No AF3 model weights found in {DATA_DIR}")
        self.assertTrue(
            (self.test_fastas_dir / f"{self.PROTEIN_ID}.fasta").is_file()
        )
        self.assertTrue(
            (self.test_features_dir / f"{self.PROTEIN_ID}.pkl").is_file()
        )
        metadata_matches = list(
            self.test_features_dir.glob(
                f"{self.PROTEIN_ID}_feature_metadata_*.json*"
            )
        )
        self.assertTrue(
            metadata_matches,
            f"No AF2 feature metadata sidecar for {self.PROTEIN_ID}",
        )

    def _run_e2e_command(
        self,
        args: list[str],
        *,
        env: Dict[str, str],
        label: str,
    ) -> None:
        print(f"\n=== {label} ===", flush=True)
        print(" ".join(str(arg) for arg in args), flush=True)
        started = time.monotonic()
        result = subprocess.run(args, text=True, env=env, check=False)
        elapsed = time.monotonic() - started
        print(
            f"=== {label}: returncode={result.returncode}, "
            f"elapsed_seconds={elapsed:.1f} ===",
            flush=True,
        )
        self.assertEqual(result.returncode, 0, f"{label} failed")

    def _write_native_af3_input(self, path: Path) -> str:
        fasta_lines = (
            self.test_fastas_dir / f"{self.PROTEIN_ID}.fasta"
        ).read_text(encoding="utf-8").splitlines()
        sequence = "".join(
            line.strip() for line in fasta_lines if not line.startswith(">")
        )
        self.assertTrue(sequence)
        raw_input = {
            "dialect": "alphafold3",
            "version": 1,
            "name": "metadata_native_features",
            "modelSeeds": [42],
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": sequence,
                        "modifications": [],
                        "unpairedMsa": None,
                        "pairedMsa": None,
                        "templates": None,
                    }
                }
            ],
            "bondedAtomPairs": [],
            "userCCD": None,
        }
        path.write_text(json.dumps(raw_input, indent=2) + "\n", encoding="utf-8")
        parsed = folding_input.Input.from_json(path.read_text(encoding="utf-8"))
        self.assertEqual(parsed.name, "metadata_native_features")
        return sequence

    def _assert_feature_metadata(
        self,
        feature_path: Path,
        *,
        expected_count: int,
        expected_pipeline: str | None = None,
    ) -> list[dict[str, Any]]:
        payload = _load_json_payload(feature_path)
        # This is the compatibility boundary: the unmodified AF3 parser must
        # accept the same JSON which carries AlphaPulldown metadata.
        folding_input.Input.from_json(json.dumps(payload))
        metadata = extract_metadata_from_af3_json(payload)
        self.assertLen(metadata, expected_count)
        if expected_pipeline is not None:
            self.assertEqual(
                {record.get("other", {}).get("data_pipeline") for record in metadata},
                {expected_pipeline},
            )
        return metadata

    def _assert_modelcif_files(
        self,
        output_dir: Path,
        *,
        required_software: set[str] = frozenset(),
        forbidden_software: set[str] = frozenset(),
        required_databases: set[str] = frozenset(),
        forbidden_databases: set[str] = frozenset(),
    ) -> dict[str, list[str]]:
        import modelcif.reader

        cif_paths = sorted(output_dir.glob("*_model.cif"))
        for sample_dir in sorted(output_dir.glob("seed-*_sample-*")):
            cif_paths.extend(
                sorted(
                    path
                    for path in sample_dir.iterdir()
                    if path.name == "model.cif" or path.name.endswith("_model.cif")
                )
            )
        self.assertGreaterEqual(
            len(cif_paths), 2, f"Expected best and sample ModelCIF files in {output_dir}"
        )
        observed_software: set[str] = set()
        observed_databases: set[str] = set()
        for cif_path in cif_paths:
            cif_text = cif_path.read_text(encoding="utf-8")
            self.assertNotIn(
                AF3_METADATA_MARKER,
                cif_text,
                f"Transport metadata leaked into final ModelCIF {cif_path}",
            )
            cif = af3_mmcif.from_string(cif_text)
            software = {str(value) for value in cif.get("_software.name", ())}
            databases = {
                str(value) for value in cif.get("_ma_data_ref_db.name", ())
            }
            parameter_names = {
                str(value)
                for value in cif.get("_ma_software_parameter.name", ())
            }
            parameter_values = [
                str(value)
                for value in cif.get("_ma_software_parameter.value", ())
            ]
            observed_software.update(software)
            observed_databases.update(databases)
            self.assertTrue(required_software.issubset(software), cif_path)
            self.assertTrue(forbidden_software.isdisjoint(software), cif_path)
            self.assertTrue(required_databases.issubset(databases), cif_path)
            self.assertTrue(forbidden_databases.isdisjoint(databases), cif_path)
            self.assertTrue(
                {
                    "--?",
                    "--data_dir",
                    "--fasta_paths",
                    "--output_dir",
                    "--hhblits_binary_path",
                }.isdisjoint(parameter_names),
                cif_path,
            )
            self.assertFalse(
                any(
                    local_prefix in value
                    for value in parameter_values
                    for local_prefix in ("/g/", "/home/", "/scratch/")
                ),
                cif_path,
            )
            if required_software or required_databases:
                self.assertTrue(
                    cif.get("_ma_protocol_step.software_group_id", ()), cif_path
                )
            with cif_path.open("rt", encoding="utf-8") as handle:
                systems = modelcif.reader.read(handle)
            self.assertTrue(systems, f"python-modelcif could not read {cif_path}")
        return {
            "software": sorted(observed_software),
            "databases": sorted(observed_databases),
        }

    def _run_alphapulldown_inference(
        self,
        *,
        input_spec: str,
        output_dir: Path,
        feature_directories: list[Path],
        cache_dir: Path,
        env: Dict[str, str],
        label: str,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._run_e2e_command(
            [
                sys.executable,
                str(self.script_single),
                f"--input={input_spec}",
                f"--output_directory={output_dir}",
                f"--data_directory={DATA_DIR}",
                "--features_directory="
                + ",".join(str(path) for path in feature_directories),
                "--fold_backend=alphafold3",
                f"--flash_attention_implementation={self._af3_flash_attention_impl()}",
                "--num_diffusion_samples=1",
                "--num_recycles=1",
                "--buckets=256",
                "--random_seed=42",
                *self._jax_compilation_cache_args(cache_dir),
                "--convert_to_modelcif=True",
                "--storage_mode=vanilla",
            ],
            env=env,
            label=label,
        )
        self._assert_af3_outputs_present(output_dir)
        return output_dir

    def test_real_feature_generation_inference_and_modelcif_matrix(self):
        self._require_metadata_e2e_environment()
        env = self._make_af3_test_env()
        cache_dir = self.output_dir / "jax_cache"
        cache_dir.mkdir()
        ap_feature_dir = self.output_dir / "features_alphapulldown_af3"
        vanilla_feature_root = self.output_dir / "features_vanilla_af3"
        ap_feature_dir.mkdir()
        vanilla_feature_root.mkdir()
        persistent_cache_value = os.getenv("AF3_METADATA_E2E_CACHE_DIR")
        persistent_cache_dir = (
            Path(persistent_cache_value).resolve()
            if persistent_cache_value
            else None
        )
        if persistent_cache_dir is not None:
            persistent_cache_dir.mkdir(parents=True, exist_ok=True)

        raw_input_path = self.output_dir / "metadata_native_raw.json"
        self._write_native_af3_input(raw_input_path)

        ap_feature_path = ap_feature_dir / f"{self.PROTEIN_ID}_af3_input.json"
        cached_ap_feature_path = (
            persistent_cache_dir / ap_feature_path.name
            if persistent_cache_dir is not None
            else None
        )
        if cached_ap_feature_path is not None and cached_ap_feature_path.is_file():
            print(
                f"Restoring validated AlphaPulldown AF3 features from "
                f"{cached_ap_feature_path}",
                flush=True,
            )
            shutil.copy2(cached_ap_feature_path, ap_feature_path)
        else:
            # Generate AF3 features through AlphaPulldown using the real databases.
            self._run_e2e_command(
                [
                    sys.executable,
                    str(self.script_create_features),
                    f"--fasta_paths={self.test_fastas_dir / f'{self.PROTEIN_ID}.fasta'}",
                    f"--output_dir={ap_feature_dir}",
                    f"--data_dir={AF3_DATABASE_DIR}",
                    "--max_template_date=2026-08-19",
                    "--data_pipeline=alphafold3",
                    "--skip_existing=False",
                ],
                env=env,
                label="AlphaPulldown AF3 feature generation",
            )
        self.assertTrue(ap_feature_path.is_file())
        self.assertFalse(
            list(ap_feature_dir.glob(f"{self.PROTEIN_ID}_feature_metadata_*.json*")),
            "AF3 metadata must be embedded, not written as a sidecar",
        )
        ap_metadata = self._assert_feature_metadata(
            ap_feature_path,
            expected_count=1,
            expected_pipeline="alphafold3",
        )
        self.assertTrue(
            {"UniRef90", "MGnify", "PDB mmCIF"}.issubset(
                ap_metadata[0].get("databases", {})
            )
        )
        self.assertTrue(
            {"NT-RNA", "Rfam", "RNAcentral", "ColabFold"}.isdisjoint(
                ap_metadata[0].get("databases", {})
            )
        )
        if cached_ap_feature_path is not None and not cached_ap_feature_path.exists():
            shutil.copy2(ap_feature_path, cached_ap_feature_path)
            print(
                f"Cached validated AlphaPulldown AF3 features at "
                f"{cached_ap_feature_path}",
                flush=True,
            )

        vanilla_feature_path = (
            vanilla_feature_root
            / "metadata_native_features"
            / "metadata_native_features_data.json"
        )
        cached_vanilla_feature_path = (
            persistent_cache_dir / vanilla_feature_path.name
            if persistent_cache_dir is not None
            else None
        )
        if (
            cached_vanilla_feature_path is not None
            and cached_vanilla_feature_path.is_file()
        ):
            print(
                f"Restoring validated vanilla AF3 features from "
                f"{cached_vanilla_feature_path}",
                flush=True,
            )
            vanilla_feature_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_vanilla_feature_path, vanilla_feature_path)
        else:
            # Generate an independent, unannotated feature JSON with vanilla AF3.
            self._run_e2e_command(
                [
                    sys.executable,
                    str(REPO_ROOT / "alphafold3" / "run_alphafold.py"),
                    f"--json_path={raw_input_path}",
                    f"--output_dir={vanilla_feature_root}",
                    f"--model_dir={DATA_DIR}",
                    f"--db_dir={AF3_DATABASE_DIR}",
                    "--max_template_date=2026-08-19",
                    "--run_data_pipeline",
                    "--norun_inference",
                    "--force_output_dir",
                ],
                env=env,
                label="vanilla AF3 feature generation",
            )
        self.assertTrue(vanilla_feature_path.is_file(), vanilla_feature_path)
        self._assert_feature_metadata(vanilla_feature_path, expected_count=0)
        if (
            cached_vanilla_feature_path is not None
            and not cached_vanilla_feature_path.exists()
        ):
            shutil.copy2(vanilla_feature_path, cached_vanilla_feature_path)
            print(
                f"Cached validated vanilla AF3 features at "
                f"{cached_vanilla_feature_path}",
                flush=True,
            )

        # Vanilla AF3 must run inference directly from the metadata-bearing AP JSON.
        vanilla_inference_root = self.output_dir / "vanilla_on_ap_features"
        self._run_e2e_command(
            [
                sys.executable,
                str(REPO_ROOT / "alphafold3" / "run_alphafold.py"),
                f"--json_path={ap_feature_path}",
                f"--output_dir={vanilla_inference_root}",
                f"--model_dir={DATA_DIR}",
                "--norun_data_pipeline",
                "--run_inference",
                "--force_output_dir",
                f"--flash_attention_implementation={self._af3_flash_attention_impl()}",
                "--num_diffusion_samples=1",
                "--num_recycles=1",
                "--buckets=256",
                *self._jax_compilation_cache_args(cache_dir),
            ],
            # Upstream AF3 also writes XLA's autotune caches into the cache dir,
            # which fails on cluster scratch; AlphaPulldown turns them off itself.
            env={**env, "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"},
            label="vanilla AF3 inference on AlphaPulldown AF3 features",
        )
        vanilla_inference_dir = vanilla_inference_root / self.PROTEIN_ID
        self.assertTrue(vanilla_inference_dir.is_dir(), vanilla_inference_dir)
        self._assert_af3_outputs_present(vanilla_inference_dir)
        self._assert_feature_metadata(
            vanilla_inference_dir / f"{self.PROTEIN_ID}_data.json",
            expected_count=1,
            expected_pipeline="alphafold3",
        )
        vanilla_cif_report = self._assert_modelcif_files(
            vanilla_inference_dir,
            forbidden_software={"AlphaPulldown", "AlphaFold 2"},
        )

        feature_directories = [
            self.test_features_dir,
            ap_feature_dir,
            vanilla_feature_path.parent,
        ]
        cases = {
            "af2_features": {
                "input": self.PROTEIN_ID,
                "metadata_count": 1,
                "software": {"AlphaPulldown", "AlphaFold 2"},
                "databases": set(),
            },
            "vanilla_af3_features": {
                "input": str(vanilla_feature_path),
                "metadata_count": 0,
                "software": set(),
                "databases": set(),
            },
            "alphapulldown_af3_features": {
                "input": str(ap_feature_path),
                "metadata_count": 1,
                "software": {"AlphaPulldown"},
                "databases": {"UniRef90", "MGnify", "PDB mmCIF"},
            },
            "mixed_af2_and_af3_features": {
                "input": f"{self.PROTEIN_ID}+{ap_feature_path}",
                "metadata_count": 2,
                "software": {"AlphaPulldown", "AlphaFold 2"},
                # The AF2 chain brings its own recorded databases: the fixture's
                # metadata lists ColabFold.
                "databases": {"UniRef90", "MGnify", "PDB mmCIF", "ColabFold"},
            },
        }
        report: dict[str, Any] = {
            "alphapulldown_feature_metadata_database_names": sorted(
                ap_metadata[0].get("databases", {})
            ),
            "vanilla_on_alphapulldown_features": vanilla_cif_report,
            "alphapulldown_backend": {},
        }
        for case_name, case in cases.items():
            case_output_dir = self.output_dir / case_name
            self._run_alphapulldown_inference(
                input_spec=case["input"],
                output_dir=case_output_dir,
                feature_directories=feature_directories,
                cache_dir=cache_dir,
                env=env,
                label=f"AlphaPulldown backend inference: {case_name}",
            )
            prepared_inputs = sorted(case_output_dir.glob("*_data.json"))
            self.assertLen(prepared_inputs, 1)
            self._assert_feature_metadata(
                prepared_inputs[0], expected_count=case["metadata_count"]
            )
            report["alphapulldown_backend"][case_name] = self._assert_modelcif_files(
                case_output_dir,
                required_software=case["software"],
                forbidden_software=(
                    {"AlphaPulldown", "AlphaFold 2"}
                    if case_name == "vanilla_af3_features"
                    else set()
                ),
                required_databases=case["databases"],
                forbidden_databases=(
                    {"NT-RNA", "Rfam", "RNAcentral", "ColabFold"}
                    if case_name == "alphapulldown_af3_features"
                    else {"NT-RNA", "Rfam", "RNAcentral"}
                    if case_name == "mixed_af2_and_af3_features"
                    else set()
                ),
            )

        print("\n=== AF3 metadata end-to-end report ===", flush=True)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)


# --------------------------------------------------------------------------- #
#                        parameterised "run mode" tests                       #
# --------------------------------------------------------------------------- #
class TestAlphaFold3RunModes(_TestBase):
    def test_af3_predicts_json_feature_ranges_as_one_gapped_chain(self):
        """Run AF3 on a Snakefile-style AF3 JSON feature input with explicit ranges."""
        self._require_af3_functional_environment()
        env = self._make_af3_test_env()
        flash_impl = self._af3_flash_attention_impl()
        feature_dir = self.test_features_dir / "af3_features" / "protein"

        res = subprocess.run(
            [
                sys.executable,
                str(self.script_single),
                "--input=A0A024R1R8_af3_input.json:2-5:8-10",
                f"--output_directory={self.output_dir}",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={feature_dir}",
                "--fold_backend=alphafold3",
                f"--flash_attention_implementation={flash_impl}",
                "--num_diffusion_samples=1",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self._runCommonTests(res)

        json_sequences = self._get_sequence_from_json(
            "af3_features/protein/A0A024R1R8_af3_input.json"
        )
        self.assertLen(json_sequences, 1)
        full_sequence = json_sequences[0][1]
        expected_sequence = full_sequence[1:5] + full_sequence[7:10]
        expected_residue_ids = [2, 3, 4, 5, 8, 9, 10]

        result_dir = self._resolve_single_af3_result_dir()
        cif_files = list(result_dir.glob("*_model.cif"))
        self.assertTrue(cif_files, f"No predicted CIF files found in {result_dir}")

        actual_chains_and_sequences = self._extract_cif_chains_and_sequences(cif_files[0])
        actual_sequences = [sequence for _, sequence in actual_chains_and_sequences]
        actual_residue_numbers = self._extract_cif_chain_residue_numbers(cif_files[0])

        self.assertEqual(actual_sequences, [expected_sequence])
        self.assertEqual(actual_residue_numbers, [("A", expected_residue_ids)])

        print("✓ AF3 prediction keeps AF3 JSON feature ranges as one gapped chain")

    def test_af3_predicts_two_out_of_order_gapped_copies_as_two_chains(self):
        """Run AF3 inference and ensure copied out-of-order gapped regions remain two chains."""
        self._require_af3_functional_environment()
        env = self._make_af3_test_env()
        flash_impl = self._af3_flash_attention_impl()

        res = subprocess.run(
            [
                sys.executable,
                str(self.script_single),
                "--input=A0A075B6L2:2:8-10:2-5",
                f"--output_directory={self.output_dir}",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={self.test_features_dir}",
                "--fold_backend=alphafold3",
                f"--flash_attention_implementation={flash_impl}",
                "--num_diffusion_samples=1",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self._runCommonTests(res)

        expected_regions = [(8, 10), (2, 5)]
        expected_sequence = "".join(
            self._get_region_sequences("A0A075B6L2", expected_regions)
        )
        expected_residue_ids = [8, 9, 10, 2, 3, 4, 5]

        result_dir = self._resolve_single_af3_result_dir()
        cif_files = list(result_dir.glob("*_model.cif"))
        self.assertTrue(cif_files, f"No predicted CIF files found in {result_dir}")

        actual_chains_and_sequences = self._extract_cif_chains_and_sequences(cif_files[0])
        residue_numbers_by_chain = dict(self._extract_cif_chain_residue_numbers(cif_files[0]))

        self.assertEqual(
            [sequence for _, sequence in actual_chains_and_sequences],
            [expected_sequence, expected_sequence],
        )
        self.assertEqual(
            [chain_id for chain_id, _ in actual_chains_and_sequences],
            ["A", "B"],
        )
        self.assertEqual(residue_numbers_by_chain["A"], expected_residue_ids)
        self.assertEqual(residue_numbers_by_chain["B"], expected_residue_ids)

        print("✓ AF3 prediction keeps two copied out-of-order gapped regions as two chains")

    def _assert_chopped_regions_keep_their_residue_ids(self):
        """The chopped chain is one gapped chain that keeps each region's residue numbers."""
        chopped_sequence = "".join(
            self._get_region_sequences("A0A075B6L2", [(1, 10), (2, 5), (12, 15)])
        )
        expected_residue_ids = (
            list(range(1, 11)) + ["2A", "3A", "4A", "5A"] + list(range(12, 16))
        )
        result_dir = self._resolve_single_af3_result_dir()
        cif_files = list(result_dir.glob("*_model.cif"))
        self.assertTrue(cif_files, f"No predicted CIF files found in {result_dir}")
        sequences_by_chain = dict(self._extract_cif_chains_and_sequences(cif_files[0]))
        residue_numbers_by_chain = dict(self._extract_cif_chain_residue_numbers(cif_files[0]))
        chopped_chain_ids = [
            chain_id
            for chain_id, sequence in sequences_by_chain.items()
            if sequence == chopped_sequence
        ]
        self.assertLen(chopped_chain_ids, 1)
        self.assertEqual(residue_numbers_by_chain[chopped_chain_ids[0]], expected_residue_ids)

    def _assert_multi_seeds_samples_outputs(self):
        """Every seed and diffusion sample is written and ranked."""
        result_dir = self._resolve_single_af3_result_dir()
        files = list(result_dir.iterdir())

        self.assertIn("TERMS_OF_USE.md", {f.name for f in files})
        self.assertIn("ranking_scores.csv", {f.name for f in files})

        conf_files = [f for f in files if f.name.endswith("_confidences.json")]
        summary_conf_files = [f for f in files if f.name.endswith("_summary_confidences.json")]
        model_files = [f for f in files if f.name.endswith("_model.cif")]

        self.assertTrue(len(conf_files) > 0, "No confidences.json files found")
        self.assertTrue(len(summary_conf_files) > 0, "No summary_confidences.json files found")
        self.assertTrue(len(model_files) > 0, "No model.cif files found")

        sample_dirs = [f for f in files if f.is_dir() and f.name.startswith("seed-")]
        self.assertEqual(
            len(sample_dirs),
            12,
            f"Expected 12 sample directories, found {len(sample_dirs)}",
        )

        for sample_dir in sample_dirs:
            sample_files = list(sample_dir.iterdir())
            self.assertIn("confidences.json", {f.name for f in sample_files})
            self.assertIn("model.cif", {f.name for f in sample_files})
            self.assertIn("summary_confidences.json", {f.name for f in sample_files})

        with open(result_dir / "ranking_scores.csv") as f:
            lines = f.readlines()
            self.assertTrue(len(lines) > 1, "ranking_scores.csv should have header and data")
            self.assertEqual(len(lines[0].strip().split(",")), 3, "ranking_scores.csv should have 3 columns")

            expected_lines = 13
            self.assertEqual(
                len(lines),
                expected_lines,
                f"Expected {expected_lines} lines in ranking_scores.csv, found {len(lines)}",
            )

            for i, line in enumerate(lines[1:], 1):
                parts = line.strip().split(",")
                self.assertEqual(
                    len(parts),
                    3,
                    f"Line {i+1} should have 3 columns: seed,sample,ranking_score",
                )
                try:
                    int(parts[0])
                    int(parts[1])
                    float(parts[2])
                except ValueError:
                    self.fail(f"Line {i+1} has invalid format: {line.strip()}")

        print(
            f"✓ Verified multi_seeds_samples output with {len(sample_dirs)} sample "
            f"directories and {len(lines)-1} ranking score entries"
        )

    def test_af3_run_structure_prediction_keeps_single_explicit_output_dir_flat_for_json(self):
        """A single explicit output dir must remain flat even with --use_ap_style."""
        self._require_af3_functional_environment()
        env = self._make_af3_test_env()
        flash_impl = self._af3_flash_attention_impl()
        json_input = self.test_features_dir / "protein_with_ptms.json"

        res = subprocess.run(
            [
                sys.executable,
                str(self.script_single),
                f"--input={json_input}",
                f"--output_directory={self.output_dir}",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={self.test_features_dir}",
                "--fold_backend=alphafold3",
                f"--flash_attention_implementation={flash_impl}",
                "--num_diffusion_samples=1",
                "--use_ap_style",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self._runCommonTests(res)
        self.assertFalse(
            (self.output_dir / "protein_ptms").exists(),
            "Single-job AF3 runs should keep outputs directly in the explicitly provided output directory.",
        )

    def test_af3_run_multimer_jobs_multiple_jobs_create_per_job_subdirs(self):
        """Shared AF3 wrapper output roots must isolate multiple jobs by subdirectory."""
        self._require_af3_functional_environment()
        env = self._make_af3_test_env()
        flash_impl = self._af3_flash_attention_impl()
        protein_list = self.test_protein_lists_dir / "test_multiple_monomers.txt"

        res = subprocess.run(
            [
                sys.executable,
                str(self.script_multimer),
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_dir={DATA_DIR}",
                f"--monomer_objects_dir={self.test_features_dir}",
                f"--output_path={self.output_dir}",
                "--mode=custom",
                f"--protein_lists={protein_list}",
                "--fold_backend=alphafold3",
                f"--flash_attention_implementation={flash_impl}",
                "--num_diffusion_samples=1",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        print(res.stdout)
        print(res.stderr)
        self.assertEqual(res.returncode, 0, "sub-process failed")
        self.assertFalse(
            (self.output_dir / "ranking_scores.csv").exists(),
            "Shared wrapper output root should not contain flattened AF3 outputs.",
        )

        # AF3 currently merges all objects passed to one run_structure_prediction
        # invocation into a single combined fold input, so shared-root
        # multi-job isolation is validated through the wrapper path instead.
        for job_dir in ("A0A024R1R8_1-5", "A0A075B6L2_2-5"):
            current_output_dir = self.output_dir / job_dir
            self.assertTrue(
                current_output_dir.is_dir(),
                f"Expected per-job output directory {current_output_dir} to be created.",
            )
            self._assert_af3_outputs_present(current_output_dir)

    def test_af3_run_multimer_jobs_multiple_json_jobs_create_per_job_subdirs(self):
        """Shared AF3 wrapper roots must isolate combined JSON folds by subdirectory."""

        self._require_af3_functional_environment()
        env = self._make_af3_test_env()
        flash_impl = self._af3_flash_attention_impl()
        json_folds = [
            [
                self.test_features_dir / "protein_with_ptms.json",
                self.test_features_dir / "P61626_af3_input.json",
            ],
            [
                self.test_features_dir / "P01308_af3_input.json",
                self.test_features_dir / "P61626_af3_input.json",
            ],
        ]
        protein_list = self.output_dir / "test_multiple_json_jobs.txt"
        protein_list.write_text(
            "\n".join(
                ";".join(json_input.name for json_input in json_fold)
                for json_fold in json_folds
            )
            + "\n",
            encoding="utf-8",
        )

        res = subprocess.run(
            [
                sys.executable,
                str(self.script_multimer),
                "--num_cycle=1",
                "--num_predictions_per_model=1",
                f"--data_dir={DATA_DIR}",
                f"--monomer_objects_dir={self.test_features_dir}",
                f"--output_path={self.output_dir}",
                "--mode=custom",
                f"--protein_lists={protein_list}",
                "--fold_backend=alphafold3",
                f"--flash_attention_implementation={flash_impl}",
                "--num_diffusion_samples=1",
                "--use_ap_style",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        print(res.stdout)
        print(res.stderr)
        self.assertEqual(res.returncode, 0, "sub-process failed")
        self.assertFalse(
            (self.output_dir / "ranking_scores.csv").exists(),
            "Shared wrapper output root should not contain flattened AF3 JSON outputs.",
        )
        self.assertFalse(
            any(self.output_dir.glob("*_data.json")),
            "Combined JSON folds should not write flat AF3 input JSONs into the shared root.",
        )

        for output_dir_name in (
            "protein_with_ptms_and_p61626",
            "p01308_and_p61626",
        ):
            current_output_dir = self.output_dir / output_dir_name
            self.assertTrue(
                current_output_dir.is_dir(),
                f"Expected per-job output directory {current_output_dir} to be created.",
            )
            self._assert_af3_outputs_present(current_output_dir)

    @parameterized.named_parameters(
        dict(testcase_name="monomer", protein_list="test_monomer.txt", script="run_structure_prediction.py"),
        dict(testcase_name="dimer", protein_list="test_dimer.txt", script="run_structure_prediction.py"),
        dict(testcase_name="trimer", protein_list="test_trimer.txt", script="run_structure_prediction.py"),
        dict(testcase_name="homo_oligomer", protein_list="test_homooligomer.txt", script="run_structure_prediction.py"),
        dict(
            testcase_name="chopped_dimer",
            protein_list="test_dimer_chopped.txt",
            script="run_structure_prediction.py",
            check="_assert_chopped_regions_keep_their_residue_ids",
        ),
        dict(testcase_name="long_name", protein_list="test_long_name.txt", script="run_structure_prediction.py"),
        # Ensure AF3 also works when launched via the multimer wrapper script
        dict(testcase_name="monomer_via_multimer_wrapper", protein_list="test_monomer.txt", script="run_multimer_jobs.py"),
        # Test cases for combining AlphaPulldown monomer with different JSON inputs
        dict(
            testcase_name="monomer_with_rna", 
            protein_list="test_monomer_with_rna.txt", 
            script="run_structure_prediction.py"
        ),
        dict(
            testcase_name="monomer_with_dna", 
            protein_list="test_monomer_with_dna.txt", 
            script="run_structure_prediction.py"
        ),
        dict(
            testcase_name="monomer_with_ligand", 
            protein_list="test_monomer_with_ligand.txt", 
            script="run_structure_prediction.py"
        ),
        # Test case for protein with PTMs from JSON
        dict(
            testcase_name="protein_with_ptms", 
            protein_list="test_protein_with_ptms.txt", 
            script="run_structure_prediction.py"
        ),
        # Test case for multiple seeds and diffusion samples
        dict(
            testcase_name="multi_seeds_samples", 
            protein_list="test_multi_seeds_samples.txt", 
            script="run_structure_prediction.py",
            check="_assert_multi_seeds_samples_outputs",
        ),
        # Test homodimer from af3 features
        dict(
            testcase_name="homodimer_from_json_features",
            protein_list="test_homodimer_from_json_features.txt",
            script="run_structure_prediction.py",
        ),
    )
    def test_(self, protein_list, script, check=None):
        # Create environment with GPU settings
        env = self._make_af3_test_env()
        
        # Debug output
        print("\nEnvironment variables:")
        print(f"XLA_FLAGS: {env.get('XLA_FLAGS')}")
        print(f"XLA_PYTHON_CLIENT_PREALLOCATE: {env.get('XLA_PYTHON_CLIENT_PREALLOCATE')}")
        print(f"XLA_CLIENT_MEM_FRACTION: {env.get('XLA_CLIENT_MEM_FRACTION')}")
        print(f"JAX_FLASH_ATTENTION_IMPL: {env.get('JAX_FLASH_ATTENTION_IMPL')}")
        
        # Check GPU availability
        try:
            import jax
            print("\nJAX GPU devices:")
            print(jax.devices())
            print("JAX GPU local devices:")
            print(jax.local_devices(backend='gpu'))
        except Exception as e:
            print(f"\nError checking JAX GPU: {e}")
        
        res = subprocess.run(
            self._args(plist=protein_list, script=script),
            capture_output=True,
            text=True,
            env=env
        )
        self._runCommonTests(res)
        
        # Check chain counts and sequences
        self._check_chain_counts_and_sequences(protein_list)
        if check:
            getattr(self, check)()

    def test_af3_writes_embeddings_and_distogram(self):
        """Run AF3 with embeddings and distogram enabled and check files exist."""
        env = self._make_af3_test_env()
        flash_impl = self._af3_flash_attention_impl()

        args = [
            sys.executable,
            str(self.script_single),
            f"--input=A0A075B6L2:1:2-5",  # small chopped example
            f"--output_directory={self.output_dir}",
            f"--data_directory={DATA_DIR}",
            f"--features_directory={self.test_features_dir}",
            "--fold_backend=alphafold3",
            f"--flash_attention_implementation={flash_impl}",
            "--save_embeddings",
            "--save_distogram",
            "--num_diffusion_samples=1",
        ]

        res = subprocess.run(args, capture_output=True, text=True, env=env)
        self._runCommonTests(res)

        # Check per-seed embeddings and distogram artifacts in output dir
        seed_emb_dirs = list(self.output_dir.glob("seed-*_*embeddings"))
        seed_dist_dirs = list(self.output_dir.glob("seed-*_*distogram"))
        self.assertTrue(len(seed_emb_dirs) >= 1, "No embeddings directories written")
        self.assertTrue(len(seed_dist_dirs) >= 1, "No distogram directories written")
        # Number of embeddings/distogram directories should equal number of unique seeds
        with open(self.output_dir / "ranking_scores.csv") as f:
            lines = [ln.strip() for ln in f.readlines()[1:] if ln.strip()]
        seeds_in_csv = {ln.split(",")[0] for ln in lines}
        self.assertEqual(len(seed_emb_dirs), len(seeds_in_csv),
                         f"Embeddings dirs ({len(seed_emb_dirs)}) != seeds ({len(seeds_in_csv)})")
        self.assertEqual(len(seed_dist_dirs), len(seeds_in_csv),
                         f"Distogram dirs ({len(seed_dist_dirs)}) != seeds ({len(seeds_in_csv)})")

        # Check expected files inside
        for emb_dir in seed_emb_dirs:
            npz_files = list(emb_dir.glob("*.npz"))
            self.assertTrue(len(npz_files) >= 1, f"No embeddings npz in {emb_dir}")
            # Validate embeddings content
            for npz in npz_files:
                with np.load(npz) as data:
                    self.assertIn('single_embeddings', data.files, f"single_embeddings missing in {npz}")
                    self.assertIn('pair_embeddings', data.files, f"pair_embeddings missing in {npz}")
                    self.assertGreater(data['single_embeddings'].size, 0, f"single_embeddings empty in {npz}")
                    self.assertGreater(data['pair_embeddings'].size, 0, f"pair_embeddings empty in {npz}")
        for d_dir in seed_dist_dirs:
            npz_files = list(d_dir.glob("*_distogram.npz"))
            self.assertTrue(len(npz_files) >= 1, f"No distogram npz in {d_dir}")
            # Validate distogram content
            for npz in npz_files:
                with np.load(npz) as data:
                    self.assertIn('distogram', data.files, f"distogram key missing in {npz}")
                    self.assertGreater(data['distogram'].size, 0, f"distogram array empty in {npz}")

    def test_af3_num_recycles_affects_runtime(self):
        """num_recycles=1 should be faster than default (keeping other knobs same)."""
        if os.getenv("AF3_RUN_PERF_TESTS", "").lower() not in ("1", "true", "yes"):
            self.skipTest(
                "Set AF3_RUN_PERF_TESTS=1 to run AF3 runtime benchmarks."
            )

        self._require_af3_functional_environment()
        env = self._make_af3_test_env()
        flash_impl = self._af3_flash_attention_impl()

        common = [
            sys.executable,
            str(self.script_single),
            f"--input=A0A075B6L2:1",
            f"--output_directory={self.output_dir}",
            f"--data_directory={DATA_DIR}",
            f"--features_directory={self.test_features_dir}",
            "--fold_backend=alphafold3",
            f"--flash_attention_implementation={flash_impl}",
            "--num_diffusion_samples=1",
            "--num_seeds=2",  # ensures second seed reuses compiled XLA and timing reflects compute
        ]

        # Default num_recycles (10) – measure per-seed inference time from logs (last seed)
        res_default = subprocess.run(common, capture_output=True, text=True, env=env)
        self._runCommonTests(res_default)
        combined_default = res_default.stdout + "\n" + res_default.stderr
        m_default = re.findall(r"Model inference for seed .* took ([0-9.]+) seconds\.", combined_default)
        self.assertTrue(len(m_default) >= 1, "Couldn't parse default inference time from logs")
        default_time = float(m_default[-1])

        # num_recycles=1
        faster_dir = self.output_dir / "fewer_recycles"
        faster_dir.mkdir(parents=True, exist_ok=True)
        args_fast = common.copy()
        args_fast[args_fast.index(f"--output_directory={self.output_dir}")] = f"--output_directory={faster_dir}"
        args_fast.append("--num_recycles=1")
        res_fast = subprocess.run(args_fast, capture_output=True, text=True, env=env)
        self._runCommonTests(res_fast)
        combined_fast = res_fast.stdout + "\n" + res_fast.stderr
        m_fast = re.findall(r"Model inference for seed .* took ([0-9.]+) seconds\.", combined_fast)
        self.assertTrue(len(m_fast) >= 1, "Couldn't parse fast inference time from logs")
        fast_time = float(m_fast[-1])

        # Allow some jitter; require at least 15% faster with fewer recycles
        self.assertLess(
            fast_time,
            0.95 * default_time,
            f"num_recycles=1 not faster enough (default {default_time:.2f}s vs {fast_time:.2f}s)",
        )

class TestAlphaFold3FastKernels(_TestBase):
    """--fast_kernels for AF3 on a GPU, and one model compile per process.

    Select with -k TestAlphaFold3FastKernels. ``on`` skips, and ``auto`` may fall back,
    only when the backend says the kernels are not supported here and nvidia-smi, read
    against the fork's device policy, agrees. Kernels that fail on a GPU the policy
    enables are a failure, never a skip.
    """

    FOLD = "A0A075B6L2:1"
    # A homodimer, so that the comparison also covers ipTM.
    DIMER = "A0A075B6L2:2"
    NOT_SUPPORTED = "AF3 fused triangle kernels are not supported here"
    RANDOM_SEED = 42
    # Fused bf16 kernels change rounding, not the model: per seed, ranking score, pTM
    # and ipTM within 0.02 of the original layers, mean pLDDT (0-100) within 1.
    SCORE_TOLERANCE = 0.02
    PLDDT_TOLERANCE = 1.0

    def _fused_triangle_support(self) -> af3_gpu_checks.Support:
        """The fork's policy for this GPU, from nvidia-smi rather than the code under test."""
        return af3_gpu_checks.fused_triangle_support(
            af3_gpu_checks.visible_gpu(),
            af3_gpu_checks.jax_memory_fraction(self._make_af3_test_env()),
            af3_gpu_checks.installed_device_policy(),
        )

    def _assert_gpu_unsupported(self, backend_reason: str) -> str:
        """The independent check's detail, once it agrees the kernels are not enabled here."""
        support = self._fused_triangle_support()
        self.assertIs(
            support.supported,
            False,
            f"the backend says {backend_reason!r}, but nvidia-smi and the fork's policy "
            f"say otherwise ({support.detail})",
        )
        return support.detail

    def _fold(self, mode: str, *flags: str, fold: str = FOLD) -> subprocess.CompletedProcess:
        self._require_af3_functional_environment()
        return subprocess.run(
            [
                sys.executable,
                str(self.script_single),
                f"--input={fold}",
                f"--output_directory={self.output_dir}",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={self.test_features_dir}",
                "--fold_backend=alphafold3",
                f"--flash_attention_implementation={self._af3_flash_attention_impl()}",
                "--num_diffusion_samples=1",
                "--num_seeds=2",
                f"--fast_kernels={mode}",
                *flags,
            ],
            capture_output=True,
            text=True,
            env=self._make_af3_test_env(),
        )

    def _kernel_records(self, mode: str) -> dict[str, dict[str, Any]]:
        """inference_kernels.json of a finished fold: one record per seed, scores finite."""
        result_dir = self._resolve_single_af3_result_dir()
        with (result_dir / "ranking_scores.csv").open() as handle:
            seeds = {line.split(",")[0] for line in handle.readlines()[1:] if line.strip()}
        records = json.loads((result_dir / "inference_kernels.json").read_text())
        self.assertEqual(set(records), {f"seed-{seed}" for seed in seeds})
        self.assertLen(records, 2)
        for key, record in records.items():
            self.assertEqual(record["backend"], "alphafold3", key)
            self.assertEqual(record["requested_mode"], mode, key)
            self.assertIsInstance(record["padded_tokens"], int, key)
        confidences = sorted(result_dir.rglob("*confidences.json"))
        self.assertTrue(confidences, f"no confidences in {result_dir}")
        for path in confidences:
            problems = af3_gpu_checks.confidence_problems(path.name, json.loads(path.read_text()))
            self.assertEqual(problems, [], path)
        return records

    def _sample_scores(self) -> dict[str, dict[str, float | None]]:
        """Ranking score, pTM, ipTM (None if undefined) and mean pLDDT of each sample."""
        scores = {}
        for sample_dir in sorted(self._resolve_single_af3_result_dir().glob("seed-*_sample-*")):
            (summary_path,) = sample_dir.glob("*summary_confidences.json")
            (full_path,) = [
                path
                for path in sample_dir.glob("*confidences.json")
                if not path.name.endswith("summary_confidences.json")
            ]
            summary = json.loads(summary_path.read_text())
            scores[sample_dir.name] = {
                "ranking_score": summary["ranking_score"],
                "ptm": summary["ptm"],
                "iptm": summary["iptm"],
                "mean_plddt": float(np.mean(json.loads(full_path.read_text())["atom_plddts"])),
            }
        return scores

    def _assert_fused(self, records: dict[str, dict[str, Any]]) -> None:
        for key, record in records.items():
            self.assertTrue(record["fused_kernels"], (key, record))
            self.assertIn("policy_version", record, key)
            self.assertEqual(
                set(record["operations"]),
                {
                    f"{operation}_c{channels}"
                    for operation in ("triangle_multiplication", "triangle_attention")
                    for channels in (128, 64)
                },
                key,
            )

    def test_af3_fast_kernels_off_keeps_the_original_layers(self):
        res = self._fold("off")
        self._runCommonTests(res)

        for key, record in self._kernel_records("off").items():
            self.assertFalse(record["fused_kernels"], key)
            self.assertEqual(record["reason"], "--fast_kernels=off", key)
            self.assertNotIn("operations", record, key)

    def test_af3_fast_kernels_on_runs_the_fused_layers(self):
        res = self._fold("on")
        log = res.stdout + res.stderr
        if res.returncode != 0 and f"--fast_kernels=on, but {self.NOT_SUPPORTED}" in log:
            reason = next(line for line in log.splitlines() if self.NOT_SUPPORTED in line)
            detail = self._assert_gpu_unsupported(reason.strip())
            self.skipTest(f"this GPU cannot run the fused triangles: {reason.strip()} ({detail})")
        self._runCommonTests(res)

        self._assert_fused(self._kernel_records("on"))

    def test_af3_fast_kernels_auto_uses_them_where_the_gpu_can(self):
        res = self._fold("auto")
        self._runCommonTests(res)

        records = self._kernel_records("auto")
        if any(record["fused_kernels"] for record in records.values()):
            self._assert_fused(records)
        else:
            for key, record in records.items():
                self.assertTrue(
                    record["reason"].startswith(f"--fast_kernels=auto: {self.NOT_SUPPORTED}"),
                    (key, record),
                )
            self._assert_gpu_unsupported(next(iter(records.values()))["reason"])

    def test_af3_fast_kernels_match_the_original_layers(self):
        """The same dimer and seeds with --fast_kernels off and on: the scores agree."""
        support = self._fused_triangle_support()
        if not support.supported:
            self.skipTest(
                f"the fork's policy does not enable the fused triangles here ({support.detail})"
            )

        root = self.output_dir
        runs = {}
        for mode in ("off", "on"):
            # _fold and the helpers that read its outputs work in self.output_dir.
            self.output_dir = root / mode
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._runCommonTests(
                self._fold(mode, f"--random_seed={self.RANDOM_SEED}", fold=self.DIMER)
            )
            runs[mode] = self._kernel_records(mode), self._sample_scores()

        (stock_records, stock), (fused_records, fused) = runs["off"], runs["on"]
        self.assertFalse(
            any(record["fused_kernels"] for record in stock_records.values()), stock_records
        )
        self._assert_fused(fused_records)
        print(f"original layers: {stock}\nfused kernels: {fused}")
        self.assertEqual(set(fused), set(stock))
        self.assertLen(stock, 2)  # two seeds, one sample each
        for sample, expected in stock.items():
            got = fused[sample]
            for key in ("ranking_score", "ptm", "iptm"):
                context = (sample, key, expected, got)
                self.assertEqual(got[key] is None, expected[key] is None, context)
                if expected[key] is not None:
                    self.assertLessEqual(
                        abs(got[key] - expected[key]), self.SCORE_TOLERANCE, context
                    )
            self.assertLessEqual(
                abs(got["mean_plddt"] - expected["mean_plddt"]),
                self.PLDDT_TOLERANCE,
                (sample, expected, got),
            )

    def test_af3_second_fold_loads_the_compiled_model_from_the_cache(self):
        """--jax_compilation_cache_dir: a second process loads what the first compiled."""
        self._require_af3_functional_environment()
        cache_dir = self.output_dir / "jax_cache"
        entries = {}
        for name in ("first", "second"):
            res = subprocess.run(
                [
                    sys.executable,
                    str(self.script_single),
                    f"--input={self.FOLD}",
                    f"--output_directory={self.output_dir / name}",
                    f"--data_directory={DATA_DIR}",
                    f"--features_directory={self.test_features_dir}",
                    "--fold_backend=alphafold3",
                    f"--flash_attention_implementation={self._af3_flash_attention_impl()}",
                    "--num_diffusion_samples=1",
                    f"--jax_compilation_cache_dir={cache_dir}",
                ],
                capture_output=True,
                text=True,
                env=self._make_af3_test_env(),
            )
            self.assertEqual(
                res.returncode, 0, f"{name} failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )
            self.assertTrue(list((self.output_dir / name).rglob("ranking_scores.csv")), name)
            entries[name] = {path.name for path in cache_dir.iterdir()}

        self.assertTrue(entries["first"], "the first fold wrote nothing to the cache")
        # Every compile of the second fold was a cache hit, so it wrote nothing new.
        self.assertEqual(entries["second"], entries["first"])

    def test_af3_identical_folds_compile_the_model_once(self):
        """Three identical folds in one process: one jit(apply_fn) compile (1afe779e)."""
        self._require_af3_functional_environment()
        manifest = self.output_dir / "manifest.jsonl"
        jobs = [f"repeat_{index}" for index in range(1, 4)]
        manifest.write_text(
            "".join(
                json.dumps(
                    {
                        "job_id": job,
                        "input": self.FOLD,
                        "output_directory": str(self.output_dir / job),
                    }
                )
                + "\n"
                for job in jobs
            )
        )
        env = self._make_af3_test_env()
        env["JAX_LOG_COMPILES"] = "1"

        res = subprocess.run(
            [
                sys.executable,
                str(self.script_single.parent / "run_structure_prediction_batch.py"),
                f"--manifest={manifest}",
                f"--data_directory={DATA_DIR}",
                f"--features_directory={self.test_features_dir}",
                "--fold_backend=alphafold3",
                f"--flash_attention_implementation={self._af3_flash_attention_impl()}",
                "--num_diffusion_samples=1",
                # A cache load would hide a recompile.
                "--jax_compilation_cache_dir=none",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(
            res.returncode, 0, f"batch failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )

        log = res.stdout + res.stderr
        for job in jobs:
            self.assertTrue(
                list((self.output_dir / job).rglob("ranking_scores.csv")),
                f"{job} wrote no predictions",
            )
        self.assertEqual(log.count("Finished XLA compilation of jit(apply_fn)"), 1)
        # "tracing + transforming apply_fn for pjit" on jax 0.9, "tracing apply_fn for jit"
        # on jax 0.10.
        traces = re.findall(r"Finished tracing (?:\+ transforming )?apply_fn for p?jit", log)
        self.assertLen(traces, 1)


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
