#!/usr/bin/env python
"""
Functional Alphapulldown tests for AlphaFold3 (parameterised).

The script is identical for Slurm and workstation users – only the
wrapper decides *how* each case is executed.
"""
from __future__ import annotations
import lzma
import os
import subprocess
import time
import sys
import tempfile
import hashlib
from pathlib import Path
import shutil
import pickle
import json
import numpy as np
import re
import unittest
from typing import Dict, List, Tuple, Any

from absl.testing import absltest, parameterized

import alphapulldown
from alphafold3.constants import residue_names as af3_residue_names
from alphapulldown.objects import MultimericObject
from alphapulldown.utils.modelling_setup import (
    create_custom_info,
    create_interactors,
    parse_fold,
)
from alphapulldown_input_parser import generate_fold_specifications


# --------------------------------------------------------------------------- #
#                       configuration / environment guards                    #
# --------------------------------------------------------------------------- #
# Point to the full Alphafold database once, via env-var.
DATA_DIR = os.getenv(
    "ALPHAFOLD_DATA_DIR",
    "/g/kosinski/dima/alphafold3_weights/"   #  default for EMBL cluster
)
if not os.path.exists(DATA_DIR):
    absltest.skip("set $ALPHAFOLD_DATA_DIR to run Alphafold functional tests")


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


def _a3m_sequences(a3m_text: str) -> list[str]:
    if not a3m_text:
        return []
    lines = [line.strip() for line in a3m_text.splitlines() if line.strip()]
    return [lines[index] for index in range(1, len(lines), 2)]


def _a3m_query_sequence(a3m_text: str) -> str:
    sequences = _a3m_sequences(a3m_text)
    return sequences[0] if sequences else ""


def _a3m_payload_sequences(a3m_text: str) -> list[str]:
    sequences = _a3m_sequences(a3m_text)
    return sequences[1:]


def _aligned_a3m_row_length(a3m_row: str) -> int:
    return len(re.sub(r"[a-z]", "", a3m_row))


def _protein_entries_from_af3_input(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        sequence_entry["protein"]
        for sequence_entry in payload.get("sequences", [])
        if "protein" in sequence_entry
    ]


def _load_json_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".xz":
        with lzma.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_feature_metadata(feature_dir: Path, protein_id: str) -> tuple[Path, dict[str, Any]]:
    matches = sorted(feature_dir.glob(f"{protein_id}_feature_metadata_*.json*"))
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one metadata file for {protein_id} in {feature_dir}, found {matches}"
        )
    return matches[0], _load_json_payload(matches[0])


def _metadata_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    raise AssertionError(f"Unsupported metadata boolean value: {value!r}")


# --------------------------------------------------------------------------- #
#                       common helper mix-in / assertions                     #
# --------------------------------------------------------------------------- #
class _TestBase(parameterized.TestCase):
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
        this_dir = Path(__file__).resolve().parent
        self.test_data_dir = this_dir / "test_data"
        self.test_fastas_dir = self.test_data_dir / "fastas"
        self.test_features_dir = this_dir / "test_data" / "features"
        self.test_protein_lists_dir = this_dir / "test_data" / "protein_lists"
        self.test_templates_dir = this_dir / "test_data" / "templates"
        self.test_modelling_dir = this_dir / "test_data" / "predictions"

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

    def _get_sequence_from_json(self, json_file: str) -> List[Tuple[str, str]]:
        """Extract sequences from a JSON file."""
        sequences = []
        json_path = self.test_features_dir / json_file
        if json_path.exists():
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            json_sequences = json_data.get('sequences', [])
            for seq_data in json_sequences:
                if 'protein' in seq_data:
                    protein_seq = seq_data['protein']
                    chain_id = protein_seq.get('id', 'A')
                    sequence = protein_seq.get('sequence', '')
                    
                    # Apply post-translational modifications if present
                    modifications = protein_seq.get('modifications', [])
                    if modifications:
                        sequence = self._apply_ptms_to_sequence(sequence, modifications)
                    
                    sequences.append((chain_id, sequence))
                elif 'rna' in seq_data:
                    rna_seq = seq_data['rna']
                    chain_id = rna_seq.get('id', 'A')
                    sequence = rna_seq.get('sequence', '')
                    sequences.append((chain_id, sequence))
                elif 'dna' in seq_data:
                    dna_seq = seq_data['dna']
                    chain_id = dna_seq.get('id', 'A')
                    sequence = dna_seq.get('sequence', '')
                    sequences.append((chain_id, sequence))
                elif 'ligand' in seq_data:
                    ligand_seq = seq_data['ligand']
                    chain_id = ligand_seq.get('id', 'L')
                    # For ligands, we use the CCD codes as the "sequence"
                    ccd_codes = ligand_seq.get('ccdCodes', [])
                    if ccd_codes:
                        # Join multiple CCD codes if present (e.g., ["ATP", "MG"] -> "ATP+MG")
                        sequence = '+'.join(ccd_codes)
                    else:
                        # Fallback to SMILES if no CCD codes
                        smiles = ligand_seq.get('smiles', '')
                        sequence = f"SMILES:{smiles}" if smiles else "UNKNOWN_LIGAND"
                    sequences.append((chain_id, sequence))
        return sequences

    def _apply_ptms_to_sequence(self, sequence: str, modifications: List[Dict]) -> str:
        """
        Apply PTMs to the expected structure-side sequence representation.
        
        Args:
            sequence: Original protein sequence
            modifications: List of PTM dictionaries with 'ptmType' and 'ptmPosition'
            
        Returns:
            Modified sequence with PTMs applied (same length as original)
        """
        # Convert to list for easier modification
        seq_list = list(sequence)
        
        for ptm in modifications:
            ptm_type = ptm.get('ptmType')
            ptm_position = ptm.get('ptmPosition', 1) - 1  # Convert to 0-based indexing
            
            if ptm_position < len(seq_list):
                if ptm_type == "HYS":
                    seq_list[ptm_position] = "H"
                elif ptm_type == "2MG":
                    seq_list[ptm_position] = "G"
                else:
                    seq_list[ptm_position] = af3_residue_names.letters_three_to_one(
                        ptm_type,
                        default='X',
                    )
        
        return ''.join(seq_list)

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

    def _chain_id_from_index(self, index: int) -> str:
        """Mirror AF3's reverse-spreadsheet chain ID progression."""
        if index < 26:
            return chr(ord('A') + index)
        first_char = chr(ord('A') + (index // 26) - 1)
        second_char = chr(ord('A') + (index % 26))
        return first_char + second_char

    def _get_region_sequences(self, protein_name: str, regions: list[tuple[int, int]]) -> list[str]:
        """Return one sequence fragment per requested 1-based closed interval."""
        full_sequence = self._get_sequence_for_protein(protein_name)
        if not full_sequence:
            return []

        region_sequences = []
        for start, end in regions:
            start_idx = start - 1
            end_idx = end
            region_sequences.append(full_sequence[start_idx:end_idx])
        return region_sequences

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
        env["XLA_FLAGS"] = "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_force_compilation_parallelism=0"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        env["XLA_CLIENT_MEM_FRACTION"] = "0.95"
        env["JAX_FLASH_ATTENTION_IMPL"] = flash_impl
        if "XLA_PYTHON_CLIENT_MEM_FRACTION" in env:
            del env["XLA_PYTHON_CLIENT_MEM_FRACTION"]
        return env

    def _af3_flash_attention_impl(self) -> str:
        return os.getenv("AF3_TEST_FLASH_ATTENTION_IMPL", "xla")

    def _require_af3_functional_environment(self) -> None:
        if not os.path.exists(DATA_DIR):
            self.skipTest(
                f"AF3 functional tests require ALPHAFOLD_DATA_DIR; missing path: {DATA_DIR}"
            )

    def _assert_af3_outputs_present(self, output_dir: Path) -> None:
        files = list(output_dir.iterdir())
        print(f"contents of {output_dir}: {[f.name for f in files]}")

        self.assertIn("TERMS_OF_USE.md", {f.name for f in files})
        self.assertIn("ranking_scores.csv", {f.name for f in files})

        conf_files = [f for f in files if f.name.endswith("_confidences.json")]
        summary_conf_files = [f for f in files if f.name.endswith("_summary_confidences.json")]
        model_files = [f for f in files if f.name.endswith("_model.cif")]

        self.assertTrue(len(conf_files) > 0, f"No confidences.json files found in {output_dir}")
        self.assertTrue(len(summary_conf_files) > 0, f"No summary_confidences.json files found in {output_dir}")
        self.assertTrue(len(model_files) > 0, f"No model.cif files found in {output_dir}")

        sample_dirs = [
            f for f in files if f.is_dir() and f.name.startswith("seed-") and "sample-" in f.name
        ]

        for sample_dir in sample_dirs:
            sample_files = list(sample_dir.iterdir())
            self.assertIn("confidences.json", {f.name for f in sample_files})
            self.assertIn("model.cif", {f.name for f in sample_files})
            self.assertIn("summary_confidences.json", {f.name for f in sample_files})

        with open(output_dir / "ranking_scores.csv") as f:
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

    def _prepare_fold_input(
        self,
        *,
        fold_spec: str,
        feature_dir: Path,
        debug_msas: bool = False,
    ):
        from alphapulldown.folding_backend.alphafold3_backend import AlphaFold3Backend

        parsed = parse_fold([fold_spec], [str(feature_dir)], "+")
        data = create_custom_info(parsed)
        all_interactors = create_interactors(data, [str(feature_dir)])
        self.assertLen(all_interactors, 1)
        self.assertGreaterEqual(len(all_interactors[0]), 1)

        interactors = all_interactors[0]
        if len(interactors) == 1:
            object_to_model = interactors[0]
        else:
            object_to_model = MultimericObject(interactors=interactors, pair_msa=True)

        mappings = AlphaFold3Backend.prepare_input(
            objects_to_model=[
                {"object": object_to_model, "output_dir": str(self.output_dir)}
            ],
            random_seed=42,
            debug_msas=debug_msas,
        )
        self.assertLen(mappings, 1)
        fold_input_obj, _ = next(iter(mappings[0].items()))
        return fold_input_obj

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
        self.assertIn(
            "af3_species_pairing_from_af2_individual_msas",
            summary["translation_modes"],
        )
        self.assertLen(summary["chains"], 2)

        for chain_summary in summary["chains"]:
            chain_id = chain_summary["chain_id"]
            expected_sequence = chain_sequences[chain_id]
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

            for msa_kind in ("paired_input", "unpaired_input"):
                msa_path = self.output_dir / f"{job_name}_chain-{chain_id}_{msa_kind}.a3m"
                self.assertTrue(msa_path.is_file(), f"Missing debug MSA {msa_path}")
                msa_text = msa_path.read_text(encoding="utf-8")
                self.assertEqual(_a3m_query_sequence(msa_text), expected_sequence)
                payload_sequences = _a3m_payload_sequences(msa_text)
                self.assertGreater(
                    len(payload_sequences),
                    0,
                    f"Expected payload rows in {msa_path}",
                )
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
            self.assertEqual(
                _a3m_query_sequence(protein_entry["pairedMsa"]),
                expected_sequence,
            )
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


# --------------------------------------------------------------------------- #
#                        parameterised "run mode" tests                       #
# --------------------------------------------------------------------------- #
class TestAlphaFold3RunModes(_TestBase):
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

    def test_af3_predicts_discontinuous_chopped_regions_as_one_gapped_chain(self):
        """Run AF3 inference and ensure discontinuous chopped regions remain one chain."""
        self._require_af3_functional_environment()
        env = self._make_af3_test_env()
        flash_impl = self._af3_flash_attention_impl()

        res = subprocess.run(
            [
                sys.executable,
                str(self.script_single),
                "--input=TEST+A0A075B6L2:1-10:2-5:12-15",
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
            + ["2A", "3A", "4A", "5A"]
            + list(range(12, 16))
        )

        result_dir = self._resolve_single_af3_result_dir()
        cif_files = list(result_dir.glob("*_model.cif"))
        self.assertTrue(cif_files, f"No predicted CIF files found in {result_dir}")

        actual_chains_and_sequences = self._extract_cif_chains_and_sequences(cif_files[0])
        actual_sequences = [sequence for _, sequence in actual_chains_and_sequences]
        residue_numbers_by_chain = dict(self._extract_cif_chain_residue_numbers(cif_files[0]))
        sequences_by_chain = dict(actual_chains_and_sequences)

        self.assertLen(actual_sequences, 2)
        self.assertCountEqual(actual_sequences, expected_sequences)
        chopped_chain_ids = [
            chain_id
            for chain_id, sequence in sequences_by_chain.items()
            if sequence == concatenated_chopped_sequence
        ]
        self.assertLen(chopped_chain_ids, 1)
        self.assertEqual(
            residue_numbers_by_chain[chopped_chain_ids[0]],
            expected_chopped_residue_ids,
        )

        print("✓ AF3 prediction keeps discontinuous chopped regions as one gapped chain")

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

    def test_dimer_chopped_expected_sequences_are_concatenated_per_chain(self):
        """Sequence expectations for AF3 chopped inputs must reflect one gapped chain."""
        expected_sequences = self._extract_expected_sequences("test_dimer_chopped.txt")
        chopped_sequence = "".join(
            self._get_region_sequences(
                "A0A075B6L2",
                [(1, 10), (2, 5), (12, 15)],
            )
        )
        self.assertCountEqual(
            [sequence for _, sequence in expected_sequences],
            [
                self._get_sequence_for_protein("TEST"),
                chopped_sequence,
            ],
        )
        self.assertLen(expected_sequences, 2)

    def test_multi_seeds_samples_sequence_extraction(self):
        """Test that sequence extraction works correctly for multi_seeds_samples."""
        # Test the sequence extraction logic directly
        expected_sequences = self._extract_expected_sequences("test_multi_seeds_samples.txt")
        
        # The expected result should be [('A', 'PLVV')] for A0A075B6L2,2-5
        self.assertEqual(expected_sequences, [('A', 'PLVV')], 
                        f"Expected [('A', 'PLVV')], got {expected_sequences}")

    def test_multi_seeds_samples_output_validation(self):
        """Test that the multi_seeds_samples output files are correct."""
        if not (self.output_dir / "ranking_scores.csv").exists():
            # Keep this validation test independently runnable under isolated temp dirs.
            env = self._make_af3_test_env()
            res = subprocess.run(
                self._args(
                    plist="test_multi_seeds_samples.txt",
                    script="run_structure_prediction.py",
                ),
                capture_output=True,
                text=True,
                env=env,
            )
            self._runCommonTests(res)

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

        self._check_chain_counts_and_sequences("test_multi_seeds_samples.txt")

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
        """Shared AF3 wrapper output roots must isolate multiple JSON jobs by subdirectory."""
        from alphapulldown.utils.output_paths import derive_af3_job_name_from_json

        self._require_af3_functional_environment()
        env = self._make_af3_test_env()
        flash_impl = self._af3_flash_attention_impl()
        json_inputs = [
            self.test_features_dir / "protein_with_ptms.json",
            self.test_features_dir / "P01308_af3_input.json",
        ]
        protein_list = self.output_dir / "test_multiple_json_jobs.txt"
        protein_list.write_text(
            "\n".join(json_input.name for json_input in json_inputs) + "\n",
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

        for json_input in json_inputs:
            current_output_dir = self.output_dir / derive_af3_job_name_from_json(
                str(json_input)
            )
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
        dict(testcase_name="chopped_dimer", protein_list="test_dimer_chopped.txt", script="run_structure_prediction.py"),
        dict(testcase_name="long_name", protein_list="test_long_name.txt", script="run_structure_prediction.py"),
        # Ensure AF3 also works when launched via the multimer wrapper script
        dict(testcase_name="monomer_via_multimer_wrapper", protein_list="test_monomer.txt", script="run_multimer_jobs.py"),
        dict(testcase_name="chopped_dimer_via_multimer_wrapper", protein_list="test_dimer_chopped.txt", script="run_multimer_jobs.py"),
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
            script="run_structure_prediction.py"
        ),
        # Test homodimer from af3 features
        dict(
            testcase_name="homodimer_from_json_features",
            protein_list="test_homodimer_from_json_features.txt",
            script="run_structure_prediction.py",
        ),
    )
    def test_(self, protein_list, script):
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

    def test_af3_rejects_alphafold2_flag(self):
        """Passing AF2-only flags to AF3 backend should fail via validator."""
        env = self._make_af3_test_env()
        flash_impl = self._af3_flash_attention_impl()

        args = [
            sys.executable,
            str(self.script_single),
            f"--input=A0A075B6L2:1:2-5",
            f"--output_directory={self.output_dir}",
            f"--data_directory={DATA_DIR}",
            f"--features_directory={self.test_features_dir}",
            "--fold_backend=alphafold3",
            f"--flash_attention_implementation={flash_impl}",
            "--num_diffusion_samples=1",
            # Intentionally invalid for AF3:
            "--num_predictions_per_model=1",
        ]
        res = subprocess.run(args, capture_output=True, text=True, env=env)
        # Expect non-zero exit and clear error message
        self.assertNotEqual(res.returncode, 0, "AF3 run unexpectedly succeeded with AF2 flag")
        self.assertRegex(
            res.stderr + res.stdout,
            r"not supported by backend 'alphafold3'",
        )


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
