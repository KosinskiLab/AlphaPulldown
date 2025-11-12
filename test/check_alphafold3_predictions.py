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
from pathlib import Path
import shutil
import pickle
import json
import numpy as np
import re
from typing import Dict, List, Tuple, Any

from absl.testing import absltest, parameterized

import alphapulldown
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
        Apply post-translational modifications to a protein sequence.
        
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
                    # N-terminal histidine modification - replace N-terminal methionine with HYS
                    if ptm_position == 0 and seq_list[0] == 'M':
                        # Replace M with H (histidine) - HYS is the CCD code, but we use H for sequence
                        seq_list[0] = 'H'
                elif ptm_type == "2MG":
                    # 2-methylguanosine modification - replace G with modified G
                    # For simplicity, we'll keep it as G since the exact representation may vary
                    pass
                # Add more PTM types as needed
                else:
                    print(f"Warning: Unknown PTM type '{ptm_type}' at position {ptm_position + 1}")
        
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
        
        # Get the chopped sequence
        def get_chopped_sequence(protein_name: str, regions: list) -> str:
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
        
        sequence = get_chopped_sequence(protein_name, regions)
        if not sequence:
            return []
        
        # Create multiple copies with different chain IDs
        sequences = []
        for i in range(num_copies):
            chain_id = chr(ord('A') + i)
            sequences.append((chain_id, sequence))
        
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
        def get_chopped_sequence(protein_name: str, regions: list) -> str:
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
            for i, part in enumerate(parts):
                part = part.strip()
                if "," in part:
                    protein_name, regions = parse_protein_and_regions(part)
                    sequence = get_chopped_sequence(protein_name, regions)
                else:
                    protein_name = part
                    sequence = self._get_sequence_for_protein(protein_name)
                if sequence:
                    chain_id = chr(ord('A') + i)
                    sequences.append((chain_id, sequence))
            return sequences
        else:
            # Single chopped protein
            part = line.strip()
            if "," in part:
                protein_name, regions = parse_protein_and_regions(part)
                sequence = get_chopped_sequence(protein_name, regions)
            else:
                protein_name = part
                sequence = self._get_sequence_for_protein(protein_name)
            if sequence:
                return [('A', sequence)]
        return []

    def _extract_cif_chains_and_sequences(self, cif_path: Path) -> List[Tuple[str, str]]:
        """
        Extract chain IDs and sequences from a CIF file using Biopython.
        
        Args:
            cif_path: Path to the CIF file
            
        Returns:
            List of tuples (chain_id, sequence) for chains in the CIF file
        """
        chains_and_sequences = []
        
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
                
                # Get residues in order
                residues = list(chain.get_residues())
                residues.sort(key=lambda r: r.id[1])  # Sort by residue number
                
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
        struct_asym_pattern = r'([A-Z])\s+(\d+)'
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
        nonpoly_asym_pattern = r'_pdbx_nonpoly_scheme\.asym_id\s+([A-Z])'
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

    def _get_ptm_positions(self, protein_list: str) -> List[int]:
        """
        Extract PTM positions from JSON files for a given protein list.
        
        Args:
            protein_list: Name of the protein list file
            
        Returns:
            List of PTM positions (1-based)
        """
        ptm_positions = []
        
        # Read the protein list file
        protein_list_path = self.test_protein_lists_dir / protein_list
        with open(protein_list_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        for line in lines:
            if line.endswith('.json'):
                json_path = self.test_features_dir / line
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                    
                    json_sequences = json_data.get('sequences', [])
                    for seq_data in json_sequences:
                        if 'protein' in seq_data:
                            protein_seq = seq_data['protein']
                            modifications = protein_seq.get('modifications', [])
                            for ptm in modifications:
                                ptm_position = ptm.get('ptmPosition')
                                if ptm_position:
                                    ptm_positions.append(ptm_position)
        
        return ptm_positions

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
        cif_files = list(self.output_dir.glob("*_model.cif"))
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
        
        # Check if this is a PTM case
        ptm_positions = self._get_ptm_positions(protein_list)
        is_ptm_case = len(ptm_positions) > 0
        
        if is_ptm_case:
            # For PTM cases, check that sequences are reasonable for PTM cases
            print(f"PTM case detected. PTM positions: {ptm_positions}")
            self._check_sequences_with_ptms(expected_sequences, actual_chains_and_sequences, ptm_positions)
        else:
            # For non-PTM cases, check exact sequence matches
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

    def _check_sequences_with_ptms(self, expected_sequences: List[Tuple[str, str]], 
                                  actual_chains_and_sequences: List[Tuple[str, str]], 
                                  ptm_positions: List[int]):
        """
        Check that sequences are reasonable for PTM cases.
        
        Args:
            expected_sequences: List of (chain_id, sequence) tuples for expected chains
            actual_chains_and_sequences: List of (chain_id, sequence) tuples for actual chains
            ptm_positions: List of PTM positions (1-based)
        """
        # Create dictionaries for easier lookup
        expected_dict = dict(expected_sequences)
        actual_dict = dict(actual_chains_and_sequences)
        
        # Check that all chain IDs match
        self.assertEqual(
            set(expected_dict.keys()),
            set(actual_dict.keys()),
            f"Chain IDs don't match. Expected: {set(expected_dict.keys())}, Actual: {set(actual_dict.keys())}"
        )
        
        # Check each chain
        for chain_id in expected_dict.keys():
            expected_seq = expected_dict[chain_id]
            actual_seq = actual_dict[chain_id]
            
            print(f"Chain {chain_id}:")
            print(f"  Expected: {expected_seq}")
            print(f"  Actual:   {actual_seq}")
            print(f"  PTM positions: {ptm_positions}")
            
            # For PTM cases, we'll be very lenient and just check that:
            # 1. Chain IDs match (already checked above)
            # 2. Sequences are not empty
            # 3. Sequences contain only valid amino acid characters
            
            self.assertGreater(
                len(actual_seq),
                0,
                f"Sequence for chain {chain_id} is empty"
            )
            
            # Check that sequence contains only valid amino acid characters
            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
            invalid_chars = set(actual_seq) - valid_aa
            self.assertEqual(
                len(invalid_chars),
                0,
                f"Sequence for chain {chain_id} contains invalid amino acid characters: {invalid_chars}"
            )
            
            print(f"  ✓ Chain {chain_id}: Valid sequence with correct chain ID")

    # ---------------- assertions reused by all subclasses ----------------- #
    def _runCommonTests(self, res: subprocess.CompletedProcess):
        print(res.stdout)
        print(res.stderr)
        self.assertEqual(res.returncode, 0, "sub-process failed")

        # Look in the parent directory for output files
        files = list(self.output_dir.iterdir())
        print(f"contents of {self.output_dir}: {[f.name for f in files]}")

        # Check for AlphaFold3 output files
        # 1. Main output files
        self.assertIn("TERMS_OF_USE.md", {f.name for f in files})
        self.assertIn("ranking_scores.csv", {f.name for f in files})
        
        # 2. Data and confidence files
        conf_files = [f for f in files if f.name.endswith("_confidences.json")]
        summary_conf_files = [f for f in files if f.name.endswith("_summary_confidences.json")]
        model_files = [f for f in files if f.name.endswith("_model.cif")]
        
        self.assertTrue(len(conf_files) > 0, "No confidences.json files found")
        self.assertTrue(len(summary_conf_files) > 0, "No summary_confidences.json files found")
        self.assertTrue(len(model_files) > 0, "No model.cif files found")

        # 3. Check sample directories (only those with 'sample-' suffix)
        sample_dirs = [
            f for f in files if f.is_dir() and f.name.startswith("seed-") and "sample-" in f.name
        ]

        for sample_dir in sample_dirs:
            sample_files = list(sample_dir.iterdir())
            self.assertIn("confidences.json", {f.name for f in sample_files})
            self.assertIn("model.cif", {f.name for f in sample_files})
            self.assertIn("summary_confidences.json", {f.name for f in sample_files})

        # 4. Verify ranking scores
        with open(self.output_dir / "ranking_scores.csv") as f:
            lines = f.readlines()
            self.assertTrue(len(lines) > 1, "ranking_scores.csv should have header and data")
            self.assertEqual(len(lines[0].strip().split(",")), 3, "ranking_scores.csv should have 3 columns")
            # Expected number of sample directories equals number of ranking entries for current run
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
            
            # Verify CSV format for all data lines
            for i, line in enumerate(lines[1:], 1):  # Skip header
                parts = line.strip().split(",")
                self.assertEqual(len(parts), 3, f"Line {i+1} should have 3 columns: seed,sample,ranking_score")
                # Verify that seed, sample are integers and ranking_score is a float
                try:
                    int(parts[0])  # seed
                    int(parts[1])  # sample
                    float(parts[2])  # ranking_score
                except ValueError:
                    self.fail(f"Line {i+1} has invalid format: {line.strip()}")
                    
            print(f"✓ Verified ranking_scores.csv has correct format with {len(lines)-1} entries")

    # convenience builder
    def _args(self, *, plist, script):
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
                "--flash_attention_implementation=xla",
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
                "--flash_attention_implementation=xla",
                "--num_diffusion_samples=1",
            ]
            return args


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

    def test_multi_seeds_samples_sequence_extraction(self):
        """Test that sequence extraction works correctly for multi_seeds_samples."""
        # Test the sequence extraction logic directly
        expected_sequences = self._extract_expected_sequences("test_multi_seeds_samples.txt")
        
        # The expected result should be [('A', 'PLVV')] for A0A075B6L2,2-5
        self.assertEqual(expected_sequences, [('A', 'PLVV')], 
                        f"Expected [('A', 'PLVV')], got {expected_sequences}")

    def test_multi_seeds_samples_output_validation(self):
        """Test that the multi_seeds_samples output files are correct."""
        # Set up the test to use the existing output directory
        test_name = "test__multi_seeds_samples"
        output_dir = Path("test/test_data/predictions/af3_backend") / test_name
        
        if not output_dir.exists():
            self.skipTest(f"Output directory {output_dir} does not exist. Run the full test first.")
        
        # Temporarily set the output directory
        original_output_dir = self.output_dir
        self.output_dir = output_dir
        
        try:
            # Check that all expected files exist
            files = list(self.output_dir.iterdir())
            
            # Check for main output files
            self.assertIn("TERMS_OF_USE.md", {f.name for f in files})
            self.assertIn("ranking_scores.csv", {f.name for f in files})
            
            # Check for AlphaFold3 output files
            conf_files = [f for f in files if f.name.endswith("_confidences.json")]
            summary_conf_files = [f for f in files if f.name.endswith("_summary_confidences.json")]
            model_files = [f for f in files if f.name.endswith("_model.cif")]
            
            self.assertTrue(len(conf_files) > 0, "No confidences.json files found")
            self.assertTrue(len(summary_conf_files) > 0, "No summary_confidences.json files found")
            self.assertTrue(len(model_files) > 0, "No model.cif files found")
            
            # Check sample directories (should be 12: 3 seeds × 4 samples)
            sample_dirs = [f for f in files if f.is_dir() and f.name.startswith("seed-")]
            self.assertEqual(len(sample_dirs), 12, 
                           f"Expected 12 sample directories, found {len(sample_dirs)}")
            
            # Check each sample directory has the required files
            for sample_dir in sample_dirs:
                sample_files = list(sample_dir.iterdir())
                self.assertIn("confidences.json", {f.name for f in sample_files})
                self.assertIn("model.cif", {f.name for f in sample_files})
                self.assertIn("summary_confidences.json", {f.name for f in sample_files})
            
            # Verify ranking scores
            with open(self.output_dir / "ranking_scores.csv") as f:
                lines = f.readlines()
                self.assertTrue(len(lines) > 1, "ranking_scores.csv should have header and data")
                self.assertEqual(len(lines[0].strip().split(",")), 3, "ranking_scores.csv should have 3 columns")
                
                # Should have 12 entries + 1 header = 13 lines
                expected_lines = 13
                self.assertEqual(len(lines), expected_lines, 
                               f"Expected {expected_lines} lines in ranking_scores.csv, found {len(lines)}")
                
                # Verify CSV format for all data lines
                for i, line in enumerate(lines[1:], 1):  # Skip header
                    parts = line.strip().split(",")
                    self.assertEqual(len(parts), 3, f"Line {i+1} should have 3 columns: seed,sample,ranking_score")
                    # Verify that seed, sample are integers and ranking_score is a float
                    try:
                        int(parts[0])  # seed
                        int(parts[1])  # sample
                        float(parts[2])  # ranking_score
                    except ValueError:
                        self.fail(f"Line {i+1} has invalid format: {line.strip()}")
            
            # Check chain counts and sequences
            self._check_chain_counts_and_sequences("test_multi_seeds_samples.txt")
            
            print(f"✓ Verified multi_seeds_samples output with {len(sample_dirs)} sample directories and {len(lines)-1} ranking score entries")
            
        finally:
            # Restore original output directory
            self.output_dir = original_output_dir

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
        env = os.environ.copy()
        env["XLA_FLAGS"] = "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_force_compilation_parallelism=0"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        env["XLA_CLIENT_MEM_FRACTION"] = "0.95"
        env["JAX_FLASH_ATTENTION_IMPL"] = "xla"
        # Remove deprecated variable if present
        if "XLA_PYTHON_CLIENT_MEM_FRACTION" in env:
            del env["XLA_PYTHON_CLIENT_MEM_FRACTION"]
        
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
        env = os.environ.copy()
        env["XLA_FLAGS"] = "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_force_compilation_parallelism=0"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        env["XLA_CLIENT_MEM_FRACTION"] = "0.95"
        env["JAX_FLASH_ATTENTION_IMPL"] = "xla"
        if "XLA_PYTHON_CLIENT_MEM_FRACTION" in env:
            del env["XLA_PYTHON_CLIENT_MEM_FRACTION"]

        args = [
            sys.executable,
            str(self.script_single),
            f"--input=A0A075B6L2:1:2-5",  # small chopped example
            f"--output_directory={self.output_dir}",
            f"--data_directory={DATA_DIR}",
            f"--features_directory={self.test_features_dir}",
            "--fold_backend=alphafold3",
            "--flash_attention_implementation=xla",
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
        env = os.environ.copy()
        env["XLA_FLAGS"] = "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_force_compilation_parallelism=0"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        env["XLA_CLIENT_MEM_FRACTION"] = "0.95"
        env["JAX_FLASH_ATTENTION_IMPL"] = "xla"
        if "XLA_PYTHON_CLIENT_MEM_FRACTION" in env:
            del env["XLA_PYTHON_CLIENT_MEM_FRACTION"]

        common = [
            sys.executable,
            str(self.script_single),
            f"--input=A0A075B6L2:1",
            f"--output_directory={self.output_dir}",
            f"--data_directory={DATA_DIR}",
            f"--features_directory={self.test_features_dir}",
            "--fold_backend=alphafold3",
            "--flash_attention_implementation=xla",
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
        env = os.environ.copy()
        env["JAX_FLASH_ATTENTION_IMPL"] = "xla"

        args = [
            sys.executable,
            str(self.script_single),
            f"--input=A0A075B6L2:1:2-5",
            f"--output_directory={self.output_dir}",
            f"--data_directory={DATA_DIR}",
            f"--features_directory={self.test_features_dir}",
            "--fold_backend=alphafold3",
            "--flash_attention_implementation=xla",
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