"""Test data helpers shared by the AF3 tests in test/integration and test/cluster.

``Af3TestData`` reads the repository's test data; a test class using it sets
``test_fastas_dir``, ``test_features_dir`` and ``output_dir``.
"""
from __future__ import annotations

import json
import lzma
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from alphafold3.constants import residue_names as af3_residue_names

from alphapulldown.objects import MultimericObject
from alphapulldown.utils.modelling_setup import (
    create_custom_info,
    create_interactors,
    parse_fold,
)


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


def _non_empty_a3m_payload_rows(a3m_text: str) -> list[str]:
    return _a3m_payload_sequences(a3m_text) if a3m_text else []


def _load_feature_dict(feature_path: Path) -> dict[str, Any]:
    payload = _load_feature_payload(feature_path)
    if hasattr(payload, "feature_dict"):
        return payload.feature_dict
    return payload


def _load_feature_payload(feature_path: Path) -> Any:
    opener = lzma.open if feature_path.suffix == ".xz" else open
    with opener(feature_path, "rb") as handle:
        return pickle.load(handle)


def _write_feature_payload(feature_path: Path, payload: Any) -> None:
    opener = lzma.open if feature_path.suffix == ".xz" else open
    with opener(feature_path, "wb") as handle:
        pickle.dump(payload, handle)


def _non_empty_identifier_count(values) -> int:
    count = 0
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if str(value).strip():
            count += 1
    return count


class Af3TestData:
    """Sequences and AF3 fold inputs built from the repository's test data."""

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
