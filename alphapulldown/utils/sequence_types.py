"""Lightweight sequence classification shared by AlphaFold feature adapters."""

from __future__ import annotations

import re


AF3_DNA_BASES = frozenset("ACGTN")
AF3_RNA_BASES = frozenset("ACGUN")
AF3_PROTEIN_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYX")
AF3_PROTEIN_ONLY_RESIDUES = AF3_PROTEIN_RESIDUES - (AF3_DNA_BASES | {"U"})


def get_af3_chain_kind(description: str, sequence: str) -> str:
    """Infer an AF3 chain kind, requiring a hint for ambiguous alphabets."""
    residues = set(sequence.upper())
    if not residues:
        raise ValueError("Sequence is empty.")

    invalid_residues = residues - (AF3_PROTEIN_RESIDUES | {"U"})
    if invalid_residues:
        invalid_list = ", ".join(sorted(invalid_residues))
        raise ValueError(f"Invalid sequence residues: {invalid_list}")

    if residues <= AF3_RNA_BASES and "U" in residues:
        return "rna"
    if residues & AF3_PROTEIN_ONLY_RESIDUES:
        return "protein"

    description_tokens = {
        token for token in re.split(r"[^A-Za-z0-9]+", description.lower()) if token
    }
    if "dna" in description_tokens:
        return "dna"
    if "rna" in description_tokens:
        return "rna"
    if {"protein", "prot", "peptide"} & description_tokens:
        return "protein"

    raise ValueError(
        "Ambiguous sequence alphabet. Add 'DNA', 'RNA', or 'protein' to "
        f"the FASTA description for '{description}' to disambiguate."
    )
