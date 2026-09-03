"""Backend-neutral MSA metrics for reproducible scientific comparisons."""

from __future__ import annotations

from typing import Any


def _sequences(a3m: str) -> list[str]:
    sequences = []
    parts: list[str] = []
    for line in a3m.splitlines():
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts))
            parts = []
        elif line.strip():
            parts.append(line.strip())
    if parts:
        sequences.append("".join(parts))
    return sequences


def measure_a3m(a3m: str, *, query_length: int) -> dict[str, Any]:
    """Measure raw/unique depth and mean aligned non-gap query coverage."""
    if query_length < 1:
        raise ValueError("query_length must be positive")
    sequences = _sequences(a3m)
    coverages = []
    for sequence in sequences:
        aligned = [residue for residue in sequence if not residue.islower()]
        coverages.append(
            min(1.0, sum(residue not in "-." for residue in aligned) / query_length)
        )
    return {
        "depth": len(sequences),
        "unique_depth": len(set(sequences)),
        "mean_non_gap_coverage": (
            sum(coverages) / len(coverages) if coverages else 0.0
        ),
    }
