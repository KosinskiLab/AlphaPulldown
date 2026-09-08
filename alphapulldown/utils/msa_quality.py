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


def _homologs(a3m: str) -> list[str]:
    """Sequences reduced to the homolog itself, so two backends are comparable.

    An A3M encodes alignment to ITS OWN query: gaps and lowercase insertion columns
    differ between backends even when the underlying hit is the same sequence.
    Stripping both leaves the residue string that was actually found, which is what
    a set comparison has to be built on. The query row is dropped.
    """
    out = []
    for sequence in _sequences(a3m)[1:]:
        residues = "".join(
            residue for residue in sequence if residue.isupper() and residue != "-"
        )
        if residues:
            out.append(residues)
    return out


def neff(a3m: str, *, identity: float = 0.8, sample: int = 2000) -> float:
    """Effective sequence count: sequences weighted by how redundant they are.

    Raw depth is a poor proxy for what an MSA is worth -- a thousand near-identical
    sequences carry roughly the information of one. Neff downweights each sequence by
    the size of its neighbourhood at `identity`, and it is Neff, not count, that
    tracks prediction quality.

    Pairwise comparison is quadratic, so above `sample` sequences this measures a
    deterministic evenly-spaced subsample and rescales. That makes it an estimate;
    it is reported as such rather than presented as exact.
    """
    sequences = _sequences(a3m)
    aligned = ["".join(r for r in s if not r.islower()) for s in sequences]
    aligned = [s for s in aligned if s]
    if not aligned:
        return 0.0
    total = len(aligned)
    if total > sample:
        step = total / sample
        aligned = [aligned[int(i * step)] for i in range(sample)]
    width = min(len(s) for s in aligned)
    if width == 0:
        return 0.0
    trimmed = [s[:width] for s in aligned]
    threshold = identity * width
    weights = 0.0
    for i, a in enumerate(trimmed):
        neighbours = 1
        for j, b in enumerate(trimmed):
            if i == j:
                continue
            if sum(x == y for x, y in zip(a, b)) >= threshold:
                neighbours += 1
        weights += 1.0 / neighbours
    return weights * (total / len(trimmed))


def compare_a3m(reference: str, candidate: str) -> dict[str, Any]:
    """How much of the reference alignment the candidate actually recovered.

    Depth ratios cannot answer this: a candidate holding entirely different sequences
    of the same count scores 1.0 on depth and 0.0 here. Recall is bounded by 1 by
    construction, unlike a ratio of counts, which is why a ratio could read above 100%.
    """
    ref = set(_homologs(reference))
    cand = set(_homologs(candidate))
    shared = ref & cand
    union = ref | cand
    return {
        "reference_unique": len(ref),
        "candidate_unique": len(cand),
        "shared": len(shared),
        # Of what the reference found, the fraction the candidate also found.
        "recall": (len(shared) / len(ref)) if ref else 0.0,
        # Of what the candidate found, the fraction the reference also found. Low
        # precision is not necessarily bad -- it may be genuine extra homologs.
        "precision": (len(shared) / len(cand)) if cand else 0.0,
        "jaccard": (len(shared) / len(union)) if union else 0.0,
        # Kept for continuity with the old report, and clearly named this time.
        "depth_ratio": (len(cand) / len(ref)) if ref else 0.0,
    }
