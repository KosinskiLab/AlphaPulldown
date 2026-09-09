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


def _accession(header: str) -> str:
    """Database accession for one A3M header, ignoring how it was aligned.

    Backends describe the same hit differently: jackhmmer writes
    ``UniRef90_A0A837XVS7/4-106 [subseq from] ...`` while MMseqs2 writes
    ``UniRef90_A0A837XVS7 ...``. The trailing ``/start-end`` is the aligned RANGE,
    not part of the identity, so it is dropped.
    """
    token = header.split()[0] if header.split() else ""
    base, slash, span = token.rpartition("/")
    if slash and span and all(c.isdigit() or c == "-" for c in span):
        token = base
    return token


def _homologs(a3m: str) -> list[str]:
    """The hits an alignment contains, keyed so two backends can be compared.

    Keyed by accession, NOT by residue string. Two backends align the same homolog
    over slightly different extents -- one row may start a residue earlier -- so
    stripping gaps yields different strings for the same sequence and a set
    comparison would report near-zero overlap between alignments that are in fact
    largely the same. Falls back to the residue string when a header carries no
    usable accession. The query row is dropped.
    """
    out = []
    header = None
    for line in a3m.splitlines():
        if line.startswith(">"):
            header = line[1:]
            continue
        if header is None or not line.strip():
            continue
        accession = _accession(header)
        if accession:
            out.append(accession)
        else:
            residues = "".join(r for r in line if r.isupper() and r != "-")
            if residues:
                out.append(residues)
        header = None
    return out[1:] if out else out


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
