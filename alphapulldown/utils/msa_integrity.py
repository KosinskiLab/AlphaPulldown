"""Integrity checks for precomputed MSA files.

``--use_precomputed_msas`` makes AlphaFold reuse whatever alignment files it
finds next to the output, which is what makes restartable feature generation
cheap. The failure mode is that it trusts them: a job killed mid-write, or one
whose MSA tool exited non-zero after creating the output file, leaves a
zero-length or truncated alignment behind. The next run reads it back without
complaint and either

* dies far from the cause -- an empty ``uniref90_hits.sto`` surfaces as
  ``StopIteration`` inside ``parsers.deduplicate_stockholm_msa``, which names
  neither the file nor the protein; or
* worse, silently succeeds on a *partial* alignment and produces features whose
  MSA depth is quietly wrong.

The second case is the reason these checks exist: a crash is recoverable, but
degraded features are indistinguishable from good ones downstream.

The checks here are deliberately structural rather than semantic. They ask
whether a file is a complete alignment of the expected format, not whether its
content is biologically sensible -- an alignment can be legitimately shallow.
"""

from __future__ import annotations

import gzip
import lzma
import zlib
from dataclasses import dataclass
from pathlib import Path

from absl import logging

# Suffixes AlphaFold/AlphaPulldown write alignments to, mapped to their format.
MSA_SUFFIXES = {
    ".sto": "sto",
    ".a3m": "a3m",
    ".fasta": "a3m",   # same structural rules: '>' headers plus residue lines
}

# Files that are not alignments and must never be judged by these rules.
NON_MSA_SUFFIXES = {".hhr", ".hmm", ".pkl", ".json"}

# Template-search *results*. hmmsearch writes these in Stockholm format, so they
# would otherwise pass for alignments and be deleted when a search legitimately
# returned no hits. They are outputs, not inputs, and are regenerated anyway.
NON_MSA_STEMS = {"pdb_hits"}


@dataclass(frozen=True)
class MsaProblem:
    """One structurally invalid alignment file."""

    path: Path
    reason: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.path}: {self.reason}"


def _read_text(path: Path) -> str:
    """Read an alignment file, transparently handling .gz and .xz."""
    name = path.name
    if name.endswith(".gz"):
        with gzip.open(path, "rt", errors="replace") as handle:
            return handle.read()
    if name.endswith(".xz"):
        with lzma.open(path, "rt", errors="replace") as handle:
            return handle.read()
    return path.read_text(errors="replace")


def _effective_suffix(path: Path) -> str | None:
    """Alignment suffix of a path, looking through a .gz/.xz wrapper."""
    name = path.name
    for wrapper in (".gz", ".xz"):
        if name.endswith(wrapper):
            name = name[: -len(wrapper)]
            break
    stem_path = Path(name)
    if stem_path.suffix in NON_MSA_SUFFIXES or stem_path.stem in NON_MSA_STEMS:
        return None
    return MSA_SUFFIXES.get(stem_path.suffix)


def check_stockholm(text: str) -> str | None:
    """Structural problem with a Stockholm alignment, or None if it is sound.

    jackhmmer writes a ``# STOCKHOLM 1.0`` banner, one or more alignment rows,
    and a closing ``//``. The terminator is what makes truncation detectable:
    a file cut off mid-write keeps a valid-looking header and some rows.
    """
    if not text.strip():
        return "empty Stockholm file"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines[0].startswith("# STOCKHOLM"):
        return "missing '# STOCKHOLM' header"
    if not any(ln.strip() == "//" for ln in lines):
        return "missing '//' terminator (file is truncated)"
    # Alignment rows are the non-comment, non-terminator lines.
    rows = [
        ln for ln in lines
        if not ln.startswith("#") and ln.strip() != "//"
    ]
    if not rows:
        return "no alignment rows"
    return None


def check_a3m(text: str) -> str | None:
    """Structural problem with an A3M/FASTA alignment, or None if it is sound."""
    if not text.strip():
        return "empty A3M file"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    headers = [i for i, ln in enumerate(lines) if ln.startswith(">")]
    if not headers:
        return "no '>' header lines"
    # The final record must actually carry residues; a run killed just after
    # writing a header leaves a danging description with no sequence.
    if headers[-1] == len(lines) - 1:
        return "last record has a header but no sequence (file is truncated)"
    return None


def check_msa_file(path) -> str | None:
    """Structural problem with one alignment file, or None if it is sound.

    Returns None for paths that are not alignments, so callers can hand this
    whole directory listings.
    """
    path = Path(path)
    fmt = _effective_suffix(path)
    if fmt is None:
        return None
    if not path.is_file():
        return "file does not exist"
    if path.stat().st_size == 0:
        return "file is empty (0 bytes)"
    try:
        text = _read_text(path)
    except (OSError, EOFError, lzma.LZMAError, gzip.BadGzipFile, zlib.error) as exc:
        # A compressed alignment truncated mid-stream fails to decompress; that
        # is itself the finding, not an error to propagate.
        return f"unreadable ({exc.__class__.__name__}: {exc})"
    return check_stockholm(text) if fmt == "sto" else check_a3m(text)


def validate_precomputed_msas(msa_dir, *, remove_invalid: bool = False):
    """Check every alignment in ``msa_dir``; optionally delete the bad ones.

    Args:
      msa_dir: directory holding a protein's alignment files.
      remove_invalid: when True, unlink each unsound file so the caller's next
        MSA run regenerates it instead of reusing it. This is the safe default
        for automated reruns: recomputing one alignment costs CPU time, whereas
        reusing a truncated one corrupts the features.

    Returns:
      A list of :class:`MsaProblem`, empty when everything is sound.
    """
    msa_dir = Path(msa_dir)
    if not msa_dir.is_dir():
        return []

    problems: list[MsaProblem] = []
    for entry in sorted(msa_dir.iterdir()):
        if not entry.is_file():
            continue
        reason = check_msa_file(entry)
        if reason is None:
            continue
        problems.append(MsaProblem(entry, reason))
        if remove_invalid:
            try:
                entry.unlink()
                logging.warning(
                    "Removed unusable precomputed MSA %s (%s); it will be "
                    "regenerated.", entry, reason,
                )
            except OSError as exc:
                logging.error("Could not remove unusable MSA %s: %s", entry, exc)
        else:
            logging.warning("Unusable precomputed MSA %s (%s)", entry, reason)
    return problems
