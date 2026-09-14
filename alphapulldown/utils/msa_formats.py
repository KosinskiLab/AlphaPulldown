"""Turn MMseqs2 output into A3M that keeps both its headers and its insertions.

No AlphaFold dependency of any kind. The search stage used to convert results
with ``alphafold3.cpp.msa_conversion``, and the AlphaFold 2 image installs
AlphaPulldown without the ``alphafold3`` extra, so the shared stage imported
cleanly there and then died on the first search result.

The formats. ``mmseqs result2msa`` offers several, and none gives both things an
AlphaFold alignment needs. Measured on the pinned build
(8cc5ce367b5638c4306c2d7cfc652dd099a4643f), with a target carrying a deliberate
four-residue insertion::

    mode 2   >sp|P00002|INSERT_MOUSE has a 4-residue insertion OS=Mus musculus
             MKTAYIAKQRQISFVKSHFSRQLEE...      <- the WWWW is gone
    mode 5   >P00002
             MKTAYIAKQRQISFVKSHwwwwFSRQLEE...  <- kept, as an A3M insertion

Mode 2 keeps the full header and drops every column the query does not span, so
an alignment built from it alone has no insertions and AlphaFold derives an
all-zero deletion matrix from it. That is not a small detail: against a real
586-residue query, 98 of 120 small_bfd hits (81.7%) and 712 of 824 uniprot hits
(86.4%) carry insertions, 13035 residues of them on uniprot alone.

Mode 5 keeps the insertions and cuts the header to the database key. How much
that loses depends on the FASTA the database was built from: small_bfd headers
are a single token and survive whole, but uniprot's ``sp|P83570|GWA_SEPOF …``
becomes ``P83570`` -- and AlphaFold 2 reads the species out of the long form, and
species is what pairs chains in a complex. The same holds for nucleotide
searches, so RNA takes the same route.

So both are run over one search result and joined by
:func:`stitch_headers_and_insertions`. The second pass costs little: 2 s against
a 947 s search on small_bfd, 13 s against 3739 s on uniprot, roughly 16 ms a hit.
"""

from __future__ import annotations

from typing import Sequence


def strip_insertions(a3m_row: str) -> str:
    """Drop the insertion residues, leaving one column per query position."""
    return "".join(residue for residue in a3m_row if not residue.islower())


class StitchMismatch(ValueError):
    """Two result2msa passes over one result set disagreed about their rows."""


def stitch_headers_and_insertions(
    headers_from: Sequence[tuple[str, str]],
    insertions_from: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Join mode-2 headers to mode-5 sequences, verifying the join row by row.

    Both arguments are ``(description, sequence)`` records read from one search
    result formatted twice: ``headers_from`` carries the full database headers and
    no insertions, ``insertions_from`` carries the insertions and a bare accession.
    MMseqs2 emits both in result order, so they line up positionally.

    Positional joins are exactly the kind that rot silently, so this one is
    checked rather than trusted: removing the insertions from a mode-5 row has to
    reproduce the mode-2 row character for character, on every row. If a future
    MMseqs2 ever reorders, filters or realigns one pass and not the other, this
    raises instead of emitting a chimeric alignment in which headers describe the
    wrong sequences -- which for the paired database would mean pairing chains by
    the wrong species, a wrong answer that looks entirely plausible.

    Verified against real data: 120/120 rows on small_bfd, and 31/31 on a
    synthetic set built with random insertions and deletions.
    """
    if len(headers_from) != len(insertions_from):
        raise StitchMismatch(
            "the two result2msa passes returned different row counts "
            f"({len(headers_from)} with headers, {len(insertions_from)} with "
            "insertions); they cannot describe the same search result"
        )
    stitched = []
    for index, ((description, aligned), (_, with_insertions)) in enumerate(
        zip(headers_from, insertions_from)
    ):
        if strip_insertions(with_insertions) != aligned:
            raise StitchMismatch(
                f"row {index} differs between the two result2msa passes: "
                f"{strip_insertions(with_insertions)!r} from the insertion pass "
                f"does not match {aligned!r} from the header pass. The two are "
                "no longer in the same order, so headers cannot be trusted to "
                "describe these sequences"
            )
        stitched.append((description, with_insertions))
    return stitched
