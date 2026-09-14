"""Joining two MMseqs2 formats into one A3M with headers AND insertions."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

import pytest

from alphapulldown.utils.msa_formats import (
    StitchMismatch,
    stitch_headers_and_insertions,
    strip_insertions,
)


def test_conversion_does_not_import_alphafold():
    """The whole point: this must work in the AlphaFold 2 image.

    Checked in a subprocess, because the test session may already have imported
    AlphaFold for other modules and would mask exactly what is being measured.
    """
    probe = (
        "import sys;"
        "from alphapulldown.utils.msa_formats import stitch_headers_and_insertions as f;"
        "f([('q', 'MKTAYI')], [('q', 'MKTAYI')]);"
        "print(','.join(n for n in sys.modules if n.startswith('alphafold')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == ""


def test_strip_insertions_leaves_one_column_per_query_position():
    # Lowercase residues are insertions; gaps and match residues are columns.
    assert strip_insertions("MKTwwAY-I") == "MKTAY-I"


def test_stitch_takes_headers_from_one_pass_and_insertions_from_the_other():
    headers = [("query", "MKTAYI"), ("sp|P1|A_HUMAN full header", "MKTAYI")]
    insertions = [("query", "MKTAYI"), ("P1", "MKTwwAYI")]
    assert stitch_headers_and_insertions(headers, insertions) == [
        ("query", "MKTAYI"),
        ("sp|P1|A_HUMAN full header", "MKTwwAYI"),
    ]


def test_stitch_refuses_rows_that_disagree():
    """A silent positional join here would pair a header with another hit's
    sequence -- for the paired database, pairing chains by the wrong species."""
    headers = [("query", "MKTAYI"), ("sp|P1|A_HUMAN", "MKTAYI")]
    reordered = [("query", "MKTAYI"), ("P2", "MKTAYQ")]
    with pytest.raises(StitchMismatch, match="no longer in the same order"):
        stitch_headers_and_insertions(headers, reordered)


def test_stitch_refuses_differing_row_counts():
    with pytest.raises(StitchMismatch, match="different row counts"):
        stitch_headers_and_insertions([("query", "MKT")], [])


def _read_records(path: Path) -> list[tuple[str, str]]:
    records, description, parts = [], None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if description is not None:
                records.append((description, "".join(parts)))
            description, parts = line[1:], []
        elif description is not None:
            parts.append(line)
    if description is not None:
        records.append((description, "".join(parts)))
    return records


REAL_MODE2 = Path(__file__).parent / "data" / "small_bfd.mode2.txt"
REAL_MODE5 = Path(__file__).parent / "data" / "small_bfd.mode5.txt"


@pytest.mark.skipif(
    not (REAL_MODE2.exists() and REAL_MODE5.exists()),
    reason="real MMseqs2 output fixtures are not checked in",
)
def test_stitch_holds_on_real_mmseqs_output():
    """The synthetic cases prove the logic; this proves the assumption.

    Fixtures are the two passes over one real search of a 586-residue query
    against small_bfd, produced by the pinned MMseqs2 build.

    Note which database this is. How much a header loses in mode 5 depends on the
    FASTA it was built from: small_bfd headers are a single token
    (``A0A163VT74_9BACL``) and survive intact, so the stitch changes only the
    sequences here. It is uniprot that loses the species --
    ``sp|P83570|GWA_SEPOF …`` becomes a bare ``P83570`` -- and uniprot is exactly
    where species decides chain pairing. The insertion half of the join is what
    this fixture exercises; ``test_stitch_recovers_uniprot_species_headers``
    covers the other half.
    """
    headers = _read_records(REAL_MODE2)
    insertions = _read_records(REAL_MODE5)
    stitched = stitch_headers_and_insertions(headers, insertions)

    assert len(stitched) == len(headers) > 1
    assert [description for description, _ in stitched] == [
        description for description, _ in headers
    ]
    assert [sequence for _, sequence in stitched] == [
        sequence for _, sequence in insertions
    ]
    # The point of the exercise: insertions the header pass discarded are back.
    recovered = sum(
        1 for _, sequence in stitched for residue in sequence if residue.islower()
    )
    assert recovered > 0, "insertions must survive the stitch"
    assert not any(
        residue.islower() for _, sequence in headers for residue in sequence
    ), "the header pass is expected to carry no insertions at all"


UNIPROT_MODE2 = Path(__file__).parent / "data" / "uniprot.mode2.txt"
UNIPROT_MODE5 = Path(__file__).parent / "data" / "uniprot.mode5.txt"


@pytest.mark.skipif(
    not (UNIPROT_MODE2.exists() and UNIPROT_MODE5.exists()),
    reason="real uniprot MMseqs2 output fixtures are not checked in",
)
def test_stitch_recovers_uniprot_species_headers():
    """The half that decides whether multimer pairing works at all.

    AlphaFold 2 reads the species mnemonic out of ``sp|ACC|NAME_SPECIES``. Mode 5
    reduces that to ``ACC``, so an alignment built from it alone would pair no
    chains; the stitched alignment must carry the parseable form.
    """
    headers = _read_records(UNIPROT_MODE2)
    insertions = _read_records(UNIPROT_MODE5)
    stitched = stitch_headers_and_insertions(headers, insertions)

    # Mirrors alphafold.data.msa_identifiers._UNIPROT_PATTERN, which is what
    # actually decides whether a row can be paired. Reproduced rather than
    # imported because this module must stay free of AlphaFold; the authoritative
    # check against the real function belongs with the AF2 finalizer tests.
    uniprot = re.compile(
        r"^(?:tr|sp)\|[A-Za-z0-9]{6,10}(?:_\d)?\|[A-Za-z0-9]+_([A-Za-z0-9]{1,5})"
        r"(?:_\d+)?$"
    )

    def species(records):
        found = set()
        for description, _ in records:
            match = uniprot.match(description.split()[0])
            if match:
                found.add(match.group(1))
        return found

    stitched_species = species(stitched)
    assert len(stitched_species) > 1, (
        "the stitched alignment must yield species mnemonics; without them "
        "AlphaFold 2 pairs no chains at all"
    )
    assert species(insertions) == set(), (
        "the insertion pass is expected to yield none, which is why it is stitched"
    )
    assert species(headers) == stitched_species
    assert (
        sum(1 for _, sequence in stitched for residue in sequence if residue.islower())
        > 0
    )
