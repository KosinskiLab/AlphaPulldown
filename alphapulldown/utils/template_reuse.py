"""Re-run template search against existing features, keeping their MSAs.

Motivation. Template search is cheap; MSA search is not. When the template
database moves underneath a set of features -- a refreshed ``pdb_seqres``, a
corrected ``max_template_date`` -- the MSAs in those features are still
perfectly good and only the ``template_*`` block is stale. Regenerating from
scratch would redo hours of jackhmmer/HHblits work to change minutes of
hmmsearch output.

What AlphaFold2 actually searches with. ``DataPipeline.process`` does not build
the template profile from the merged MSA that ends up in the features. It uses
the *uniref90* Stockholm alignment alone::

    msa_for_templates = jackhmmer_uniref90_result['sto']
    msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
    msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(...)
    pdb_templates_result = template_searcher.query(msa_for_templates)

So reproducing a template search faithfully means recovering that file, which
is why :func:`msa_for_template_search` prefers ``uniref90_hits.sto`` on disk and
treats reconstruction from the pickled features as a clearly-labelled fallback.
The fallback is not equivalent: the features' ``msa`` is the *merged* alignment
(uniref90 + mgnify + BFD), so the profile built from it is deeper and can
return different hits. That is a defensible result but not a reproduction of
what a fresh run would do, so callers are expected to surface which source was
used rather than let it pass silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from absl import logging

from alphapulldown.utils.msa_encoding import ids_to_a3m
from alphapulldown.utils.msa_integrity import check_msa_file

# The alignment AF2 builds its template profile from.
UNIREF90_STO = "uniref90_hits.sto"

SOURCE_UNIREF90_FILE = "uniref90_hits.sto"
SOURCE_RECONSTRUCTED = "reconstructed-from-features"


@dataclass(frozen=True)
class TemplateSearchMsa:
    """A Stockholm alignment to drive template search, and where it came from."""

    stockholm: str
    source: str

    @property
    def is_reconstructed(self) -> bool:
        return self.source == SOURCE_RECONSTRUCTED


def _iter_a3m_records(a3m_text: str):
    """Yield ``(name, sequence)`` from A3M/FASTA text, dropping insertions.

    A3M marks insertions relative to the query with lowercase letters. Removing
    them leaves every row at query length, which is what an alignment handed to
    hmmbuild has to be.
    """
    name = None
    chunks: list[str] = []
    for line in a3m_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(chunks)
            name = line[1:].split()[0] if len(line) > 1 else "seq"
            chunks = []
        elif name is not None:
            chunks.append("".join(ch for ch in line if not ch.islower()))
    if name is not None:
        yield name, "".join(chunks)


def stockholm_from_a3m(a3m_text: str) -> str:
    """Convert insertion-free A3M text into a minimal Stockholm alignment.

    Emits the banner, one ``name sequence`` row per record, a ``#=GC RF``
    reference annotation, and the ``//`` terminator.

    The reference annotation is not decorative. ``remove_empty_columns_from_
    stockholm_msa`` treats ``#=GC RF`` as the end of an alignment chunk and only
    then moves the buffered rows into its output; without it every row is
    dropped and the function fails with a bare ``KeyError``. Real jackhmmer
    output always carries one. Every column here is a match column (insertions
    were removed), so the annotation is all ``x``.
    """
    rows = []
    seen: dict[str, int] = {}
    width = None
    for name, seq in _iter_a3m_records(a3m_text):
        if not seq:
            continue
        if width is None:
            width = len(seq)
        elif len(seq) != width:
            # Ragged input cannot be a valid alignment; refuse rather than let
            # hmmbuild fail with a less obvious message.
            raise ValueError(
                f"alignment rows differ in length ({len(seq)} vs {width}); "
                "input is not a query-anchored alignment"
            )
        name = name.replace(" ", "_")
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        rows.append((name, seq))

    if not rows:
        raise ValueError("no sequences to build a Stockholm alignment from")

    pad = max(len(name) for name, _ in rows + [("#=GC RF", "")]) + 2
    body = "\n".join(f"{name:<{pad}}{seq}" for name, seq in rows)
    reference = f"{'#=GC RF':<{pad}}{'x' * width}"
    return f"# STOCKHOLM 1.0\n{body}\n{reference}\n//\n"


def stockholm_from_feature_dict(feature_dict) -> str:
    """Rebuild a Stockholm alignment from AF2 features' integer MSA.

    ``feature_dict['msa']`` is the merged, query-length, integer-encoded
    alignment, so it decodes to A3M rows that are already insertion-free.
    """
    msa = feature_dict.get("msa")
    if msa is None:
        raise ValueError("features contain no 'msa' array to reconstruct from")
    msa = np.asarray(msa)
    if msa.ndim != 2 or msa.size == 0:
        raise ValueError(f"features have an unusable 'msa' array of shape {msa.shape}")
    return stockholm_from_a3m(ids_to_a3m(msa))


def msa_for_template_search(monomer, msa_output_dir) -> TemplateSearchMsa:
    """Recover an alignment to re-run template search for an existing monomer.

    Prefers the on-disk ``uniref90_hits.sto`` because that is what AlphaFold2
    itself searches with; falls back to reconstructing the merged alignment from
    the stored features when MSA files were not kept (``--save_msa_files=False``
    deletes them once features are written).
    """
    sto_path = Path(msa_output_dir) / UNIREF90_STO
    problem = check_msa_file(sto_path) if sto_path.exists() else "not present"
    if problem is None:
        return TemplateSearchMsa(sto_path.read_text(), SOURCE_UNIREF90_FILE)

    logging.warning(
        "Cannot use %s (%s); rebuilding the template-search alignment from the "
        "stored features instead. That alignment is the merged MSA rather than "
        "uniref90 alone, so hits may differ from a fresh run.",
        sto_path, problem,
    )
    return TemplateSearchMsa(
        stockholm_from_feature_dict(monomer.feature_dict), SOURCE_RECONSTRUCTED
    )


def search_templates(
    template_searcher,
    template_featurizer,
    *,
    query_sequence: str,
    stockholm_msa: str,
    msa_output_dir=None,
):
    """Run template search + featurisation, mirroring ``DataPipeline.process``.

    Returns the ``template_*`` feature mapping.
    """
    from alphafold.data import parsers

    # remove_empty_columns_from_stockholm_msa uses '#=GC RF' to delimit an
    # alignment chunk and silently drops every row without one, then dies with a
    # bare KeyError from its own bookkeeping. jackhmmer always writes the
    # annotation, so its absence means the alignment is not what it claims to
    # be - say so here rather than 30 frames down.
    if "#=GC RF" not in stockholm_msa:
        raise ValueError(
            "Stockholm alignment has no '#=GC RF' reference annotation; it "
            "cannot be used for template search. Genuine jackhmmer output "
            "always contains one, so this file is truncated or not Stockholm."
        )

    msa_for_templates = parsers.deduplicate_stockholm_msa(stockholm_msa)
    msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
        msa_for_templates
    )

    if template_searcher.input_format == "sto":
        query = msa_for_templates
    elif template_searcher.input_format == "a3m":
        query = parsers.convert_stockholm_to_a3m(msa_for_templates)
    else:
        raise ValueError(
            f"Unrecognized template input format: {template_searcher.input_format}"
        )

    pdb_templates_result = template_searcher.query(query)

    if msa_output_dir is not None:
        hits_path = Path(msa_output_dir) / (
            f"pdb_hits.{template_searcher.output_format}"
        )
        hits_path.parent.mkdir(parents=True, exist_ok=True)
        hits_path.write_text(pdb_templates_result)

    hits = template_searcher.get_template_hits(
        output_string=pdb_templates_result, input_sequence=query_sequence
    )
    result = template_featurizer.get_templates(
        query_sequence=query_sequence, hits=hits
    )
    return dict(result.features)
