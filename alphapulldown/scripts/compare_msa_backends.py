#!/usr/bin/env python3
"""Compare MSA depth/template counts without claiming prediction equivalence."""

from __future__ import annotations

import json
import lzma
from pathlib import Path

from absl import app, flags

from alphapulldown.utils.msa_quality import compare_a3m, measure_a3m, neff


flags.DEFINE_string(
    "reference_dir", None, "Directory of native/jackhmmer feature artifacts."
)
flags.DEFINE_string("candidate_dir", None, "Directory of MMseqs2 feature artifacts.")
flags.DEFINE_string("output_path", None, "JSON report path.")
flags.DEFINE_enum(
    "artifact_format",
    "af3_json",
    ["af3_json", "af2_pickle"],
    "af3_json: <name>_af3_input.json[.xz]. af2_pickle: <name>.pkl[.xz], the "
    "MonomericObject pickles AlphaFold 2 reads.",
)
FLAGS = flags.FLAGS


def _read(path: Path) -> dict:
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _artifacts(directory: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(directory.glob("*_af3_input.json*")):
        filename = path.name
        if filename.endswith(".xz"):
            filename = filename[: -len(".xz")]
        name = filename[: -len("_af3_input.json")]
        result[name] = path
    return result


def compare_directories(reference_dir: Path, candidate_dir: Path) -> list[dict]:
    """Return paired, machine-readable inputs for the scientific merge gate."""
    references = _artifacts(reference_dir)
    candidates = _artifacts(candidate_dir)
    missing_candidates = sorted(references.keys() - candidates.keys())
    missing_references = sorted(candidates.keys() - references.keys())
    if missing_candidates or missing_references:
        details = []
        if missing_candidates:
            details.append("missing from candidate: " + ", ".join(missing_candidates))
        if missing_references:
            details.append("missing from reference: " + ", ".join(missing_references))
        raise ValueError("Artifact sets differ; " + "; ".join(details))
    rows = []
    for name in sorted(references):
        reference = _read(references[name])
        candidate = _read(candidates[name])
        reference_protein = reference["sequences"][0]["protein"]
        candidate_protein = candidate["sequences"][0]["protein"]
        if reference_protein["sequence"] != candidate_protein["sequence"]:
            raise ValueError(f"Sequence mismatch for {name!r}")
        length = len(reference_protein["sequence"])
        rows.append(
            {
                "name": name,
                "reference_path": str(references[name]),
                "candidate_path": str(candidates[name]),
                "reference_unpaired": measure_a3m(
                    reference_protein["unpairedMsa"], query_length=length
                ),
                "candidate_unpaired": measure_a3m(
                    candidate_protein["unpairedMsa"], query_length=length
                ),
                "reference_paired": measure_a3m(
                    reference_protein["pairedMsa"], query_length=length
                ),
                "candidate_paired": measure_a3m(
                    candidate_protein["pairedMsa"], query_length=length
                ),
                # Depth ratios alone cannot say whether the two backends found the
                # SAME sequences; these do.
                "unpaired_overlap": compare_a3m(
                    reference_protein["unpairedMsa"], candidate_protein["unpairedMsa"]
                ),
                "paired_overlap": compare_a3m(
                    reference_protein["pairedMsa"], candidate_protein["pairedMsa"]
                ),
                "reference_unpaired_neff": neff(reference_protein["unpairedMsa"]),
                "candidate_unpaired_neff": neff(candidate_protein["unpairedMsa"]),
                "reference_template_count": len(
                    reference_protein.get("templates") or []
                ),
                "candidate_template_count": len(
                    candidate_protein.get("templates") or []
                ),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    """Aggregate paired counts while retaining zero-depth cases explicitly."""
    if not rows:
        return {"protein_count": 0}

    def mean(key: str, backend: str) -> float:
        return sum(row[f"{backend}_{key}"]["depth"] for row in rows) / len(rows)

    return {
        "protein_count": len(rows),
        "mean_reference_unpaired_depth": mean("unpaired", "reference"),
        "mean_candidate_unpaired_depth": mean("unpaired", "candidate"),
        "mean_reference_paired_depth": mean("paired", "reference"),
        "mean_candidate_paired_depth": mean("paired", "candidate"),
        # The headline numbers. Recall is what "did MMseqs2 find what jackhmmer
        # found" actually means; the depth ratio it replaces could exceed 1.
        "mean_unpaired_recall": sum(
            row["unpaired_overlap"]["recall"] for row in rows
        ) / len(rows),
        "mean_paired_recall": sum(
            row["paired_overlap"]["recall"] for row in rows
        ) / len(rows),
        "mean_unpaired_jaccard": sum(
            row["unpaired_overlap"]["jaccard"] for row in rows
        ) / len(rows),
        "mean_reference_unpaired_neff": sum(
            row["reference_unpaired_neff"] for row in rows
        ) / len(rows),
        "mean_candidate_unpaired_neff": sum(
            row["candidate_unpaired_neff"] for row in rows
        ) / len(rows),
        "total_reference_templates": sum(
            row["reference_template_count"] for row in rows
        ),
        "total_candidate_templates": sum(
            row["candidate_template_count"] for row in rows
        ),
    }


def _af2_artifacts(directory: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(directory.glob("*.pkl*")):
        filename = path.name
        for suffix in (".pkl.xz", ".pkl"):
            if filename.endswith(suffix):
                result[filename[: -len(suffix)]] = path
                break
    return result


def _af2_side(features: dict) -> dict:
    """What one AlphaFold 2 feature set offers the model, measured."""
    import numpy as np

    # The alphabet AlphaFold 2 encodes MSAs in, spelled out without importing it.
    from alphapulldown.utils.af2_to_af3_msa import AF2_ID_TO_A3M

    msa = np.asarray(features["msa"])
    length = int(msa.shape[1])
    unpaired = "".join(
        f">row{index}\n{''.join(AF2_ID_TO_A3M[int(i)] for i in row)}\n"
        for index, row in enumerate(msa)
    )
    deletions = np.asarray(features["deletion_matrix_int"])
    species = {
        value.decode() if isinstance(value, bytes) else str(value)
        for value in features.get("msa_species_identifiers_all_seq", [])
    }
    species.discard("")
    templates = [
        name for name in features.get("template_domain_names", [])
        if (name.decode() if isinstance(name, bytes) else str(name))
    ]
    return {
        "unpaired": measure_a3m(unpaired, query_length=length),
        "unpaired_neff": neff(unpaired),
        # Rows carrying at least one insertion: all zero for a local MMseqs2
        # alignment built from result2msa mode 2 alone.
        "rows_with_insertions": int((deletions.sum(axis=1) > 0).sum()),
        "paired_depth": int(np.asarray(features.get("msa_all_seq", msa[:1])).shape[0]),
        # Pairing needs a species in common; distinct labels are what it has to use.
        "paired_species": len(species),
        "template_count": len(templates),
    }


def compare_af2_directories(reference_dir: Path, candidate_dir: Path) -> list[dict]:
    """Paired measurements of two directories of AlphaFold 2 feature pickles.

    No homolog overlap: a pickle stores its alignment as integer rows without the
    sequence headers, so there is no accession to match on, and residue strings
    are not a substitute -- two backends align the same homolog over different
    extents. Compare the raw alignments for overlap; this compares what the
    model is given.
    """
    from alphapulldown.utils.lightweight_pickles import (
        extract_feature_dict,
        load_lightweight_pickle,
    )

    references = _af2_artifacts(reference_dir)
    candidates = _af2_artifacts(candidate_dir)
    missing = sorted(references.keys() ^ candidates.keys())
    if missing:
        raise ValueError("Artifact sets differ: " + ", ".join(missing))
    rows = []
    for name in sorted(references):
        reference = load_lightweight_pickle(references[name])
        candidate = load_lightweight_pickle(candidates[name])
        if reference.sequence != candidate.sequence:
            raise ValueError(f"Sequence mismatch for {name!r}")
        rows.append(
            {
                "name": name,
                "reference_path": str(references[name]),
                "candidate_path": str(candidates[name]),
                "reference": _af2_side(extract_feature_dict(reference)),
                "candidate": _af2_side(extract_feature_dict(candidate)),
            }
        )
    return rows


def summarize_af2(rows: list[dict]) -> dict:
    if not rows:
        return {"protein_count": 0}

    def mean(side: str, *path: str) -> float:
        total = 0.0
        for row in rows:
            value = row[side]
            for key in path:
                value = value[key]
            total += value
        return total / len(rows)

    summary = {"protein_count": len(rows)}
    for side in ("reference", "candidate"):
        summary.update(
            {
                f"mean_{side}_unpaired_depth": mean(side, "unpaired", "depth"),
                f"mean_{side}_unpaired_neff": mean(side, "unpaired_neff"),
                f"mean_{side}_rows_with_insertions": mean(side, "rows_with_insertions"),
                f"mean_{side}_paired_depth": mean(side, "paired_depth"),
                f"mean_{side}_paired_species": mean(side, "paired_species"),
                f"mean_{side}_template_count": mean(side, "template_count"),
            }
        )
    return summary


def main(argv) -> None:
    del argv
    if FLAGS.artifact_format == "af2_pickle":
        proteins = compare_af2_directories(
            Path(FLAGS.reference_dir), Path(FLAGS.candidate_dir)
        )
        summary = summarize_af2(proteins)
    else:
        proteins = compare_directories(
            Path(FLAGS.reference_dir), Path(FLAGS.candidate_dir)
        )
        summary = summarize(proteins)
    report = {
        "schemaVersion": 1,
        "warning": (
            "MSA/template metrics are diagnostic only; run matched inference and "
            "DockQ against experimental references before claiming accuracy equivalence."
        ),
        "artifactFormat": FLAGS.artifact_format,
        "summary": summary,
        "proteins": proteins,
    }
    Path(FLAGS.output_path).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    flags.mark_flags_as_required(["reference_dir", "candidate_dir", "output_path"])
    app.run(main)
