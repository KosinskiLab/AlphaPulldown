#!/usr/bin/env python3
"""Compare MSA depth/template counts without claiming prediction equivalence."""

from __future__ import annotations

import json
import lzma
from pathlib import Path

from absl import app, flags

from alphapulldown.utils.msa_quality import measure_a3m


flags.DEFINE_string(
    "reference_dir", None, "Directory of native/jackhmmer AF3 JSON artifacts."
)
flags.DEFINE_string("candidate_dir", None, "Directory of MMseqs2 AF3 JSON artifacts.")
flags.DEFINE_string("output_path", None, "JSON report path.")
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
        "total_reference_templates": sum(
            row["reference_template_count"] for row in rows
        ),
        "total_candidate_templates": sum(
            row["candidate_template_count"] for row in rows
        ),
    }


def main(argv) -> None:
    del argv
    proteins = compare_directories(Path(FLAGS.reference_dir), Path(FLAGS.candidate_dir))
    report = {
        "schemaVersion": 1,
        "warning": (
            "MSA/template metrics are diagnostic only; run matched inference and "
            "DockQ against experimental references before claiming accuracy equivalence."
        ),
        "summary": summarize(proteins),
        "proteins": proteins,
    }
    Path(FLAGS.output_path).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    flags.mark_flags_as_required(["reference_dir", "candidate_dir", "output_path"])
    app.run(main)
