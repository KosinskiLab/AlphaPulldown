#!/usr/bin/env python3
"""GPU-only MMseqs2 MSA stage; intentionally imports neither AlphaFold nor JAX."""

from __future__ import annotations

from pathlib import Path

from absl import app, flags, logging

from alphapulldown.feature_batch import (
    DatabaseSpec,
    MsaBatch,
    MsaBatchSettings,
    SubprocessMmseqsProcess,
    protein_requests_from_fastas,
    write_batch_summary,
)


flags.DEFINE_list("fasta_paths", None, "Paths to protein FASTA files.")
flags.DEFINE_string("msa_output_dir", None, "Durable per-protein MSA bundle directory.")
flags.DEFINE_string("summary_path", None, "Atomic whole-shard completion record.")
flags.DEFINE_string(
    "mmseqs_binary_path",
    "/opt/mmseqs/bin/mmseqs",
    "Path to the bundled GPU-capable MMseqs2 executable.",
)
flags.DEFINE_string("mmseqs_temp_dir", None, "Fast local MMseqs2 scratch directory.")
flags.DEFINE_integer(
    "mmseqs_batch_max_sequences", None, "Maximum unique sequences per query database."
)
flags.DEFINE_integer(
    "mmseqs_batch_max_residues", None, "Maximum residues per query database."
)
flags.DEFINE_float("mmseqs_e_value", 1e-4, "MMseqs2 search E-value cutoff.")
flags.DEFINE_integer("mmseqs_threads", 8, "CPU threads for MMseqs2 operations.")

for database_name in ("uniref90", "mgnify", "small_bfd", "uniprot"):
    flags.DEFINE_string(
        f"mmseqs_{database_name}_database_path",
        None,
        f"Explicit GPU-compatible MMseqs2 {database_name} database prefix.",
    )
    flags.DEFINE_string(
        f"mmseqs_{database_name}_database_id",
        None,
        f"Immutable identifier for the {database_name} database build.",
    )

flags.DEFINE_integer(
    "mmseqs_uniref90_max_sequences", 10_000, "Maximum UniRef90 hits per query."
)
flags.DEFINE_integer(
    "mmseqs_mgnify_max_sequences", 5_000, "Maximum MGnify hits per query."
)
flags.DEFINE_integer(
    "mmseqs_small_bfd_max_sequences", 5_000, "Maximum small-BFD hits per query."
)
flags.DEFINE_integer(
    "mmseqs_uniprot_max_sequences", 50_000, "Maximum paired UniProt hits per query."
)

FLAGS = flags.FLAGS


def _database(name: str) -> DatabaseSpec:
    return DatabaseSpec(
        name=name,
        path=Path(getattr(FLAGS, f"mmseqs_{name}_database_path")),
        identifier=getattr(FLAGS, f"mmseqs_{name}_database_id"),
        max_sequences=getattr(FLAGS, f"mmseqs_{name}_max_sequences"),
    )


def main(argv) -> None:
    del argv
    summary_path = Path(FLAGS.summary_path)
    summary_path.unlink(missing_ok=True)
    requests = protein_requests_from_fastas(FLAGS.fasta_paths)
    settings = MsaBatchSettings(
        output_dir=Path(FLAGS.msa_output_dir),
        temp_dir=Path(FLAGS.mmseqs_temp_dir),
        unpaired_databases=tuple(
            _database(name) for name in ("uniref90", "mgnify", "small_bfd")
        ),
        paired_database=_database("uniprot"),
        max_sequences_per_batch=FLAGS.mmseqs_batch_max_sequences,
        max_residues_per_batch=FLAGS.mmseqs_batch_max_residues,
        threads=FLAGS.mmseqs_threads,
        e_value=FLAGS.mmseqs_e_value,
    )
    result = MsaBatch(
        settings=settings,
        mmseqs_process=SubprocessMmseqsProcess(FLAGS.mmseqs_binary_path),
    ).generate(requests)
    logging.info(
        "MMseqs2-GPU MSA stage: %d written, %d reused, %d failed",
        len(result.written),
        len(result.reused),
        len(result.failures),
    )
    if result.failures:
        detail = ", ".join(
            f"{failure.name} ({failure.error})" for failure in result.failures
        )
        raise RuntimeError(
            f"MMseqs2-GPU MSA stage failed for {len(result.failures)} protein(s): {detail}"
        )
    write_batch_summary(summary_path, result)


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "fasta_paths",
            "msa_output_dir",
            "summary_path",
            "mmseqs_binary_path",
            "mmseqs_temp_dir",
            "mmseqs_batch_max_sequences",
            "mmseqs_batch_max_residues",
            *[
                f"mmseqs_{database_name}_{suffix}"
                for database_name in ("uniref90", "mgnify", "small_bfd", "uniprot")
                for suffix in ("database_path", "database_id")
            ],
        ]
    )
    app.run(main)
