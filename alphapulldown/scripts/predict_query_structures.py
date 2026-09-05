#!/usr/bin/env python3
"""GPU stage of the structural template path: fold query sequences and cache them.

Structure prediction is the only part of Foldseek template search that wants a
GPU, and it depends on nothing but the sequence and the ESMFold weights. Running
it as its own step lets a scheduler put it on a GPU node and leave the Foldseek
search and the rest of feature generation on CPUs.

Running it is optional. ``create_individual_features.py --use_foldseek_templates``
folds whatever it does not find in the cache, so this stage only ever moves work
earlier -- it never becomes a prerequisite.
"""

from __future__ import annotations

from absl import app, flags, logging

from alphapulldown.scripts._foldseek_cli import (
    build_structure_cache,
    define_structural_template_flags,
)
from alphapulldown.utils.file_handling import iter_seqs


define_structural_template_flags(include_switch=False)
flags.DEFINE_list("fasta_paths", None, "Paths to protein FASTA files.")

FLAGS = flags.FLAGS


def main(argv) -> None:
    del argv
    cache = build_structure_cache(FLAGS)
    written: list[str] = []
    reused: list[str] = []
    failures: list[tuple[str, str]] = []
    for sequence, description in iter_seqs(FLAGS.fasta_paths):
        already_cached = cache.cached(sequence)
        try:
            cache.structure(sequence)
        except Exception as exc:
            # One unfoldable sequence should not cost the rest of the shard its
            # structures; the run still exits nonzero below.
            failures.append((description, str(exc)))
            continue
        (reused if already_cached else written).append(description)
    logging.info(
        "Structure prediction stage: %d folded, %d already cached, %d failed",
        len(written),
        len(reused),
        len(failures),
    )
    if failures:
        detail = ", ".join(f"{name} ({error})" for name, error in failures)
        raise RuntimeError(
            f"Structure prediction failed for {len(failures)} sequence(s): {detail}"
        )


if __name__ == "__main__":
    flags.mark_flags_as_required(
        ["fasta_paths", "structural_template_cache_dir", "esmfold_model_dir"]
    )
    app.run(main)
