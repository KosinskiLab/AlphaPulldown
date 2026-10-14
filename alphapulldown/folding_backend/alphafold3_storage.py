"""
A finished AlphaFold 3 output directory: its ranking ledger (``ranking_scores.csv``),
the best sample it names, and the ``--storage_mode`` presets that slim the directory.

No AlphaFold 3 import, so tools without AF3 can read an AF3 output directory.
"""

import csv
import logging
import os

from alphapulldown.utils.post_modelling import compress_file

AF3_RANKING_SCORES = "ranking_scores.csv"


def post_prediction_process_af3(output_path, job_name, storage_mode="vanilla"):
    """Apply the AlphaFold3 storage preset to a single prediction directory.

    Layout written by the AF3 backend (vanilla):
        <output_path>/<job_name>_confidences.json      <- copy of best sample
        <output_path>/<job_name>_data.json             <- copy of features input
        <output_path>/<job_name>_model.cif             <- best model
        <output_path>/<job_name>_summary_confidences.json
        <output_path>/seed-*_sample-*/confidences.json <- per-sample (large)
        <output_path>/seed-*_sample-*/model.cif
        <output_path>/seed-*_sample-*/summary_confidences.json

    The best sample's ``confidences.json`` is always left UNCOMPRESSED so
    AlphaJudge (which reads the best model's PAE from it and has no xz support)
    keeps working.

    ``"slim"`` and ``"minimal"`` behave identically for AlphaFold3. Deleting a
    non-best sample's ``confidences.json`` would not lose disk-only data: it is
    the sole source of that sample's full token x token PAE matrix, and with it
    gone AlphaJudge silently falls back to ``summary_confidences.json`` (a much
    coarser per-chain-pair PAE), degrading PAE-derived scores for that sample
    without any error. So both presets compress rather than delete it; the AF2
    backend is where ``minimal`` additionally drops the (genuinely unused)
    result pickles.

    - ``"vanilla"``: no-op, keep byte-identical AlphaFold3 output.
    - ``"slim"`` / ``"minimal"``: delete the two top-level duplicates
      (``*_confidences.json`` is byte-identical to the best sample's,
      ``*_data.json`` duplicates the saved features input) and xz-compress the
      per-sample ``confidences.json`` of the NON-best samples. The best sample's
      stays plain; every structure and summary is retained, and no PAE is lost.
    """
    if storage_mode == "vanilla":
        return

    try:
        # Identify the best sample dir from ranking_scores.csv (highest score).
        best_sample_dir = af3_best_sample_dir(output_path)

        # Top-level duplicates: confidences.json mirrors the best sample,
        # data.json mirrors features/<name>_af3_input.json. Remove them only if
        # the best sample's own confidences.json is present (so the information
        # is not lost), since AlphaJudge can fall back to either location.
        best_conf_present = best_sample_dir is not None and os.path.isfile(
            os.path.join(output_path, best_sample_dir, "confidences.json")
        )
        top_dups = [f"{job_name}_data.json"]
        if best_conf_present:
            top_dups.append(f"{job_name}_confidences.json")
        for suffix in top_dups:
            dup = os.path.join(output_path, suffix)
            if os.path.isfile(dup):
                logging.info(f"storage_mode={storage_mode}: removing top-level duplicate {dup}")
                os.remove(dup)

        sample_dirs = sorted(
            d for d in os.listdir(output_path)
            if os.path.isdir(os.path.join(output_path, d))
            and d.startswith("seed-") and "_sample-" in d
        )
        for sample in sample_dirs:
            conf = os.path.join(output_path, sample, "confidences.json")
            if not os.path.isfile(conf):
                continue
            if sample == best_sample_dir:
                # Keep plain so AlphaJudge reads best-model PAE directly.
                continue
            # slim and minimal both compress (never delete) non-best confidences
            # so the full per-sample PAE matrix is never lost.
            compress_file(conf, method="xz")

    except FileNotFoundError as e:
        logging.error(f"AF3 post-processing error: {e}.")
    except Exception as e:
        logging.error(f"Unexpected AF3 post-processing error: {e}")


def af3_ranking_scores(output_path):
    """The (seed, sample, ranking score) rows of an AF3 fold's ranking_scores.csv, in
    order; empty if it is missing or unreadable."""
    try:
        with open(os.path.join(output_path, AF3_RANKING_SCORES), newline="") as f:
            return [
                (int(row["seed"]), int(row["sample"]), float(row["ranking_score"]))
                for row in csv.DictReader(f)
            ]
    except (OSError, KeyError, ValueError):
        return []


def af3_best_sample_dir(output_path):
    """The ``seed-*_sample-*`` dir with the highest ranking score; the first of equals."""
    rows = af3_ranking_scores(output_path)
    if not rows:
        return None
    seed, sample, _ = max(rows, key=lambda row: row[2])
    return f"seed-{seed}_sample-{sample}"
