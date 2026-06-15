import os
import json
import gzip
import lzma
import shutil
import logging
import pickle

# Keys whose arrays duplicate information already saved as standalone files in
# the AF2 output directory. ``predicted_aligned_error`` is byte-for-byte the same
# information as ``pae_<model>.json`` (which AlphaJudge and convert_to_modelcif
# read), so it is pure redundancy inside the pickle.
PAE_KEY = "predicted_aligned_error"
MAX_PAE_KEY = "max_predicted_aligned_error"


def compress_file(file_path, *, method="gzip"):
    """Compress a single file with gzip or xz, removing the original.

    ``method`` may be ``"gzip"`` (legacy default, ``.gz``) or ``"xz"`` (``.xz``,
    ~2-3x smaller on AF2 result pickles / PAE JSON at the cost of CPU time).
    """
    logging.info(f"Compressing file ({method}): {file_path}")
    if method == "xz":
        out_path = file_path + ".xz"
        opener = lambda p: lzma.open(p, "wb", preset=6)
    else:
        out_path = file_path + ".gz"
        opener = lambda p: gzip.open(p, "wb")
    try:
        with open(file_path, "rb") as f_in:
            with opener(out_path) as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)  # Remove the original file after compression
        logging.info(f"File compressed and original removed: {file_path}")
    except Exception as e:
        logging.error(f"Failed to compress file: {file_path} with error: {e}")
    return out_path


def compress_result_pickles(output_path, *, method="gzip"):
    """Compress all .pkl files in the output directory."""
    for file_name in os.listdir(output_path):
        if file_name.endswith('.pkl'):
            compress_file(os.path.join(output_path, file_name), method=method)


def remove_keys_from_pickle(file_path, keys_to_remove):
    """Remove specific keys from a .pkl file."""
    logging.info(f"Removing keys {keys_to_remove} from pickle file: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Remove the specified keys
        for key in keys_to_remove:
            if key in data:
                logging.info(f"Removing key: {key}")
                del data[key]

        # Save the modified data back to the pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=4)

        logging.info(f"Keys removed and file updated: {file_path}")
    except Exception as e:
        logging.error(f"Failed to remove keys from file: {file_path} with error: {e}")


def post_prediction_process(
    output_path,
    compress_pickles=False,
    remove_pickles=False,
    remove_keys=False,
    storage_mode="vanilla",
):
    """Process resulted files after the prediction.

    ``storage_mode`` is the high-level AlphaFold2 storage preset:

    - ``"vanilla"``: leave the result pickles untouched beyond the explicit
      ``compress_pickles`` / ``remove_pickles`` / ``remove_keys`` flags. Output
      stays a drop-in for tools expecting raw AlphaFold2 layout.
    - ``"slim"``: additionally strip the redundant ``predicted_aligned_error``
      array from every pickle (it is preserved in ``pae_<model>.json``, which is
      what AlphaJudge and convert_to_modelcif actually read) and xz-compress the
      remaining pickles.
    - ``"minimal"``: drop all result pickles entirely. Neither AlphaJudge nor
      convert_to_modelcif read pickle *contents*; scores come from the JSON
      sidecars and structures from the PDBs.
    """
    keys_to_remove = ['aligned_confidence_probs', 'distogram', 'masked_msa']
    # In slim mode the PAE array inside the pickle duplicates pae_<model>.json.
    if storage_mode == "slim":
        keys_to_remove = keys_to_remove + [PAE_KEY, MAX_PAE_KEY]

    try:
        # Get the best model from ranking_debug.json
        with open(os.path.join(output_path, "ranking_debug.json"), 'r') as f:
            best_model = json.load(f)['order'][0]

        # Best pickle file based on the known naming convention
        best_pickle = f"result_{best_model}.pkl"

        logging.info(f"Identified best pickle file: {best_pickle}")

        if storage_mode == "minimal":
            # Pickle contents are unused downstream; delete them all.
            logging.info("storage_mode=minimal: removing all result pickles.")
            for file_name in os.listdir(output_path):
                if file_name.endswith('.pkl'):
                    os.remove(os.path.join(output_path, file_name))
            return

        if remove_keys or storage_mode == "slim":
            logging.info(f"Removing specified keys from all pickle files in {output_path}")
            for file_name in os.listdir(output_path):
                if file_name.endswith('.pkl'):
                    remove_keys_from_pickle(os.path.join(output_path, file_name), keys_to_remove)

        # slim mode compresses with xz; explicit compress_pickles keeps gzip for
        # backwards compatibility.
        compress = compress_pickles or storage_mode == "slim"
        method = "xz" if storage_mode == "slim" else "gzip"

        if compress and remove_pickles:
            # Compress only the best .pkl file and remove the others
            logging.info("Compressing and removing pickles based on conditions.")
            compress_file(os.path.join(output_path, best_pickle), method=method)
            remove_irrelevant_pickles(output_path, best_pickle)
        else:
            if compress:
                logging.info("Compressing all pickle files.")
                compress_result_pickles(output_path, method=method)
            if remove_pickles:
                logging.info("Removing all non-best pickle files.")
                remove_irrelevant_pickles(output_path, best_pickle)

    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Please check your inputs.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


def remove_irrelevant_pickles(output_path, best_pickle):
    """Remove all .pkl files that do not belong to the best model."""
    for file_name in os.listdir(output_path):
        file_path = os.path.join(output_path, file_name)
        if file_name.endswith('.pkl') and file_name != best_pickle:
            logging.info(f"Removing irrelevant pickle file: {file_path}")
            os.remove(file_path)


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

    - ``"vanilla"``: no-op, keep byte-identical AlphaFold3 output.
    - ``"slim"``: delete the two top-level duplicates (``*_confidences.json`` is
      byte-identical to the best sample's, ``*_data.json`` duplicates the saved
      features input) and xz-compress the per-sample ``confidences.json`` of the
      NON-best samples. Best sample's stays plain; structures and summaries for
      all samples are retained.
    - ``"minimal"``: slim, but delete (rather than compress) the non-best
      ``confidences.json`` entirely. Best sample keeps its plain
      ``confidences.json``; all structures and summary scores are retained.
    """
    if storage_mode == "vanilla":
        return

    try:
        # Identify the best sample dir from ranking_scores.csv (highest score).
        best_sample_dir = _af3_best_sample_dir(output_path)

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
            if storage_mode == "minimal":
                logging.info(f"storage_mode=minimal: removing {conf}")
                os.remove(conf)
            else:  # slim
                compress_file(conf, method="xz")

    except FileNotFoundError as e:
        logging.error(f"AF3 post-processing error: {e}.")
    except Exception as e:
        logging.error(f"Unexpected AF3 post-processing error: {e}")


def _af3_best_sample_dir(output_path):
    """Return the ``seed-*_sample-*`` dir name with the highest ranking score."""
    import csv
    ranking_csv = os.path.join(output_path, "ranking_scores.csv")
    if not os.path.isfile(ranking_csv):
        return None
    best = (None, float("-inf"))
    with open(ranking_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                score = float(row["ranking_score"])
            except (KeyError, ValueError):
                continue
            if score > best[1]:
                best = (f"seed-{row['seed']}_sample-{row['sample']}", score)
    return best[0]
