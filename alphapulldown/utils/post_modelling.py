import os
import json
import gzip
import shutil
import logging
import pickle


def compress_file(file_path):
    """Compress a single file with gzip."""
    logging.info(f"Compressing file: {file_path}")
    gz_path = file_path + '.gz'
    try:
        with open(file_path, 'rb') as f_in:
            with gzip.open(gz_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)  # Remove the original file after compression
        logging.info(f"File compressed and original removed: {file_path}")
    except Exception as e:
        logging.error(f"Failed to compress file: {file_path} with error: {e}")
    return gz_path


def compress_result_pickles(output_path):
    """Compress all .pkl files in the output directory."""
    for file_name in os.listdir(output_path):
        if file_name.endswith('.pkl'):
            compress_file(os.path.join(output_path, file_name))


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
            pickle.dump(data, f)

        logging.info(f"Keys removed and file updated: {file_path}")
    except Exception as e:
        logging.error(f"Failed to remove keys from file: {file_path} with error: {e}")


def post_prediction_process(output_path, compress_pickles=False, remove_pickles=False, remove_keys=False):
    """Process resulted files after the prediction."""
    keys_to_remove = ['aligned_confidence_probs', 'distogram', 'masked_msa']

    try:
        # Get the best model from ranking_debug.json
        with open(os.path.join(output_path, "ranking_debug.json"), 'r') as f:
            best_model = json.load(f)['order'][0]

        # Best pickle file based on the known naming convention
        best_pickle = f"result_{best_model}.pkl"

        logging.info(f"Identified best pickle file: {best_pickle}")

        if remove_keys:
            logging.info(f"Removing specified keys from all pickle files in {output_path}")
            for file_name in os.listdir(output_path):
                if file_name.endswith('.pkl'):
                    remove_keys_from_pickle(os.path.join(output_path, file_name), keys_to_remove)

        if compress_pickles and remove_pickles:
            # Compress only the best .pkl file and remove the others
            logging.info("Compressing and removing pickles based on conditions.")
            compress_file(os.path.join(output_path, best_pickle))
            remove_irrelevant_pickles(output_path, best_pickle)
        else:
            if compress_pickles:
                logging.info("Compressing all pickle files.")
                compress_result_pickles(output_path)
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
