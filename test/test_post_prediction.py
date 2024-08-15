import os
import logging
from absl.testing import parameterized
import shutil
import tempfile
from os.path import join, dirname, abspath
import gzip
import json
import pickle
from alphapulldown.utils.post_modelling import post_prediction_process

class TestPostPrediction(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Get path of the alphapulldown module
        parent_dir = join(dirname(dirname(abspath(__file__))))
        # Join the path with the script name
        self.input_dir = join(parent_dir, "test/test_data/predictions")
        # Set logging level to INFO
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    @parameterized.parameters(
        ('TEST', False, False, False),
        ('TEST', True, False, False),
        ('TEST', True, True, False),
        ('TEST', False, True, False),
        ('TEST_and_TEST', False, False, False),
        ('TEST_and_TEST', True, False, False),
        ('TEST_and_TEST', True, True, False),
        ('TEST_and_TEST', False, True, False),
        ('TEST', False, False, True),
        ('TEST', True, False, True),
        ('TEST', True, True, True),
        ('TEST', False, True, True),
        ('TEST_and_TEST', False, False, True),
        ('TEST_and_TEST', True, False, True),
        ('TEST_and_TEST', True, True, True),
        ('TEST_and_TEST', False, True, True)
    )
    def test_files(self, prediction_dir, compress_pickles, remove_pickles, remove_keys):
        temp_dir = tempfile.TemporaryDirectory()
        try:
            logging.info(f"Running test for prediction_dir='{prediction_dir}', compress_pickles={compress_pickles}, remove_pickles={remove_pickles}, remove_keys={remove_keys}")
            temp_dir_path = temp_dir.name
            # Copy the files to the temporary directory
            shutil.copytree(join(self.input_dir, prediction_dir), join(temp_dir_path, prediction_dir))
            # Remove existing gz files
            gz_files = [f for f in os.listdir(join(temp_dir_path, prediction_dir)) if f.endswith('.gz')]
            for f in gz_files:
                os.remove(join(temp_dir_path, prediction_dir, f))
            # Run the postprocessing function
            post_prediction_process(join(temp_dir_path, prediction_dir), compress_pickles, remove_pickles, remove_keys)

            # Get the best model from ranking_debug.json
            with open(join(temp_dir_path, prediction_dir, 'ranking_debug.json')) as f:
                best_model = json.load(f)['order'][0]

            # Define the expected best result pickle path
            best_result_pickle = join(temp_dir_path, prediction_dir, f"result_{best_model}.pkl")

            # Check if files are removed and/or compressed based on the parameters
            pickle_files = [f for f in os.listdir(join(temp_dir_path, prediction_dir)) if f.endswith('.pkl')]
            gz_files = [f for f in os.listdir(join(temp_dir_path, prediction_dir)) if f.endswith('.gz')]

            if remove_keys:
                # Ensure specified keys are removed from the pickle files
                for pickle_file in pickle_files:
                    with open(join(temp_dir_path, prediction_dir, pickle_file), 'rb') as f:
                        data = pickle.load(f)
                    for key in ['aligned_confidence_probs', 'distogram', 'masked_msa']:
                        self.assertNotIn(key, data, f"Key {key} was not removed from {pickle_file}")

            if not compress_pickles and not remove_pickles:
                # All pickle files should be present, no gz files
                logging.info("Checking condition: not compress_pickles and not remove_pickles")
                self.assertEqual(len(pickle_files), 5, f"Expected 5 pickle files, found {len(pickle_files)}.")
                self.assertEqual(len(gz_files), 0, f"Expected 0 gz files, found {len(gz_files)}.")

            if compress_pickles and not remove_pickles:
                # No pickle files should be present, each compressed separately
                logging.info("Checking condition: compress_pickles and not remove_pickles")
                self.assertEqual(len(pickle_files), 0, f"Expected 0 pickle files, found {len(pickle_files)}.")
                self.assertEqual(len(gz_files), 5, f"Expected 5 gz files, found {len(gz_files)}.")
                for gz_file in gz_files:
                    with gzip.open(join(temp_dir_path, prediction_dir, gz_file), 'rb') as f:
                        f.read(1)  # Ensure it's a valid gzip file

            if not compress_pickles and remove_pickles:
                # Only the best result pickle should be present
                logging.info("Checking condition: not compress_pickles and remove_pickles")
                self.assertEqual(len(pickle_files), 1, f"Expected 1 pickle file, found {len(pickle_files)}.")
                self.assertEqual(len(gz_files), 0, f"Expected 0 gz files, found {len(gz_files)}.")
                self.assertTrue(os.path.exists(best_result_pickle), f"Best result pickle file does not exist: {best_result_pickle}")

            if compress_pickles and remove_pickles:
                # Only the best result pickle should be compressed, no pickle files present
                logging.info("Checking condition: compress_pickles and remove_pickles")
                self.assertEqual(len(pickle_files), 0, f"Expected 0 pickle files, found {len(pickle_files)}.")
                self.assertEqual(len(gz_files), 1, f"Expected 1 gz file, found {len(gz_files)}.")
                self.assertTrue(os.path.exists(best_result_pickle + ".gz"), f"Best result pickle file not compressed: {best_result_pickle}.gz")
                with gzip.open(join(temp_dir_path, prediction_dir, gz_files[0]), 'rb') as f:
                    f.read(1)  # Ensure it's a valid gzip file
        except AssertionError as e:
            logging.error(f"AssertionError: {e}")
            all_files = os.listdir(join(temp_dir_path, prediction_dir))
            relevant_files = [f for f in all_files if f.endswith('.gz') or f.endswith('.pkl')]
            logging.error(f".gz and .pkl files in {join(temp_dir_path, prediction_dir)}: {relevant_files}")
            raise  # Re-raise the exception to ensure the test is marked as failed
        finally:
            temp_dir.cleanup()
