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
        parent_dir = join(dirname(dirname(abspath(__file__))))
        self.input_dir = join(parent_dir, "test/test_data/predictions")
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
            logging.info(f"Running test for prediction_dir='{prediction_dir}', "
                         f"compress_pickles={compress_pickles}, remove_pickles={remove_pickles}, remove_keys={remove_keys}")
            temp_dir_path = temp_dir.name
            shutil.copytree(join(self.input_dir, prediction_dir), join(temp_dir_path, prediction_dir))

            # Remove existing gz files
            gz_files_existing = [f for f in os.listdir(join(temp_dir_path, prediction_dir)) if f.endswith('.gz')]
            for f_ in gz_files_existing:
                os.remove(join(temp_dir_path, prediction_dir, f_))

            # Run the postprocessing
            post_prediction_process(join(temp_dir_path, prediction_dir),
                                    compress_pickles,
                                    remove_pickles,
                                    remove_keys)

            # Identify the best model
            with open(join(temp_dir_path, prediction_dir, 'ranking_debug.json')) as f:
                best_model = json.load(f)['order'][0]
            best_result_pickle = join(temp_dir_path, prediction_dir, f"result_{best_model}.pkl")

            # Gather .pkl and .gz files
            pickle_files = [f for f in os.listdir(join(temp_dir_path, prediction_dir)) if f.endswith('.pkl')]
            gz_files = [f for f in os.listdir(join(temp_dir_path, prediction_dir)) if f.endswith('.gz')]

            # Check if specified keys exist or were removed
            if remove_keys:
                for pf in pickle_files:
                    with open(join(temp_dir_path, prediction_dir, pf), 'rb') as f:
                        data = pickle.load(f)
                    for key in ['aligned_confidence_probs', 'distogram', 'masked_msa']:
                        self.assertNotIn(key, data, f"Key '{key}' was not removed from {pf}")
            else:
                # If we're not removing keys, verify they still exist in the pickle
                for pf in pickle_files:
                    with open(join(temp_dir_path, prediction_dir, pf), 'rb') as f:
                        data = pickle.load(f)
                    for key in ['aligned_confidence_probs', 'distogram', 'masked_msa']:
                        self.assertIn(key, data, f"Key '{key}' was unexpectedly removed from {pf}")

            # Now check file counts / compressions
            if not compress_pickles and not remove_pickles:
                # Expect all .pkl files (5 in your scenario), no .gz
                self.assertEqual(len(pickle_files), 5,
                                 f"Expected 5 pickle files, found {len(pickle_files)}.")
                self.assertEqual(len(gz_files), 0,
                                 f"Expected 0 gz files, found {len(gz_files)}.")

            if compress_pickles and not remove_pickles:
                # Expect 0 .pkl files, all compressed (5)
                self.assertEqual(len(pickle_files), 0,
                                 f"Expected 0 pickle files, found {len(pickle_files)}.")
                self.assertEqual(len(gz_files), 5,
                                 f"Expected 5 gz files, found {len(gz_files)}.")
                # Validate that gz files are readable
                for gz_file in gz_files:
                    with gzip.open(join(temp_dir_path, prediction_dir, gz_file), 'rb') as f:
                        f.read(1)

            if not compress_pickles and remove_pickles:
                # Only the best pickle remains
                self.assertEqual(len(pickle_files), 1,
                                 f"Expected 1 pickle file, found {len(pickle_files)}.")
                self.assertEqual(len(gz_files), 0,
                                 f"Expected 0 gz files, found {len(gz_files)}.")
                self.assertTrue(os.path.exists(best_result_pickle),
                                f"Best result pickle file does not exist: {best_result_pickle}")

            if compress_pickles and remove_pickles:
                # Only the best pickle is compressed
                self.assertEqual(len(pickle_files), 0,
                                 f"Expected 0 pickle files, found {len(pickle_files)}.")
                self.assertEqual(len(gz_files), 1,
                                 f"Expected 1 gz file, found {len(gz_files)}.")
                self.assertTrue(os.path.exists(best_result_pickle + ".gz"),
                                f"Best result pickle file not compressed: {best_result_pickle}.gz")
                with gzip.open(join(temp_dir_path, prediction_dir, gz_files[0]), 'rb') as f:
                    f.read(1)  # Check it's valid gzip

        except AssertionError as e:
            logging.error(f"AssertionError: {e}")
            all_files = os.listdir(join(temp_dir_path, prediction_dir))
            relevant_files = [f for f in all_files if f.endswith('.gz') or f.endswith('.pkl')]
            logging.error(f".gz and .pkl files in {join(temp_dir_path, prediction_dir)}: {relevant_files}")
            raise
        finally:
            temp_dir.cleanup()
