import os
import logging
from absl.testing import parameterized
import shutil
import tempfile
from os.path import join, dirname, abspath
import zipfile
from alphapulldown.utils.post_modelling import post_prediction_process
import json

"""
Test removing result pickles and archiving the results
"""


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
        ('TEST', False, False),
        ('TEST', True, False),
        ('TEST', True, True),
        ('TEST', False, True),
        ('TEST_and_TEST', False, False),
        ('TEST_and_TEST', True, False),
        ('TEST_and_TEST', True, True),
        ('TEST_and_TEST', False, True)
    )
    def test_files(self, prediction_dir, zip_pickles, remove_pickles):
        """Test postprocessing of the prediction results"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy the files to the temporary directory
            shutil.copytree(join(self.input_dir, prediction_dir), join(temp_dir, prediction_dir))
            # remove existing zip files
            zip_files = [f for f in os.listdir(join(temp_dir, prediction_dir)) if f.endswith('.zip')]
            for f in zip_files:
                os.remove(join(temp_dir, prediction_dir, f))
            # Run the postprocessing function
            post_prediction_process(join(temp_dir, prediction_dir), zip_pickles, remove_pickles)

            # Get the best model from ranking_debug.json
            with open(join(temp_dir, prediction_dir, 'ranking_debug.json')) as f:
                best_model = json.load(f)['order'][0]

            # Define the expected best result pickle path
            if prediction_dir == 'TEST':
                best_result_pickle = join(temp_dir,
                                          prediction_dir,
                                          f"result_model_{best_model}_ptm_pred_0.pkl")
            elif prediction_dir == 'TEST_and_TEST':
                best_result_pickle = join(temp_dir,
                                          prediction_dir,
                                          f"result_model_{best_model}_multimer_v3_pred.pkl")

            # Check if files are removed and/or zipped based on the parameters
            pickle_files = [f for f in os.listdir(join(temp_dir, prediction_dir)) if f.endswith('.pkl')]
            zip_files = [f for f in os.listdir(join(temp_dir, prediction_dir)) if f.endswith('.zip')]

            if not zip_pickles and not remove_pickles:
                # All pickle files should be present, no zip files
                self.assertEqual(len(pickle_files), 5)
                self.assertEqual(len(zip_files), 0)

            if zip_pickles and not remove_pickles:
                # No pickle files should be present, each zipped separately
                self.assertEqual(len(pickle_files), 0)
                self.assertEqual(len(zip_files), 5)
                for zip_file in zip_files:
                    with zipfile.ZipFile(join(temp_dir, prediction_dir, zip_file), 'r') as z:
                        self.assertTrue(all(f.endswith('.pkl') for f in z.namelist()))

            if not zip_pickles and remove_pickles:
                # Only the best result pickle should be present
                self.assertEqual(len(pickle_files), 1)
                self.assertEqual(len(zip_files), 0)
                self.assertTrue(os.path.exists(best_result_pickle))

            if zip_pickles and remove_pickles:
                # Only the best result pickle should be zipped, no pickle files present
                self.assertEqual(len(pickle_files), 0)
                self.assertEqual(len(zip_files), 1)
                self.assertTrue(os.path.exists(best_result_pickle + ".zip"))
                with zipfile.ZipFile(join(temp_dir, prediction_dir, zip_files[0]), 'r') as z:
                    self.assertTrue(all(f.endswith('.pkl') for f in z.namelist()))
