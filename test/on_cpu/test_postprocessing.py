from absl.testing import absltest
import os
import tempfile
import shutil
import pickle
import gzip
import numpy as np
import json
from alphafold.common import protein


class _TestBase(absltest.TestCase):
    def setUp(self) -> None:
        self.data_dir = "/scratch/AlphaFold_DBs/2.3.2/"
        #Get test_data directory as relative path to this script
        self.test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

class TestScript(_TestBase):
    #Add setup that creates empty output directory temporary
    def setUp(self) -> None:
        #Call the setUp method of the parent class
        super().setUp()

        #Create a temporary directory for the output
        self.output_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        self.protein_lists = os.path.join(self.test_data_dir, "tiny_monomeric_features_homodimer.txt")
        self.monomer_objects_dir = self.test_data_dir


    def tearDown(self) -> None:
        #Remove the temporary directory
        shutil.rmtree(self.output_dir)


class TestFunctions(_TestBase):
    def setUp(self):
        # Call the setUp method of the parent class
        super().setUp()

        from alphapulldown.utils.modelling_setup import create_model_runners_and_random_seed
        self.model_runners, random_seed = create_model_runners_and_random_seed(
            "multimer",
            3,
            1,
            self.data_dir,
            1,
        )

    def get_score_from_result_pkl(pkl_path):
        """Get the score from the model result pkl file.

        Parameters:
            pkl_path (str): The file path to the pickle file.

        Returns:
            tuple: A tuple containing the score type (str) and the score (float).
        """
        try:
            with open(pkl_path, "rb") as f:
                result = pickle.load(f)
        except (EOFError, FileNotFoundError) as e:
            try:
                with gzip.open(pkl_path, "rb") as f:
                    result = pickle.load(f)
            except Exception as e:
                raise IOError("Failed to load the pickle file.") from e

        if "iptm" in result and "ptm" in result:
            score_type = "iptm+ptm"
            score = 0.8 * result["iptm"] + 0.2 * result["ptm"]
        elif "plddt" in result:
            score_type = "plddt"
            score = np.mean(result["plddt"])
        else:
            raise ValueError("Result does not contain expected keys.")

        return score_type, score

    def get_existing_model_info(self, output_dir, model_runners):
        ranking_confidences = {}
        unrelaxed_proteins = {}
        unrelaxed_pdbs = {}
        processed_models = 0

        for model_name, _ in model_runners.items():
            pdb_path = os.path.join(output_dir, f"unrelaxed_{model_name}.pdb")
            pkl_path = os.path.join(output_dir, f"result_{model_name}.pkl")
            pkl_gz_path = os.path.join(output_dir, f"result_{model_name}.pkl.gz")

            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, "rb") as f:
                        pickle.load(f)
                except (EOFError, pickle.UnpicklingError):
                    break
                score_name, score = self.get_score_from_result_pkl(pkl_path)
                ranking_confidences[model_name] = score
            if os.path.exists(pkl_gz_path):
                try:
                    with gzip.open(pkl_gz_path, "rb") as f:
                        pickle.load(f)
                except (EOFError, pickle.UnpicklingError):
                    break
                score_name, score = self.get_score_from_result_pkl_gz(pkl_gz_path)
                ranking_confidences[model_name] = score
            if os.path.exists(pdb_path):
                with open(pdb_path, "r") as f:
                    unrelaxed_pdb_str = f.read()
                unrelaxed_proteins[model_name] = protein.from_pdb_string(unrelaxed_pdb_str)
                unrelaxed_pdbs[model_name] = unrelaxed_pdb_str
                processed_models += 1

        return ranking_confidences, unrelaxed_proteins, unrelaxed_pdbs, processed_models

    def test_get_1(self):
        """Oligomer: Check that iptm+ptm are equal in json and result pkl"""
        self.output_dir = os.path.join(self.test_data_dir, "P0DPR3_and_P0DPR3")
        # Open ranking_debug.json from self.output_dir and load to results
        with open(os.path.join(self.output_dir, "ranking_debug.json"), "r") as f:
            results = json.load(f)
            # Get the expected score from the results
            expected_iptm_ptm = results["iptm+ptm"]["model_1_multimer_v3_pred_0"]

        pkl_path = os.path.join(self.test_data_dir, "P0DPR3_and_P0DPR3", "result_model_1_multimer_v3_pred_0.pkl")
        out = self.get_score_from_result_pkl(pkl_path)
        self.assertTupleEqual(out, ('iptm+ptm', expected_iptm_ptm))

    def test_get_2(self):
        """Oligomer: Check get_existing_model_info for all models finished"""
        self.output_dir = os.path.join(self.test_data_dir, "P0DPR3_and_P0DPR3")
        ranking_confidences, unrelaxed_proteins, unrelaxed_pdbs, START = self.get_existing_model_info(self.output_dir,
                                                                                                      self.model_runners)
        self.assertEqual(len(ranking_confidences), len(unrelaxed_proteins))
        self.assertEqual(len(ranking_confidences), len(unrelaxed_pdbs))
        self.assertEqual(len(ranking_confidences), len(self.model_runners))
        self.assertEqual(START, 5)
        with open(os.path.join(self.output_dir, "ranking_debug.json"), "r") as f:
            results = json.load(f)
            # Get the expected score from the results
            expected_iptm_ptm = results["iptm+ptm"]
        self.assertDictEqual(ranking_confidences, expected_iptm_ptm)

    def test_get_3(self):
        """Oligomer: Check get_existing_model_info, resume after 2 models finished"""
        self.output_dir = os.path.join(self.test_data_dir, "P0DPR3_and_P0DPR3_partial")
        ranking_confidences, unrelaxed_proteins, unrelaxed_pdbs, START = self.get_existing_model_info(self.output_dir,
                                                                                                      self.model_runners)
        self.assertEqual(len(ranking_confidences), len(unrelaxed_proteins))
        self.assertEqual(len(ranking_confidences), len(unrelaxed_pdbs))
        self.assertNotEqual(len(ranking_confidences), len(self.model_runners))
        self.assertEqual(START, 2)
        with open(os.path.join(self.output_dir, "ranking_debug_temp.json"), "r") as f:
            results = json.load(f)
            # Get the expected score from the results
            expected_iptm_ptm = results["iptm+ptm"]
        self.assertDictEqual(ranking_confidences, expected_iptm_ptm)

    # TODO: Test monomeric runs (where score is pLDDT)
    def test_get_4(self):
        """Monomer: Check that plddt are equal in json and result pkl"""
        pass

    def test_get_5(self):
        """Monomer: Check get_existing_model_info for all models finished"""
        pass

    def test_get_6(self):
        """Monomer: Check get_existing_model_info, resume after 2 models finished"""
        pass
