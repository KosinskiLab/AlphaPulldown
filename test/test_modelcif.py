import os
import logging
import subprocess
from absl.testing import parameterized
import shutil
import tempfile
from os.path import join, dirname, abspath
import zipfile

"""
Test conversion of PDB to CIF for monomers and multimers
"""


class TestConvertPDB2CIF(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Get path of the alphapulldown module
        parent_dir = join(dirname(dirname(abspath(__file__))))
        # Join the path with the script name
        self.input_dir = join(parent_dir, "test/test_data/predictions")
        self.script_path = join(parent_dir, "alphapulldown/scripts/convert_to_modelcif.py")
        # Set logging level to INFO
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    @parameterized.named_parameters(
        {'testcase_name': 'monomer_add_no_compress_model', 'input_dir': "TEST", 'add_associated': True,
         'compress': False, 'model_selected': 0},
        {'testcase_name': 'monomer_add_compress_model', 'input_dir': "TEST", 'add_associated': True, 'compress': True,
         'model_selected': 0},
        {'testcase_name': 'monomer_no_add_no_compress_model', 'input_dir': "TEST", 'add_associated': False,
         'compress': False, 'model_selected': 0},
        {'testcase_name': 'monomer_no_add_compress_model', 'input_dir': "TEST", 'add_associated': False,
         'compress': True, 'model_selected': 0},
        {'testcase_name': 'monomer_no_add_no_compress_no_model', 'input_dir': "TEST", 'add_associated': False,
         'compress': False, 'model_selected': None},
        {'testcase_name': 'monomer_no_add_compress_no_model', 'input_dir': "TEST", 'add_associated': False,
         'compress': True, 'model_selected': None},
        {'testcase_name': 'dimer_add_no_compress_model', 'input_dir': "TEST_and_TEST", 'add_associated': True,
         'compress': False, 'model_selected': 0},
        {'testcase_name': 'dimer_add_compress_model', 'input_dir': "TEST_and_TEST", 'add_associated': True,
         'compress': True, 'model_selected': 0},
        {'testcase_name': 'dimer_no_add_no_compress_model', 'input_dir': "TEST_and_TEST", 'add_associated': False,
         'compress': False, 'model_selected': 0},
        {'testcase_name': 'dimer_no_add_compress_model', 'input_dir': "TEST_and_TEST", 'add_associated': False,
         'compress': True, 'model_selected': 0},
        {'testcase_name': 'dimer_no_add_no_compress_no_model', 'input_dir': "TEST_and_TEST", 'add_associated': False,
         'compress': False, 'model_selected': None},
        {'testcase_name': 'dimer_no_add_compress_no_model', 'input_dir': "TEST_and_TEST", 'add_associated': False,
         'compress': True, 'model_selected': None}
    )
    def test_(self, input_dir, add_associated, compress, model_selected):
        """Test conversion of PDB to CIF for monomers and multimers"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_output_dir = join(temp_dir, 'output')
            shutil.copytree(join(self.input_dir, input_dir), test_output_dir)
            logging.info(f"Converting {test_output_dir} to ModelCIF format...")
            command = self.build_command(test_output_dir, add_associated, compress, model_selected)
            logging.info(" ".join(command))
            try:
                result = subprocess.run(command,
                                        check=True,
                                        capture_output=True,
                                        text=True)
                if result.stderr:
                    logging.error(f"Conversion errors: {result.stderr}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Subprocess failed with error: {e.stderr}")
                raise
            logging.info(os.listdir(test_output_dir))

            expected_ids = [model_selected] if model_selected is not None else [i for i in range(5)]
            for idx in expected_ids:
                rnk = f"ranked_{idx}"
                zip_dir = f"{rnk}.zip"
                logging.info(f"Checking existence of file: {zip_dir}")
                self.assertTrue(os.path.exists(join(test_output_dir, zip_dir)),
                                f"File {zip_dir} does not exist")
                cif = f"ranked_{idx}.cif"
                if compress:
                    cif = f"{cif}.gz"
                logging.info(f"Checking existence of file: {cif}")
                self.assertTrue(cif, f"File {cif} exists")

            if add_associated:
                associated_file = f"ranked_{idx}.zip"
                logging.info(f"Unzipping: {associated_file}")
                with zipfile.ZipFile(join(test_output_dir, associated_file), 'r') as zip_ref:
                    ass_dir = join(test_output_dir, "associated", f"ranked_{idx}")
                    zip_ref.extractall(ass_dir)
                    logging.info(os.listdir(ass_dir))
                    local_pairwise_file = f"ranked_{idx}_local_pairwise_qa.cif"
                    logging.info(f"Checking existence of extracted file: {local_pairwise_file}")
                    self.assertTrue(os.path.exists(join(ass_dir, local_pairwise_file)),
                                    f"File {local_pairwise_file} does not exist")
                    expected_ids = list(set([0, 1, 2, 3, 4]) - set(expected_ids))
                    for idx in expected_ids:
                        ass_mdl_cif = join(ass_dir, f"ranked_{idx}.cif")
                        if compress:
                            ass_mdl_cif = f"{ass_mdl_cif}.gz"
                        logging.info(f"Checking existense of {ass_mdl_cif} in {ass_dir}")
                        self.assertTrue(os.path.exists(ass_mdl_cif))
                        ass_mdl_zip = join(ass_dir, f"ranked_{idx}.zip")
                        logging.info(f"Checking existense of {ass_mdl_zip} in {ass_dir}")
                        self.assertTrue(os.path.exists(ass_mdl_zip))



    def build_command(self, output_dir, add_associated, compress, model_selected):
        """Build the command for subprocess"""
        command = [
            "python3", self.script_path,
            "--ap_output", output_dir,
            "--add_associated" if add_associated else "--noadd_associated",
            "--compress" if compress else "--nocompress"
        ]

        if model_selected is not None:
            command.extend(["--model_selected", str(model_selected)])

        return command
