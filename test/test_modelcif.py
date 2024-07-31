from absl import logging
import subprocess
from absl.testing import parameterized
import shutil
import tempfile
from os.path import join, dirname, abspath
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

    @parameterized.named_parameters(
        {'testcase_name': 'monomer_add_no_compress_model', 'input_dir': "TEST", 'add_associated': True,
         'compress': False, 'model_selected': 0},
        {'testcase_name': 'monomer_add_compress_model', 'input_dir': "TEST", 'add_associated': True, 'compress': True,
         'model_selected': 0},
        {'testcase_name': 'monomer_no_add_no_compress_model', 'input_dir': "TEST", 'add_associated': False,
         'compress': False, 'model_selected': 0},
        {'testcase_name': 'monomer_no_add_compress_model', 'input_dir': "TEST", 'add_associated': False,
         'compress': True, 'model_selected': 0},
        {'testcase_name': 'monomer_add_no_compress_no_model', 'input_dir': "TEST", 'add_associated': True,
         'compress': False, 'model_selected': None},
        {'testcase_name': 'monomer_add_compress_no_model', 'input_dir': "TEST", 'add_associated': True,
         'compress': True, 'model_selected': None},
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
        {'testcase_name': 'dimer_add_no_compress_no_model', 'input_dir': "TEST_and_TEST", 'add_associated': True,
         'compress': False, 'model_selected': None},
        {'testcase_name': 'dimer_add_compress_no_model', 'input_dir': "TEST_and_TEST", 'add_associated': True,
         'compress': True, 'model_selected': None},
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

            try:
                result = subprocess.run(command,
                                        check=True,
                                        capture_output=True,
                                        text=True)
                logging.info(f"Conversion output: {result.stdout}")
                logging.error(f"Conversion errors: {result.stderr}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Subprocess failed with error: {e.stderr}")
                raise

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
