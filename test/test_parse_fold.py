import logging
from absl.testing import parameterized
from unittest import mock
from alphapulldown.utils.modelling_setup import parse_fold

"""
Test parse_fold function with different scenarios
"""

class TestParseFold(parameterized.TestCase):

    def setUp(self) -> None:
        super().setUp()
        # Set logging level to INFO
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    @parameterized.named_parameters(
        {
            'testcase_name': 'single_protein_no_copy',
            'input': ['protein1'],
            'features_directory': ['dir1'],
            'protein_delimiter': '_',
            'mock_side_effect': {
                'dir1/protein1.pkl': True,
                'dir1/protein1.pkl.xz': False,
            },
            'expected_result': [[{'protein1': 'all'}]],
        },
        {
            'testcase_name': 'single_protein_with_copy_number',
            'input': ['protein1:2'],
            'features_directory': ['dir1'],
            'protein_delimiter': '_',
            'mock_side_effect': {
                'dir1/protein1.pkl': True,
                'dir1/protein1.pkl.xz': False,
            },
            'expected_result': [[{'protein1': 'all'}, {'protein1': 'all'}]],
        },
        {
            'testcase_name': 'single_protein_with_region',
            'input': ['protein1:1-10'],
            'features_directory': ['dir1'],
            'protein_delimiter': '_',
            'mock_side_effect': {
                'dir1/protein1.pkl': True,
                'dir1/protein1.pkl.xz': False,
            },
            'expected_result': [[{'protein1': [(1, 10)]}]],
        },
        {
            'testcase_name': 'single_protein_with_copy_and_regions',
            'input': ['protein1:2:1-10:20-30'],
            'features_directory': ['dir1'],
            'protein_delimiter': '_',
            'mock_side_effect': {
                'dir1/protein1.pkl': True,
                'dir1/protein1.pkl.xz': False,
            },
            'expected_result': [[{'protein1': [(1, 10), (20, 30)]}, {'protein1': [(1, 10), (20, 30)]}]],
        },
        {
            'testcase_name': 'single_protein_with_region_and_copy',
            'input': ['protein1:1-10:20-30:2'],
            'features_directory': ['dir1'],
            'protein_delimiter': '_',
            'mock_side_effect': {
                'dir1/protein1.pkl': True,
                'dir1/protein1.pkl.xz': False,
            },
            'expected_result': [[{'protein1': [(1, 10), (20, 30)]}, {'protein1': [(1, 10), (20, 30)]}]],
        },
        {
            'testcase_name': 'multiple_proteins',
            'input': ['protein1:2_protein2:1-50'],
            'features_directory': ['dir1'],
            'protein_delimiter': '_',
            'mock_side_effect': {
                'dir1/protein1.pkl': True,
                'dir1/protein1.pkl.xz': False,
                'dir1/protein2.pkl': True,
                'dir1/protein2.pkl.xz': False,
            },
            'expected_result': [[{'protein1': 'all'}, {'protein1': 'all'}, {'protein2': [(1, 50)]}]],
        },
        {
            'testcase_name': 'missing_features',
            'input': ['protein1', 'protein2'],
            'features_directory': ['dir1'],
            'protein_delimiter': '_',
            'mock_side_effect': {
                'dir1/protein1.pkl': False,
                'dir1/protein1.pkl.xz': False,
                'dir1/protein2.pkl': False,
                'dir1/protein2.pkl.xz': False,
            },
            'expected_exception': FileNotFoundError,
            'expected_exception_message': "['protein1', 'protein2'] not found in ['dir1']",
        },
        {
            'testcase_name': 'invalid_format',
            'input': ['protein1::1-10'],
            'features_directory': ['dir1'],
            'protein_delimiter': '_',
            'mock_side_effect': {},
            'expected_exception': SystemExit,
        },
        {
            'testcase_name': 'feature_exists_in_multiple_dirs',
            'input': ['protein1'],
            'features_directory': ['dir1', 'dir2'],
            'protein_delimiter': '_',
            'mock_side_effect': {
                'dir1/protein1.pkl': False,
                'dir1/protein1.pkl.xz': False,
                'dir2/protein1.pkl': True,
                'dir2/protein1.pkl.xz': False,
            },
            'expected_result': [[{'protein1': 'all'}]],
        },
        # New test cases for JSON handling
        {
            'testcase_name': 'single_json_file',
            'input': ['rna.json'],
            'features_directory': ['dir1'],
            'protein_delimiter': '+',
            'mock_side_effect': {
                'dir1/rna.json': True,
            },
            'expected_result': [[{'json_input': 'dir1/rna.json'}]],
        },
        {
            'testcase_name': 'json_with_protein',
            'input': ['protein1+rna.json'],
            'features_directory': ['dir1'],
            'protein_delimiter': '+',
            'mock_side_effect': {
                'dir1/protein1.pkl': True,
                'dir1/protein1.pkl.xz': False,
                'dir1/rna.json': True,
            },
            'expected_result': [[{'protein1': 'all'}, {'json_input': 'dir1/rna.json'}]],
        },
        {
            'testcase_name': 'missing_json_file',
            'input': ['rna.json'],
            'features_directory': ['dir1'],
            'protein_delimiter': '+',
            'mock_side_effect': {
                'dir1/rna.json': False,
            },
            'expected_exception': FileNotFoundError,
            'expected_exception_message': "['rna.json'] not found in ['dir1']",
        },
        {
            'testcase_name': 'json_in_multiple_dirs',
            'input': ['rna.json'],
            'features_directory': ['dir1', 'dir2'],
            'protein_delimiter': '+',
            'mock_side_effect': {
                'dir1/rna.json': False,
                'dir2/rna.json': True,
            },
            'expected_result': [[{'json_input': 'dir2/rna.json'}]],
        },
    )
    def test_parse_fold(self, input, features_directory, protein_delimiter, mock_side_effect,
                        expected_result=None, expected_exception=None, expected_exception_message=None):
        """Test parse_fold with different input scenarios"""
        with mock.patch('alphapulldown.utils.modelling_setup.exists') as mock_exists, \
             mock.patch('sys.exit') as mock_exit:
            mock_exists.side_effect = lambda path: mock_side_effect.get(path, False)
            # Mock sys.exit to raise SystemExit exception
            mock_exit.side_effect = SystemExit
            logging.info(f"Testing with input: {input}, features_directory: {features_directory}, "
                         f"protein_delimiter: '{protein_delimiter}'")
            logging.info(f"Mock side effects: {mock_side_effect}")
            if expected_exception:
                with self.assertRaises(expected_exception) as context:
                    result = parse_fold(input, features_directory, protein_delimiter)
                if expected_exception_message:
                    self.assertEqual(str(context.exception), expected_exception_message)
            else:
                result = parse_fold(input, features_directory, protein_delimiter)
                logging.info(f"Result: {result}, Expected: {expected_result}")
                self.assertEqual(result, expected_result)
