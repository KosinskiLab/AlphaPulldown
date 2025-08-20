"""
Test dropout_during_inference flag functionality
"""

import unittest
import sys
import os

class TestDropoutDuringInference(unittest.TestCase):

    def test_flag_addition_in_run_structure_prediction(self):
        """Test that the dropout_during_inference flag was added to run_structure_prediction.py"""
        script_path = os.path.join(os.path.dirname(__file__), '..', 'alphapulldown', 'scripts', 'run_structure_prediction.py')
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check that the flag is defined
        self.assertIn('dropout_during_inference', content)
        self.assertIn('FLAGS.dropout_during_inference', content)
        self.assertIn('enables dropout during inference', content)

    def test_backend_function_signatures(self):
        """Test that all backend setup functions accept the dropout_during_inference parameter"""
        
        # Test alphafold_backend.py
        alphafold_backend_path = os.path.join(os.path.dirname(__file__), '..', 'alphapulldown', 'folding_backend', 'alphafold_backend.py')
        with open(alphafold_backend_path, 'r') as f:
            content = f.read()
        
        self.assertIn('dropout_during_inference=False', content)
        self.assertIn('eval_dropout = True', content)
        
        # Test alphafold3_backend.py
        alphafold3_backend_path = os.path.join(os.path.dirname(__file__), '..', 'alphapulldown', 'folding_backend', 'alphafold3_backend.py')
        with open(alphafold3_backend_path, 'r') as f:
            content = f.read()
        
        self.assertIn('dropout_during_inference', content)
        self.assertIn('eval_dropout = True', content)
        
        # Test alphalink_backend.py
        alphalink_backend_path = os.path.join(os.path.dirname(__file__), '..', 'alphapulldown', 'folding_backend', 'alphalink_backend.py')
        with open(alphalink_backend_path, 'r') as f:
            content = f.read()
        
        self.assertIn('dropout_during_inference', content)
        self.assertIn('eval_dropout = True', content)
        
        # Test unifold_backend.py
        unifold_backend_path = os.path.join(os.path.dirname(__file__), '..', 'alphapulldown', 'folding_backend', 'unifold_backend.py')
        with open(unifold_backend_path, 'r') as f:
            content = f.read()
        
        self.assertIn('dropout_during_inference', content)
        self.assertIn('eval_dropout = True', content)

    def test_flag_documentation(self):
        """Test that the flag is properly documented in all files"""
        
        # Check run_structure_prediction.py for flag description
        script_path = os.path.join(os.path.dirname(__file__), '..', 'alphapulldown', 'scripts', 'run_structure_prediction.py')
        with open(script_path, 'r') as f:
            content = f.read()
        
        self.assertIn('improved uncertainty estimation', content)
        
        # Check backend files for parameter documentation
        backend_files = [
            'alphafold_backend.py',
            'alphalink_backend.py',
            'unifold_backend.py'
        ]
        
        for backend_file in backend_files:
            backend_path = os.path.join(os.path.dirname(__file__), '..', 'alphapulldown', 'folding_backend', backend_file)
            with open(backend_path, 'r') as f:
                content = f.read()
            
            self.assertIn('dropout_during_inference', content)
            self.assertIn('uncertainty estimation', content.lower())

if __name__ == '__main__':
    unittest.main()