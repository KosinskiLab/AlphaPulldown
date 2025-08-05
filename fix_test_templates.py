#!/usr/bin/env python3
"""
Script to add empty template features to TEST.pkl using ColabFold's mk_mock_template function.
This will make the test data compatible with AlphaLink2.
"""

import pickle
import sys
import os

# Add ColabFold to path
sys.path.insert(0, 'ColabFold')

from colabfold.batch import mk_mock_template
import numpy as np

def fix_test_templates():
    """Add empty template features to TEST.pkl"""
    
    # Load the TEST.pkl file
    test_file = "test/test_data/features/TEST.pkl"
    
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found")
        return
    
    print(f"Loading {test_file}...")
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"Original data type: {type(test_data)}")
    if hasattr(test_data, 'feature_dict'):
        print(f"Original feature dict keys: {list(test_data.feature_dict.keys())}")
        print(f"Sequence length: {len(test_data.sequence)}")
        
        # Create empty template features using ColabFold's mk_mock_template
        print("Creating empty template features...")
        empty_templates = mk_mock_template(test_data.sequence, num_temp=1)
        
        # Add the empty template features to the feature dict
        for key, value in empty_templates.items():
            test_data.feature_dict[key] = value
            if hasattr(value, 'shape'):
                print(f"Added template feature: {key} with shape {value.shape}")
            else:
                print(f"Added template feature: {key} with type {type(value)} and length {len(value)}")
        
        # Save the modified data
        output_file = "test/test_data/features/TEST_fixed.pkl"
        print(f"Saving fixed data to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        print("Done! The fixed file is saved as TEST_fixed.pkl")
        print("You can now use this file for testing AlphaLink2.")
        
    else:
        print("Error: test_data does not have feature_dict attribute")
        print(f"Available attributes: {dir(test_data)}")

if __name__ == "__main__":
    fix_test_templates() 