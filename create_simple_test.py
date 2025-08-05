#!/usr/bin/env python3
"""
Script to create a simple test protein without template features for AlphaLink testing.
"""

import pickle
import numpy as np
import torch
import sys
import os

# Add ColabFold to path
sys.path.insert(0, 'ColabFold')

from colabfold.batch import mk_mock_template
from alphapulldown.objects import MonomericObject

def create_simple_test_protein():
    """Create a simple test protein with empty template features"""
    
    # Create a simple sequence
    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUAIGLNKALN"
    
    # Create basic feature dict without templates
    feature_dict = {
        'aatype': np.array([0] * len(sequence)),  # All alanine
        'between_segment_residues': np.array([0] * len(sequence)),
        'domain_name': np.array([b'protein']),
        'residue_index': np.arange(len(sequence)),
        'seq_length': np.array([len(sequence)]),
        'sequence': sequence,
        'deletion_matrix_int': np.zeros((1, len(sequence))),
        'msa': np.array([[0] * len(sequence)]),  # Single MSA sequence
        'num_alignments': np.array([1]),
        'msa_species_identifiers': np.array([b'protein']),
        'deletion_matrix_int_all_seq': np.zeros((1, len(sequence))),
        'msa_all_seq': np.array([[0] * len(sequence)]),
        'msa_species_identifiers_all_seq': np.array([b'protein']),
        # Add missing features for AlphaLink2
        'deletion_matrix': np.zeros((1, len(sequence))),
        'extra_deletion_matrix': np.zeros((1, len(sequence))),
        'msa_mask': np.ones((1, len(sequence))),
        'msa_row_mask': np.ones((1,)),
        # Add multimer-specific features
        'asym_id': np.zeros(len(sequence), dtype=np.int32),  # Single chain
        'entity_id': np.zeros(len(sequence), dtype=np.int32),  # Single entity
        'sym_id': np.ones(len(sequence), dtype=np.int32),     # Single symmetry group
    }
    
    # Add empty template features using ColabFold's mk_mock_template
    print("Creating empty template features...")
    empty_templates = mk_mock_template(sequence, num_temp=1)
    
    # Add the empty template features to the feature dict
    for key, value in empty_templates.items():
        if key == 'template_aatype':
            # Convert one-hot encoding to integer encoding for AlphaLink2
            # The one-hot tensor has shape (1, 231, 22), convert to (1, 231)
            if hasattr(value, 'shape') and len(value.shape) == 3:
                value = np.argmax(value, axis=-1)
        elif key == 'template_sum_probs':
            # Reshape to match expected schema: (1, 1) instead of (1,)
            if hasattr(value, 'shape') and len(value.shape) == 1:
                value = value.reshape(1, 1)
        feature_dict[key] = value
        if hasattr(value, 'shape'):
            print(f"Added template feature: {key} with shape {value.shape}")
        else:
            print(f"Added template feature: {key} with type {type(value)} and length {len(value)}")
    
    # Create the MonomericObject
    test_protein = MonomericObject("SIMPLE_TEST", sequence)
    test_protein.feature_dict = feature_dict
    
    # Save the test protein
    output_file = "test/test_data/features/SIMPLE_TEST.pkl"
    print(f"Saving simple test protein to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(test_protein, f)
    
    print("Done! Created SIMPLE_TEST.pkl with empty template features.")
    print(f"Sequence length: {len(sequence)}")
    print(f"Feature dict keys: {list(feature_dict.keys())}")
    print(f"Has template features: {any('template' in k for k in feature_dict.keys())}")

if __name__ == "__main__":
    create_simple_test_protein() 