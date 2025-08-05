#!/usr/bin/env python3
"""
Simple test script to run AlphaLink with SIMPLE_TEST protein.
"""

import subprocess
import os

def test_simple_alphalink():
    """Test AlphaLink with SIMPLE_TEST protein"""
    
    # Set up environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Set threading controls
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["NUMEXPR_NUM_THREADS"] = "4"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    # Create output directory
    output_dir = "test/test_data/predictions/alphalink_backend/test_simple"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run AlphaLink
    cmd = [
        "python", "alphapulldown/scripts/run_structure_prediction.py",
        "--output_directory", output_dir,
        "--num_cycle", "1",
        "--num_predictions_per_model", "1",
        "--data_directory", "/scratch/AlphaFold_DBs/alphalink_weights/AlphaLink-Multimer_SDA_v3.pt",
        "--features_directory", "/g/kosinski/dima/PycharmProjects/AlphaPulldown/test/test_data/features",
        "--pair_msa", "",
        "--nomsa_depth_scan", "",
        "--nomultimeric_template", "",
        "--fold_backend", "alphalink",
        "--nocompress_result_pickles", "",
        "--noremove_result_pickles", "",
        "--remove_keys_from_pickles", "",
        "--use_ap_style", "",
        "--use_gpu_relax", "",
        "--protein_delimiter", "+",
        "--models_to_relax", "None",
        "--input", "SIMPLE_TEST"
    ]
    
    print("Running AlphaLink with SIMPLE_TEST...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ SUCCESS: AlphaLink test passed!")
        else:
            print("❌ FAILED: AlphaLink test failed!")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_simple_alphalink() 