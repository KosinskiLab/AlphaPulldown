1. Clone the GitHub repo
    ```
    git clone --recurse-submodules git@github.com:KosinskiLab/AlphaPulldown.git
    cd AlphaPulldown 
    git submodule init
    git submodule update 
    ```
1. Create the Conda environment as described in [https://github.com/KosinskiLab/AlphaPulldown/blob/installation-intro-update/README.md#create-anaconda-environment](https://github.com/KosinskiLab/AlphaPulldown/tree/main?tab=readme-ov-file#create-anaconda-environment) 
1. Add AlphaPulldown package and its submodules to the Conda environment
    ```
    source activate AlphaPulldown
    cd AlphaPulldown
    pip install -e .
    cp alphapulldown/package_data/stereo_chemical_props.txt alphafold/alphafold/common/
    pip install -e ColabFold --no-deps
    pip install -e alphafold --no-deps
    ```
    You need to do it only once.
1. When you want to develop, activate the environment, modify files, and the changes should be automatically recognized.
1. Test your package during development using tests in ```test/```, e.g.:
   ```
   pip install pytest
   pytest -s test/
   pytest -s test/test_predictions_slurm.py
   pytest -s test/test_features_with_templates.py::TestCreateIndividualFeaturesWithTemplates::test_1a_run_features_generation
   ```
1. Before pushing to the remote or submitting pull request
    ```
    pip install .
    pytest -s test/
    ```
    to install the package and test. Pytest for predictions only work if slurm is available. Check the created log files in your current directory.
    
    
