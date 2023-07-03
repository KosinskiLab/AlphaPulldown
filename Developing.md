1. Clone the GitHub repo
    ```
    git clone --recurse-submodules git@github.com:KosinskiLab/AlphaPulldown.git
    cd AlphaPulldown 
    git submodule init
    git submodule update 
    ```
1. Create the Conda environment and install the latest version of AlphaPulldown as described in https://github.com/KosinskiLab/AlphaPulldown/tree/DevelopReadme#for-users-pip-installation
1. Add AlphaPulldown package and its submodules to the Conda environment
    ```
    cd AlphaPulldown
    pip install -e .
    pip install -e alphapulldown/ColabFold --no-deps
    pip install -e alphafold --no-deps
    ```
    You need to do it only once.
1. When you want to develop, activate the environment, modify files, and the changes should be automatically recognized.
1. Test your package during development using tests in ```test/```, e.g.:
   ```
   pip install pytest
   pytest
   pytest test
   python test/test_predict_structure.py
   sbatch test/test_predict_structure.sh
   python -m unittest test/test_predict_structure.<name of the test>
   ```
1. Before pushing to the remote or submitting pull request
    ```
    pip install .
    pytest test
    ```
    to install the package and test
    
    
