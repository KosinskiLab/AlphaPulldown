#!/bin/bash
#SBATCH --job-name=tests_on_gpu
#SBATCH -o tests_on_gpu_out.txt
#SBATCH -e tests_on_gpu_error.txt
#SBATCH --time=01-00:00:00
#SBATCH --qos=highest
#SBATCH -p htc-el8
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --mem=32000
module load Anaconda3
eval "$(conda shell.bash hook)"

AlphaPulldownENV=$1
source activate $AlphaPulldownENV

# export PYTHONPATH=/g/kosinski/geoffrey/alphapulldown/:$PYTHONPATH
# export PYTHONPATH=/g/kosinski/geoffrey/alphapulldown/alphafold:$PYTHONPATH

python3 -m unittest discover ./test/on_cpu
