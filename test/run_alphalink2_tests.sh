#!/bin/bash
#SBATCH --job-name=tests_on_gpu
#SBATCH -o tests_on_alphalink_out.txt
#SBATCH -e tests_on_alphalink_error.txt
#SBATCH --time=05-00:00:00
#SBATCH --qos=highest
#SBATCH -p gpu-el8
#SBATCH -N 1
#SBATCH -C gaming
#SBATCH --ntasks=8
#SBATCH --mem-per-gpu=64GB
module load Anaconda3
eval "$(conda shell.bash hook)"
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
AlphaPulldownENV=$1
source activate $AlphaPulldownENV

# export PYTHONPATH=/g/kosinski/geoffrey/alphapulldown/:$PYTHONPATH
# export PYTHONPATH=/g/kosinski/geoffrey/alphapulldown/alphafold:$PYTHONPATH

python3 -m unittest discover ./test/alphalink