#!/bin/bash
#SBATCH --job-name=test_predict_structure
#SBATCH --time=01:00:00
#SBATCH --qos=normal
#SBATCH -p gpu-el8
#SBATCH -C gaming
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --mem=16000

#module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

#Print error message if no arguments and help message that explains how to use the script
if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    echo "Usage: test_predict_structure.sh YourAlphaPulldownEnvironment"
    exit 1
fi

AlphaPulldownENV=$1
conda activate $AlphaPulldownENV

MAXRAM=$(echo `ulimit -m` '/ 1024.0'|bc)
GPUMEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|tail -1`
export XLA_PYTHON_CLIENT_MEM_FRACTION=`echo "scale=3;$MAXRAM / $GPUMEM"|bc`
export TF_FORCE_UNIFIED_MEMORY='1'
echo "Running TestScript::testRun_$SLURM_ARRAY_TASK_ID"
pytest -s test/check_predict_structure.py::TestScript::testRun_$SLURM_ARRAY_TASK_ID