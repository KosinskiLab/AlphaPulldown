#!/bin/bash

#A typical run takes couple of hours but may be much longer
#SBATCH --job-name=test_predict_structure
#SBATCH --time=05:00:00

#log files:
#SBATCH -e %x.%j_err.txt
#SBATCH -o %x.%j_out.txt

#qos sets priority, you can set to high or highest but there is a limit of high priority jobs per user: https://wiki.embl.de/cluster/Slurm#QoS
#SBATCH --qos=normal

#SBATCH -p gpu-el8
#SBATCH -C gaming

#Reserve the entire GPU so no-one else slows you down
#SBATCH --gres=gpu:1

#Limit the run to a single node
#SBATCH -N 1

#Adjust this depending on the node
#SBATCH --ntasks=8
#SBATCH --mem=16000

module load Anaconda3 
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

#Print error message if no arguments and help message that explains how to use the script
if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    echo "Usage: test_predict_structure.sh YourAlphaPulldownEnvironment"
    exit 1
fi

AlphaPulldownENV=$1
source activate $AlphaPulldownENV

MAXRAM=$(echo `ulimit -m` '/ 1024.0'|bc)
GPUMEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|tail -1`
export XLA_PYTHON_CLIENT_MEM_FRACTION=`echo "scale=3;$MAXRAM / $GPUMEM"|bc`
export TF_FORCE_UNIFIED_MEMORY='1'

python test_predict_structure.py
