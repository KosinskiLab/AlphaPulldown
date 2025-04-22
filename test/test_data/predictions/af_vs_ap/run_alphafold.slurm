#!/bin/bash

#SBATCH --job-name=run_af
#SBATCH --time=1-00:00:00
#SBATCH -e run_alphafold_%A_%a_err.txt
#SBATCH -o run_alphafold_%A_%a_out.txt
#SBATCH --qos=high
#SBATCH -p htc-el8
##SBATCH -C "gpu=A100"
##SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --mem=32000

module load AlphaFold/2.3.2-foss-2023a-CUDA-12.1.1 

MAXRAM=$(echo `ulimit -m` '/ 1024.0'|bc)
GPUMEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|tail -1`
export XLA_PYTHON_CLIENT_MEM_FRACTION=`echo "scale=3;$MAXRAM / $GPUMEM"|bc`
export TF_FORCE_UNIFIED_MEMORY='1'

run_alphafold.py   --output_dir=/g/kosinski/dima/PycharmProjects/AlphaPulldown/test/test_data/predictions/af_vs_ap --fasta_paths=/g/kosinski/dima/PycharmProjects/AlphaPulldown/test/test_data/fastas/A0A024R1R8+P61626_orig.fasta --max_template_date=2050-10-10 --model_preset=multimer --use_precomputed_msas
#To create monomeric features with create_individual_features.py run this command:
#create_individual_features.py   --output_dir=/g/kosinski/dima/PycharmProjects/AlphaPulldown/test/test_data/predictions/af_vs_ap --fasta_paths=/g/kosinski/dima/PycharmProjects/AlphaPulldown/test/test_data/fastas/A0A024R1R8+P61626_orig.fasta --max_template_date=2050-10-10 --use_precomputed_msas --data_dir=/scratch/AlphaFold_DBs/2.3.2
