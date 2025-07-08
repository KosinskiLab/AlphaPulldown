#!/bin/bash

# AlphaFold3 Feature Creation Commands
# This script creates AlphaFold3 JSON input files from FASTA sequences

# Set database paths
AF2_DB_DIR="/g/alphafold/AlphaFold_DBs/2.3.0"
AF3_DB_DIR="/g/alphafold/AlphaFold_DBs/3.0.0"

# Create output directories
mkdir -p test/test_data/features/af2_features/{protein,rna,dna,mixed}
mkdir -p test/test_data/features/af3_features/{protein,rna,dna,mixed}

echo "=== Creating AlphaFold2 Features ==="

echo "Creating AlphaFold2 features for protein sequences..."
python alphapulldown/scripts/create_individual_features.py \
  --fasta_paths test/test_data/fastas/A0A024R1R8.fasta,test/test_data/fastas/P61626.fasta \
  --data_dir $AF2_DB_DIR \
  --data_pipeline alphafold2 \
  --output_dir test/test_data/features/af2_features/protein \
  --max_template_date 2021-09-30

echo "=== Creating AlphaFold3 Features ==="

echo "Creating AlphaFold3 features for protein sequences..."
python alphapulldown/scripts/create_individual_features.py \
  --fasta_paths test/test_data/fastas/A0A024R1R8.fasta,test/test_data/fastas/P61626.fasta \
  --data_dir $AF3_DB_DIR \
  --data_pipeline alphafold3 \
  --output_dir test/test_data/features/af3_features/protein \
  --max_template_date 2021-09-30 \
  --use_mmseqs2

echo "Creating AlphaFold3 features for RNA sequences..."
python alphapulldown/scripts/create_individual_features.py \
  --fasta_paths test/test_data/fastas/rna.fasta \
  --data_dir $AF3_DB_DIR \
  --data_pipeline alphafold3 \
  --output_dir test/test_data/features/af3_features/rna \
  --max_template_date 2021-09-30 \
  --use_mmseqs2

echo "Creating AlphaFold3 features for DNA sequences..."
python alphapulldown/scripts/create_individual_features.py \
  --fasta_paths test/test_data/fastas/dna_af3.fasta \
  --data_dir $AF3_DB_DIR \
  --data_pipeline alphafold3 \
  --output_dir test/test_data/features/af3_features/dna \
  --max_template_date 2021-09-30 \
  --use_mmseqs2

echo "Creating AlphaFold3 features for protein and RNA sequences..."
python alphapulldown/scripts/create_individual_features.py \
  --fasta_paths test/test_data/fastas/protein_rna_af3.fasta \
  --data_dir $AF3_DB_DIR \
  --data_pipeline alphafold3 \
  --output_dir test/test_data/features/af3_features/protein_rna \
  --max_template_date 2021-09-30 \
  --use_mmseqs2

echo "=== Converting AlphaFold2 Features to AlphaFold3 JSON ==="

# Convert AlphaFold2 features to AlphaFold3 JSON format
echo "Converting AlphaFold2 protein features to AlphaFold3 JSON..."
python convert_to_alphafold3_json.py \
  --pickle_dir test/test_data/features/af2_features/protein \
  --output_dir test/test_data/features/af2_features/protein


echo "=== Feature Creation Complete ==="
echo ""
echo "Generated AlphaFold2 pickle files:"
find test/test_data/features/af2_features -name "*.pkl" | sort
echo ""
echo "Generated AlphaFold3 pickle files:"
find test/test_data/features/af3_features -name "*.pkl" | sort
echo ""
echo "Generated AlphaFold2 JSON files:"
find test/test_data/features/af2_features -name "*_af3_input.json" | sort
echo ""
echo "Generated AlphaFold3 JSON files:"
find test/test_data/features/af3_features -name "*_af3_input.json" | sort 