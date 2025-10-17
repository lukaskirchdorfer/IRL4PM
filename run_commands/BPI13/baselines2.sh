#!/bin/bash
#SBATCH --job-name=BPI13_b2
#SBATCH --cpus-per-task=5
#SBATCH --mem=70G
#SBATCH --partition=cpu
#SBATCH --chdir=/ceph/lkirchdo/IRL4PM

# Ensure conda is available
source ~/.bashrc || { echo "Failed to source ~/.bashrc"; exit 1; }

# Activate conda environment
conda activate irl4pm || { echo "Failed to activate irl4pm environment"; exit 1; }

# Navigate to project directory
echo "Changing directory to the irl4pm repository..."
cd /ceph/lkirchdo/IRL4PM || { echo "Failed to change directory"; exit 1; }
echo "Current directory: $(pwd)"

# -------------------------------
echo "Starting Python script execution..."
echo "-------------------------------------"

python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize --data-config-file BPI13.yaml
python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize --data-config-file BPI13.yaml
python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize --data-config-file BPI13.yaml
python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize --data-config-file BPI13.yaml
python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize --data-config-file BPI13.yaml
python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize --data-config-file BPI13.yaml

python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize --data-config-file BPI13.yaml
python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize --data-config-file BPI13.yaml
python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize --data-config-file BPI13.yaml