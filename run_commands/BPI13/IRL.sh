#!/bin/bash
#SBATCH --job-name=BPI13_IRL
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

python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method irl --irl-lr 0.001 --irl-epochs 100 --greedy --data-config-file BPI13.yaml
python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method irl --irl-lr 0.005 --irl-epochs 100 --greedy --data-config-file BPI13.yaml
python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method irl --irl-lr 0.01 --irl-epochs 100 --greedy --data-config-file BPI13.yaml
python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method irl --irl-lr 0.05 --irl-epochs 100 --greedy --data-config-file BPI13.yaml
python main.py --seed 42 --input-file input_data/BPI13_processed.csv --method irl --irl-lr 0.1 --irl-epochs 100 --greedy --data-config-file BPI13.yaml