#!/bin/bash
#SBATCH --job-name=BPI12W_b1
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

python main.py --seed 42 --input-file input_data/BPI12W_arrivals.csv --method random --data-config-file BPI12W.yaml
python main.py --seed 42 --input-file input_data/BPI12W_arrivals.csv --method fifo --data-config-file BPI12W.yaml

python main.py --seed 42 --input-file input_data/BPI12W_arrivals.csv --method dt_local --dt-negatives 10 --dt-max-depth 5 --data-config-file BPI12W.yaml
python main.py --seed 42 --input-file input_data/BPI12W_arrivals.csv --method dt_local --dt-negatives 10 --dt-max-depth 10 --data-config-file BPI12W.yaml
python main.py --seed 42 --input-file input_data/BPI12W_arrivals.csv --method dt_local --dt-negatives 10 --dt-max-depth 20 --data-config-file BPI12W.yaml
python main.py --seed 42 --input-file input_data/BPI12W_arrivals.csv --method dt_local --dt-negatives 10 --dt-max-depth -1 --data-config-file BPI12W.yaml

python main.py --seed 42 --input-file input_data/BPI12W_arrivals.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --data-config-file BPI12W.yaml --lr-standardize

python main.py --seed 42 --input-file input_data/BPI12W_arrivals.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1 --data-config-file BPI12W.yaml
python main.py --seed 42 --input-file input_data/BPI12W_arrivals.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01 --data-config-file BPI12W.yaml
python main.py --seed 42 --input-file input_data/BPI12W_arrivals.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05 --data-config-file BPI12W.yaml