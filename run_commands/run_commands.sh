#!/bin/bash
#SBATCH --job-name=base
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

# FCFS
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method random
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method fifo
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method irl --irl-lr 0.001 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method irl --irl-lr 0.005 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method irl --irl-lr 0.01 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method irl --irl-lr 0.05 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method irl --irl-lr 0.1 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method dt_local --dt-negatives 10 --dt-max-depth 5
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method dt_local --dt-negatives 10 --dt-max-depth 10
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method dt_local --dt-negatives 10 --dt-max-depth 20
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method dt_local --dt-negatives 10 --dt-max-depth -1
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

# EDD
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method random
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method fifo
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method irl --irl-lr 0.001 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method irl --irl-lr 0.005 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method irl --irl-lr 0.01 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method irl --irl-lr 0.05 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method irl --irl-lr 0.1 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method dt_local --dt-negatives 10 --dt-max-depth 5
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method dt_local --dt-negatives 10 --dt-max-depth 10
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method dt_local --dt-negatives 10 --dt-max-depth 20
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method dt_local --dt-negatives 10 --dt-max-depth -1
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

# SPT
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method random
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method fifo
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method irl --irl-lr 0.001 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method irl --irl-lr 0.005 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method irl --irl-lr 0.01 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method irl --irl-lr 0.05 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method irl --irl-lr 0.1 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method dt_local --dt-negatives 10 --dt-max-depth 5
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method dt_local --dt-negatives 10 --dt-max-depth 10
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method dt_local --dt-negatives 10 --dt-max-depth 20
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method dt_local --dt-negatives 10 --dt-max-depth -1
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

# Random
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method random
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method fifo
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method irl --irl-lr 0.001 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method irl --irl-lr 0.005 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method irl --irl-lr 0.01 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method irl --irl-lr 0.05 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method irl --irl-lr 0.1 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method dt_local --dt-negatives 10 --dt-max-depth 5
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method dt_local --dt-negatives 10 --dt-max-depth 10
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method dt_local --dt-negatives 10 --dt-max-depth 20
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method dt_local --dt-negatives 10 --dt-max-depth -1
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

# value
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method random
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method fifo
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method irl --irl-lr 0.001 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method irl --irl-lr 0.005 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method irl --irl-lr 0.01 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method irl --irl-lr 0.05 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method irl --irl-lr 0.1 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method dt_local --dt-negatives 10 --dt-max-depth 5
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method dt_local --dt-negatives 10 --dt-max-depth 10
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method dt_local --dt-negatives 10 --dt-max-depth 20
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method dt_local --dt-negatives 10 --dt-max-depth -1
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

# platinum
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method random
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method fifo
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method irl --irl-lr 0.001 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method irl --irl-lr 0.005 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method irl --irl-lr 0.01 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method irl --irl-lr 0.05 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method irl --irl-lr 0.1 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method dt_local --dt-negatives 10 --dt-max-depth 5
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method dt_local --dt-negatives 10 --dt-max-depth 10
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method dt_local --dt-negatives 10 --dt-max-depth 20
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method dt_local --dt-negatives 10 --dt-max-depth -1
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

# base scenario
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method random
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method fifo
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method irl --irl-lr 0.001 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method irl --irl-lr 0.005 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method irl --irl-lr 0.01 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method irl --irl-lr 0.05 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method irl --irl-lr 0.1 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method dt_local --dt-negatives 10 --dt-max-depth 5
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method dt_local --dt-negatives 10 --dt-max-depth 10
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method dt_local --dt-negatives 10 --dt-max-depth 20
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method dt_local --dt-negatives 10 --dt-max-depth -1
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

# diff2
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method random
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method fifo
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method irl --irl-lr 0.001 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method irl --irl-lr 0.005 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method irl --irl-lr 0.01 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method irl --irl-lr 0.05 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method irl --irl-lr 0.1 --irl-epochs 100 --greedy
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method dt_local --dt-negatives 10 --dt-max-depth 5
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method dt_local --dt-negatives 10 --dt-max-depth 10
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method dt_local --dt-negatives 10 --dt-max-depth 20
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method dt_local --dt-negatives 10 --dt-max-depth -1
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_diff2.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize