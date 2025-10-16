#!/bin/bash

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

# random
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

# level
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

# Diff2
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

# Logistic Regression experiments
echo "Running Logistic Regression experiments..."

# FCFS
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize

# EDD
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize

# platinum
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize

# random
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize

# SPT
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize

# value
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize

# base scenario
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method lr_local --lr-negatives 10 --lr-C 1.0 --lr-standardize

echo "Logistic Regression experiments completed."

# SVM experiments
echo "Running SVM experiments..."

# FCFS
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize

# EDD
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize

# platinum
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize

# random
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize

# SPT
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize

# value
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize

# base scenario
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel rbf --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method svm --svm-negatives 10 --svm-C 0.1 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method svm --svm-negatives 10 --svm-C 1.0 --svm-kernel linear --svm-gamma scale --svm-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method svm --svm-negatives 10 --svm-C 10.0 --svm-kernel linear --svm-gamma scale --svm-standardize

echo "SVM experiments completed."

# XGBoost experiments
echo "Running XGBoost experiments..."

# FCFS
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05

# EDD
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05

# platinum
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05

# random
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05

# SPT
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05

# value
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05

# base scenario
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.1
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.01
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method xgboost --xgb-negatives 10 --xgb-learning-rate 0.05

echo "XGBoost experiments completed."

# Neural Network experiments
echo "Running Neural Network experiments..."

# FCFS
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_FCFS.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

# EDD
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_EDD.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

# platinum
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_platinum.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

# random
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_random.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

# SPT
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_SPT.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

# value
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_value.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

# base scenario
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "50,25" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50" --nn-activation relu --nn-solver adam --nn-standardize
python main.py --seed 42 --input-file input_data/ticket_log_base_scenario.csv --method neural_network --nn-negatives 10 --nn-hidden-layers "100,50,25" --nn-activation relu --nn-solver adam --nn-standardize

echo "Neural Network experiments completed."
echo "All new methods experiments completed!"