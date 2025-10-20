import os
import sys
import argparse
import json
import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_workflow_data import run_simulation
from get_agent_trajectories import get_agent_trajectories
from learn_reward import learn_reward, test_reward
from utils.utils import split_log, split_log_random, split_log_train_val_test, get_unique_values_for_categorical_features
from baselines.decision_tree import train_decision_trees_per_agent, evaluate_topk_decisions_per_agent, train_decision_tree_global, evaluate_topk_decisions_global
from baselines.logistic_regression import train_logistic_regression_per_agent, evaluate_topk_decisions_per_agent as evaluate_lr_per_agent, train_logistic_regression_global, evaluate_topk_decisions_global as evaluate_lr_global
from baselines.svm import train_svm_per_agent, evaluate_topk_decisions_per_agent as evaluate_svm_per_agent
from baselines.xgboost import train_xgboost_per_agent, evaluate_topk_decisions_per_agent as evaluate_xgb_per_agent
from baselines.neural_network import train_neural_network_per_agent, evaluate_topk_decisions_per_agent as evaluate_nn_per_agent
from baselines.frequentist import train_freq_baseline_per_agent, evaluate_freq_baseline_per_agent
from baselines.random import evaluate_random_baseline_per_agent
from baselines.fifo import evaluate_fifo_baseline_per_agent
from generate_workflow_data import agent_platinum_priority, agent_region_specific, agent_high_value, agent_first_come_first_serve, agent_earliest_due_date, agent_smallest_processing_time, agent_random, agent_level_priority



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="IRL4BPM experiments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--input-file", type=str, default="input_data/ticket_log_FCFS.csv", help="Path to input CSV log")
    parser.add_argument("--method", type=str, default="irl", help="Method to use for learning reward")
    parser.add_argument("--data-config-file", type=str, default="ticket_log.yaml", help="Data config to use")
    # IRL
    parser.add_argument("--greedy", action='store_true', help="Use greedy ranking for IRL evaluation")
    parser.add_argument("--irl-tau", type=float, default=1.0, help="Softmax temperature for IRL (if not greedy)")
    parser.add_argument("--ks", type=str, default="1,2,3", help="Comma-separated Top-k values for evaluation")
    parser.add_argument("--irl-lr", type=float, default=0.005, help="Learning rate for IRL")
    parser.add_argument("--irl-epochs", type=int, default=100, help="Number of epochs for IRL")

    # Decision tree
    parser.add_argument("--dt-negatives", type=int, default=10, help="Negatives per positive for DT training")
    parser.add_argument("--dt-max-depth", type=int, default=5, help="Max depth for DT (use -1 for None)")
    
    # Logistic Regression
    parser.add_argument("--lr-negatives", type=int, default=10, help="Negatives per positive for LR training")
    parser.add_argument("--lr-C", type=float, default=1.0, help="Regularization strength for LR (inverse of C)")
    parser.add_argument("--lr-max-iter", type=int, default=1000, help="Maximum iterations for LR")
    parser.add_argument("--lr-standardize", action='store_true', help="Standardize features for LR")
    
    # SVM
    parser.add_argument("--svm-negatives", type=int, default=10, help="Negatives per positive for SVM training")
    parser.add_argument("--svm-C", type=float, default=1.0, help="Regularization strength for SVM")
    parser.add_argument("--svm-kernel", type=str, default='rbf', help="Kernel type for SVM")
    parser.add_argument("--svm-gamma", type=str, default='scale', help="Gamma parameter for SVM")
    parser.add_argument("--svm-standardize", action='store_true', help="Standardize features for SVM")
    
    # XGBoost
    parser.add_argument("--xgb-negatives", type=int, default=10, help="Negatives per positive for XGBoost training")
    parser.add_argument("--xgb-n-estimators", type=int, default=100, help="Number of estimators for XGBoost")
    parser.add_argument("--xgb-max-depth", type=int, default=6, help="Max depth for XGBoost")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1, help="Learning rate for XGBoost")
    parser.add_argument("--xgb-subsample", type=float, default=0.8, help="Subsample ratio for XGBoost")
    
    # Neural Network
    parser.add_argument("--nn-negatives", type=int, default=10, help="Negatives per positive for NN training")
    parser.add_argument("--nn-hidden-layers", type=str, default="100,50", help="Hidden layer sizes for NN (comma-separated)")
    parser.add_argument("--nn-activation", type=str, default='relu', help="Activation function for NN")
    parser.add_argument("--nn-solver", type=str, default='adam', help="Solver for NN")
    parser.add_argument("--nn-alpha", type=float, default=0.0001, help="L2 regularization for NN")
    parser.add_argument("--nn-learning-rate", type=float, default=0.001, help="Learning rate for NN")
    parser.add_argument("--nn-max-iter", type=int, default=1000, help="Maximum iterations for NN")
    parser.add_argument("--nn-standardize", action='store_true', help="Standardize features for NN")
    # Optional data generation
    parser.add_argument("--simulate", action="store_true", help="Generate a synthetic log before training")
    parser.add_argument("--n-cases", type=int, default=1000, help="Number of cases for simulation")

    args = parser.parse_args()

    SEED = args.seed
    name_output_file = args.input_file

    # Optionally generate workflow data
    if args.simulate:
        print("Generating workflow data...")
        run_simulation(
            n_cases=args.n_cases,
            agents={
                'Alice': agent_platinum_priority,
                'Bob': agent_platinum_priority,
                'Carol': agent_platinum_priority
            },
            name_output_file=name_output_file,
            seed=SEED
        )
    else:
        print("Using existing workflow data...")


    # load numerical and categorical features from data_configs/
    with open(os.path.join("data_configs", args.data_config_file), "r") as f:
        data_config = yaml.safe_load(f)
    numerical_features = data_config["numerical_features"]
    categorical_features = data_config["categorical_features"]
    temporal_columns = data_config["temporal_columns"]
    print("Numerical features: ", numerical_features)
    print("Categorical features: ", categorical_features)

    # log = pd.read_csv(name_output_file, parse_dates=['arrival', 'start_timestamp', 'end_timestamp', 'due_date'])
    log = pd.read_csv(name_output_file)
    for col in temporal_columns:
        log[col] = pd.to_datetime(log[col], format='mixed').dt.tz_localize(None)

    # agent column should be converted to string
    log['agent'] = log['agent'].astype(str)
    if 'case_id' in log.columns:
        log['case_id'] = log['case_id'].astype(str)
    else:
        log['id'] = log['id'].astype(str)

    cat_features_dict = get_unique_values_for_categorical_features(log, categorical_features)
    print("Cat features dict: ", cat_features_dict)

    print("Splitting log into train, val, test...")
    train_log, val_log, test_log = split_log_train_val_test(log)
    ks = [int(k) for k in str(args.ks).split(',') if k.strip()]
    max_depth_val = None if args.dt_max_depth is not None and args.dt_max_depth < 0 else args.dt_max_depth

    current_results = None
    val_results = None
    irl_weights = None
    if args.method == "irl":
        print("Training IRL...")
        # get agent trajectories
        print("Getting agent trajectories...")
        train_log = log
        trajectories = get_agent_trajectories(log=train_log, numerical_features=numerical_features, categorical_features=categorical_features)

        # learn reward
        print("Learning reward...")
        weights, scaler = learn_reward(trajectories, cat_features_dict, numerical_features, seed=SEED, lr=args.irl_lr, epochs=args.irl_epochs)
        irl_weights = weights

        # validate reward
        print("Validating reward...")
        val_trajectories = get_agent_trajectories(log=val_log, numerical_features=numerical_features, categorical_features=categorical_features)
        irl_results = test_reward(val_trajectories, weights, scaler, cat_features_dict, numerical_features, ks=ks, greedy=args.greedy, tau=args.irl_tau)
        val_results = irl_results

        # test reward
        print("Testing reward...")
        test_trajectories = get_agent_trajectories(log=test_log, numerical_features=numerical_features, categorical_features=categorical_features)
        irl_results = test_reward(test_trajectories, weights, scaler, cat_features_dict, numerical_features, ks=ks, greedy=args.greedy, tau=args.irl_tau)
        current_results = irl_results


    elif args.method == "dt_local":
        print("Training Decision Tree (local)...")
        # Decision tree local
        models, encoders = train_decision_trees_per_agent(
            train_log, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            negatives_per_pos=args.dt_negatives, 
            max_depth=max_depth_val, 
            random_state=SEED
        )
        # get validation results
        dt_results_local = evaluate_topk_decisions_per_agent(
            val_log, models, encoders, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            ks=ks
        )
        val_results = dt_results_local
        # get test results
        dt_results_local = evaluate_topk_decisions_per_agent(
            test_log, models, encoders, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            ks=ks
        )
        current_results = dt_results_local

    elif args.method == "random":
        print("Training Random...")
        # Random
        rand_results = evaluate_random_baseline_per_agent(test_log, train_log, ks=ks, n_runs=1, random_state=SEED)
        current_results = rand_results

    elif args.method == "lr_local":
        print("Training Logistic Regression (local)...")
        # Logistic Regression local
        models, encoders, scalers = train_logistic_regression_per_agent(
            train_log, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            negatives_per_pos=args.lr_negatives, 
            C=args.lr_C, 
            max_iter=args.lr_max_iter,
            random_state=SEED,
            standardize_features=args.lr_standardize
        )
        # get validation results
        lr_results_local = evaluate_lr_per_agent(
            val_log, models, encoders, scalers, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            ks=ks
        )
        val_results = lr_results_local
        # get test results
        lr_results_local = evaluate_lr_per_agent(
            test_log, models, encoders, scalers, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            ks=ks
        )
        current_results = lr_results_local

    elif args.method == "svm":
        print("Training SVM...")
        # SVM
        models, encoders, scalers = train_svm_per_agent(
            train_log, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            negatives_per_pos=args.svm_negatives, 
            C=args.svm_C, 
            kernel=args.svm_kernel,
            gamma=args.svm_gamma,
            random_state=SEED,
            standardize_features=args.svm_standardize
        )
        # get validation results
        svm_results = evaluate_svm_per_agent(
            val_log, models, encoders, scalers, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            ks=ks
        )
        val_results = svm_results
        # get test results
        svm_results = evaluate_svm_per_agent(
            test_log, models, encoders, scalers, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            ks=ks
        )
        current_results = svm_results

    elif args.method == "xgboost":
        print("Training XGBoost...")
        # XGBoost
        models, encoders = train_xgboost_per_agent(
            train_log, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            negatives_per_pos=args.xgb_negatives, 
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            random_state=SEED
        )
        # get validation results
        xgb_results = evaluate_xgb_per_agent(
            val_log, models, encoders, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            ks=ks
        )
        val_results = xgb_results
        # get test results
        xgb_results = evaluate_xgb_per_agent(
            test_log, models, encoders, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            ks=ks
        )
        current_results = xgb_results

    elif args.method == "neural_network":
        print("Training Neural Network...")
        # Neural Network
        hidden_layers = tuple(int(x) for x in args.nn_hidden_layers.split(','))
        models, encoders, scalers = train_neural_network_per_agent(
            train_log, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            negatives_per_pos=args.nn_negatives, 
            hidden_layer_sizes=hidden_layers,
            activation=args.nn_activation,
            solver=args.nn_solver,
            alpha=args.nn_alpha,
            learning_rate_init=args.nn_learning_rate,
            max_iter=args.nn_max_iter,
            random_state=SEED,
            standardize_features=args.nn_standardize
        )
        # get validation results
        nn_results = evaluate_nn_per_agent(
            val_log, models, encoders, scalers, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            ks=ks
        )
        val_results = nn_results
        # get test results
        nn_results = evaluate_nn_per_agent(
            test_log, models, encoders, scalers, 
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            ks=ks
        )
        current_results = nn_results

    elif args.method == "fifo":
        print("Training FIFO...")
        # FIFO
        fifo_results = evaluate_fifo_baseline_per_agent(test_log, train_log, ks=ks)
        current_results = fifo_results

    # --- Append a single summary row to overall_results.csv ---
    overall_path = os.path.join("results", "results_new.csv")
    os.makedirs(os.path.dirname(overall_path), exist_ok=True)

    # Prepare summary payload
    method_label = args.method
    agent_names = sorted(list(current_results.keys())) if current_results else []
    agent_topk = {}
    top1_list = []
    for ag in agent_names:
        m = current_results[ag]
        # collect accuracies in ks order
        agent_topk[ag] = [m.get(f"top{k}_accuracy", None) for k in ks]
        if m.get("top1_accuracy") is not None:
            top1_list.append(m["top1_accuracy"])
    avg_top1 = float(sum(top1_list) / len(top1_list)) if top1_list else None

    # get average top-1 accuracy of val results
    val_avg_top1 = None
    if val_results is not None:
        val_top1_list = []
        for ag in sorted(list(val_results.keys())):
            m = val_results[ag]
            if m.get("top1_accuracy") is not None:
                val_top1_list.append(m["top1_accuracy"])
        if val_top1_list:
            val_avg_top1 = float(sum(val_top1_list) / len(val_top1_list))

    row = {
        "input_dataset": os.path.basename(name_output_file),
        "method": method_label,
        "agents": ",".join(agent_names),
        "agent_topk": agent_topk,
        "avg_top1": avg_top1,
        "val_avg_top1": val_avg_top1,
        "seed": SEED,
        # IRL hparams
        "irl_lr": args.irl_lr if hasattr(args, "irl_lr") else None,
        "irl_epochs": args.irl_epochs if hasattr(args, "irl_epochs") else None,
        "irl_greedy": args.greedy if hasattr(args, "greedy") else None,
        "irl_tau": args.irl_tau if hasattr(args, "irl_tau") else None,
        "irl_weights": irl_weights if irl_weights is not None else None,
        # DT hparams
        "dt_negatives": args.dt_negatives if hasattr(args, "dt_negatives") else None,
        "dt_max_depth": max_depth_val,
    }

    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(overall_path)
    df_row.to_csv(overall_path, mode='a', header=write_header, index=False)
    print(f"Appended summary row to {overall_path}")



    
    