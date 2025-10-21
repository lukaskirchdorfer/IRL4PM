#!/usr/bin/env python3
"""
Evaluation script for IRL4PM results.

This script analyzes the results_new.csv file and computes average top1, top2, and top3
accuracy for each unique input_dataset. For each dataset-method combination, it finds
the configuration with the highest val_avg_top1 score and displays the results in a table.
"""

import pandas as pd
import ast
import numpy as np
from collections import defaultdict
import sys

def parse_agent_topk(agent_topk_str):
    """
    Parse the agent_topk string and extract top1, top2, top3 values for each agent.
    
    Args:
        agent_topk_str: String representation of dictionary with agent topk results
        
    Returns:
        dict: Dictionary with agent names as keys and [top1, top2, top3] as values
    """
    try:
        # Parse the string as a Python dictionary
        agent_dict = ast.literal_eval(agent_topk_str)
        return agent_dict
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing agent_topk: {e}")
        return {}

def calculate_averages(agent_dict):
    """
    Calculate average top1, top2, top3 across all agents.
    
    Args:
        agent_dict: Dictionary with agent names as keys and [top1, top2, top3] as values
        
    Returns:
        tuple: (avg_top1, avg_top2, avg_top3)
    """
    if not agent_dict:
        return 0.0, 0.0, 0.0
    
    top1_values = []
    top2_values = []
    top3_values = []
    
    for agent, scores in agent_dict.items():
        if len(scores) >= 3:
            top1_values.append(scores[0])
            top2_values.append(scores[1])
            top3_values.append(scores[2])
    
    if not top1_values:
        return 0.0, 0.0, 0.0
    
    avg_top1 = np.mean(top1_values)
    avg_top2 = np.mean(top2_values)
    avg_top3 = np.mean(top3_values)
    
    return avg_top1, avg_top2, avg_top3

def main():
    """Main function to process results and generate evaluation table."""
    
    # Read the CSV file
    try:
        df = pd.read_csv('results/results_new.csv')
        print(f"Loaded {len(df)} rows from results_new.csv")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Process each row and calculate averages
    results = []
    
    for idx, row in df.iterrows():
        dataset = row['input_dataset']
        method = row['method']
        agent_topk_str = row['agent_topk']
        val_avg_top1 = row['val_avg_top1']
        
        # Parse agent topk results
        agent_dict = parse_agent_topk(agent_topk_str)
        
        # Calculate averages
        avg_top1, avg_top2, avg_top3 = calculate_averages(agent_dict)
        
        # Store results
        results.append({
            'dataset': dataset,
            'method': method,
            'avg_top1': avg_top1,
            'avg_top2': avg_top2,
            'avg_top3': avg_top3,
            'val_avg_top1': val_avg_top1 if pd.notna(val_avg_top1) else 0.0,
            'row_idx': idx
        })
    
    # Convert to DataFrame for easier manipulation
    results_df = pd.DataFrame(results)
    
    # Find best configuration for each dataset-method combination
    # Group by dataset and method, then find the row with highest val_avg_top1
    best_configs = []
    
    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset]
        
        for method in dataset_results['method'].unique():
            method_results = dataset_results[dataset_results['method'] == method]
            
            # Find the configuration with highest val_avg_top1
            best_idx = method_results['val_avg_top1'].idxmax()
            best_config = method_results.loc[best_idx]
            
            best_configs.append({
                'dataset': best_config['dataset'],
                'method': best_config['method'],
                'avg_top1': best_config['avg_top1'],
                'avg_top2': best_config['avg_top2'],
                'avg_top3': best_config['avg_top3'],
                'val_avg_top1': best_config['val_avg_top1']
            })
    
    # Convert to DataFrame and sort by dataset, then by method
    best_configs_df = pd.DataFrame(best_configs)
    best_configs_df = best_configs_df.sort_values(['dataset', 'method'])
    
    # Print results table
    print("\n" + "="*100)
    print("EVALUATION RESULTS - Best Configuration per Dataset-Method Combination")
    print("="*100)
    print(f"{'Dataset':<25} {'Method':<15} {'Top1':<8} {'Top2':<8} {'Top3':<8} {'Val_Top1':<10}")
    print("-"*100)
    
    for _, row in best_configs_df.iterrows():
        print(f"{row['dataset']:<25} {row['method']:<15} "
              f"{row['avg_top1']:<8.4f} {row['avg_top2']:<8.4f} {row['avg_top3']:<8.4f} "
              f"{row['val_avg_top1']:<10.4f}")
    
    print("-"*100)
    print(f"Total configurations: {len(best_configs_df)}")
    

if __name__ == "__main__":
    main()
