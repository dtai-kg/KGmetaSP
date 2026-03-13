from config.config import Config
from utils.file_helpers import check_if_results_directory_exists, get_evaluation_dicts, save_json

import itertools
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_plot(
    metrics: list,
    values: list,
    meta_model_mode: str,
    target_metric: str,
    metafeature_setting: str,
    meta_model_label: str,
    split_mode: str,
    n_simulations_threshold: int
):
    """
    Generate a plot for the given metrics.
    """
    plt.figure(figsize=(6, 4))
    bars = plt.bar(metrics, values, color=["skyblue", "lightgreen"])

    # Add text labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom')

    plt.ylim(0, 1)
    plt.ylabel("Average Evaluation Score")
    plt.title(f"Average Evaluation Comparison (Min {n_simulations_threshold} Training Sims)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # Create the directory if it doesn't exist
    analysis_dir = Config.DIR_PREFIX + f"baselines/{meta_model_mode}/analysis/{split_mode}_split/min_{n_simulations_threshold}_samples/target_metric_{target_metric}/{metafeature_setting}_metafeatures"
    os.makedirs(analysis_dir, exist_ok=True)
    
    chart_path = f"{analysis_dir}/{meta_model_label}_average_evaluation_chart.png"
    plt.savefig(chart_path)

    return

def generate_analysis(
    target_metric: str,
    metafeature_setting: str,
    meta_model_label: str,
    eval_metric1: str,
    eval_metric2: str,
    n_simulations_threshold: int,
    eval_dicts: dict,
    meta_model_mode: str,
    results_dict: dict,
    split_mode: str 
):
    """
    Generate analysis for the given parameters.
    """
    
    # Filter the evaluation dictionaries based on the number of simulations
    eval_dicts = {k: v for k, v in eval_dicts.items() if v["n_simulations"] >= n_simulations_threshold}
    #print(eval_dicts)
    
    # Perform analysis on the filtered evaluation dictionaries
    # min_eval1, min_eval2 = float("inf"), float("inf")
    # max_eval1, max_eval2 = float("-inf"), float("-inf")
    avg_eval1, avg_eval2 = 0, 0
    n_valid_evals = 0

    for eval_dict in eval_dicts:
        eval_value1 = eval_dicts[eval_dict]["test_metrics"][eval_metric1]
        eval_value2 = eval_dicts[eval_dict]["test_metrics"][eval_metric2] 

        if np.isnan(eval_value1) or np.isnan(eval_value2):
            continue
        
        avg_eval2 += eval_value2
        avg_eval1 += eval_value1
        n_valid_evals += 1

    avg_eval1 /= n_valid_evals
    avg_eval2 /= n_valid_evals
        
    generate_plot(
        metrics=[eval_metric1, eval_metric2],
        values=[avg_eval1, avg_eval2],
        meta_model_mode=meta_model_mode,
        target_metric=target_metric,
        metafeature_setting=metafeature_setting,
        meta_model_label=meta_model_label,
        split_mode=split_mode,
        n_simulations_threshold=n_simulations_threshold
    )

    if f"target_metric_{target_metric}" not in results_dict:
        results_dict[f"target_metric_{target_metric}"] = {}
    if f"metafeatures_{metafeature_setting}" not in results_dict[f"target_metric_{target_metric}"]:
        results_dict[f"target_metric_{target_metric}"][f"metafeatures_{metafeature_setting}"] = {}
    if meta_model_label not in results_dict[f"target_metric_{target_metric}"][f"metafeatures_{metafeature_setting}"]:
        results_dict[f"target_metric_{target_metric}"][f"metafeatures_{metafeature_setting}"][f"{meta_model_label}_metamodel"] = {}
    results_dict[f"target_metric_{target_metric}"][f"metafeatures_{metafeature_setting}"][f"{meta_model_label}_metamodel"] = {
        eval_metric1: avg_eval1,
        eval_metric2: avg_eval2,}
    
    
    
    return results_dict
    
    

def main():
    target_metrics = Config.TARGET_METRICS
    metafeature_settings = Config.METAFEATURE_SETTINGS
    meta_model_labels = Config.CLASS_META_MODEL_LABELS + Config.REG_META_MODEL_LABELS
    n_simulations_thresholds = [1,5,10,20,50,100] # Minimum number of simulations of a run to be considered for the evaluations
    meta_model_mode = "model_specific"  # Set the meta-training modes
    split_mode = "global"  # Set the split mode for the meta-model training

    # Generate all combinations
    print(f"Performing analysis of results for {split_mode} split mode...")
    for n_simulations_threshold in n_simulations_thresholds:
        results_dict = {"min_simulations_considered": n_simulations_threshold}
        combinations = itertools.product(target_metrics, metafeature_settings, meta_model_labels)
        print(f"\n\nStarting analysis for minimum {n_simulations_threshold} training simulations...\n\n")

        for target_metric, meta_feature, meta_model in combinations:

            print(f"\n\nAnalysing evaluations for target metric: {target_metric}, metafeature setting: {meta_feature}, meta model: {meta_model}")

            if not check_if_results_directory_exists(
                target_metric=target_metric,
                metafeature_setting=meta_feature,
                meta_model_label=meta_model,
                split_mode=split_mode
            ):  
                print(f"Evaluations not available for this combination. Skipping...")
                continue

            if meta_model in Config.CLASS_META_MODEL_LABELS: eval_metric1 = "accuracy"; eval_metric2 = "weighted_f1"
            else: eval_metric1 = "mse"; eval_metric2 = "r2"

            eval_dicts = get_evaluation_dicts(
                target_metric=target_metric,
                metafeature_setting=meta_feature,
                meta_model_label=meta_model,
                split_mode=split_mode
            )

            results_dict = generate_analysis(
                target_metric=target_metric,
                metafeature_setting=meta_feature,
                meta_model_label=meta_model,
                eval_metric1=eval_metric1,
                eval_metric2=eval_metric2,
                n_simulations_threshold=n_simulations_threshold,
                eval_dicts=eval_dicts,
                meta_model_mode=meta_model_mode,
                results_dict=results_dict,
                split_mode=split_mode
            )

        #Save results dictionary to a file if needed
        save_json(Config.DIR_PREFIX + f"baselines/{meta_model_mode}/analysis/{split_mode}_split/min_{n_simulations_threshold}_samples/analysis_results.json", results_dict)
        
    return

if __name__ == "__main__":
    
    main()
    