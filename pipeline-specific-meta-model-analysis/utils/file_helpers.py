import pickle 
import os 
import pandas as pd
import json
import shutil
from typing import Optional

from config.config import Config

def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

def save_json(filename, data):
    """
    Save data to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    return

def save_model_results(
    run_id: int,
    target_metric: str,
    meta_model_label: str,
    meta_model_mode: str,
    test_metric1: float,
    test_metric2: float,
    best_model: object,
    best_params: dict,
    metafeature_setting: str,
    cv_df: pd.DataFrame,
    n_simulations: int,
    split_mode: str
): 
    
    """
    Save the model results to a file.
    """
    # Create the directory structure
    results_dir = Config.DIR_PREFIX + f"baselines/{meta_model_mode}/results/{split_mode}_split"
    cur_results_dir = f"{results_dir}/target_metric_{target_metric}/{metafeature_setting}_metafeatures/{meta_model_label}_metamodel/run_{run_id}"

    # Create the directory if it doesn't exist
    os.makedirs(cur_results_dir, exist_ok=True)
    
    # Save HPO results (skip for now to save some storage)
    #cv_df.to_csv(f"{cur_results_dir}/hpo_results.csv", index=False)

    #Save best model
    model_path = f"{cur_results_dir}/best_model.pkl"
    save(model_path, best_model)

    # Save evaluaition metrics
    eval_results = {
        "best_model_params": best_params,
        "test_metrics": {
            "accuracy" if meta_model_label in Config.CLASS_META_MODEL_LABELS else "mse": test_metric1,
            "weighted_f1" if meta_model_label in Config.CLASS_META_MODEL_LABELS else "r2": test_metric2,
        },
        "n_simulations": n_simulations
    }

    eval_results_path = f"{cur_results_dir}/best_model_results.json"
    save_json(eval_results_path, eval_results)

    # zip_path = Config.DIR_PREFIX + f"baselines/{meta_model_mode}/results.zip"
    # if os.path.exists(zip_path):
    #     os.remove(zip_path)
    # shutil.make_archive(results_dir, 
    #                     'zip', 
    #                     root_dir=Config.DIR_PREFIX+f"baselines/{meta_model_mode}", 
    #                     base_dir="results")
    return


def check_if_results_directory_exists(
    target_metric: str,
    metafeature_setting: str,
    meta_model_label: str,
    split_mode: str,
    run_id: Optional[int] = None
) -> bool:
    """
    Check if the results directory exists for a given run.
    """

    results_dir = Config.DIR_PREFIX + f"baselines/model_specific/results/{split_mode}_split/target_metric_{target_metric}/{metafeature_setting}_metafeatures/{meta_model_label}_metamodel"
    if run_id: results_dir = f"{results_dir}/run_{run_id}"
    
    return os.path.exists(results_dir)

def get_evaluation_dicts(
    target_metric: str,
    metafeature_setting: str,
    meta_model_label: str,
    split_mode: str
    ) -> dict:
    """
    Get the evaluation dictionaries for a given run.
    """
    evaluation_file = "best_model_results.json"
    results_dir = Config.DIR_PREFIX + f"baselines/model_specific/results/{split_mode}_split/target_metric_{target_metric}/{metafeature_setting}_metafeatures/{meta_model_label}_metamodel"
    
    # Get all the directories in the results directory
    dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    # Create a dictionary to store the evaluation results
    eval_dicts = {}
    
    # Iterate through each directory and load the evaluation results
    for d in dirs:
        run_id = int(d.split("_")[-1])
        eval_results_path = os.path.join(results_dir, d, evaluation_file)
        with open(eval_results_path, 'r') as f:
            eval_dicts[run_id] = json.load(f)
    
    return eval_dicts

def zip_results(meta_model_mode: str):
    """
    Zip the results directory.
    """
    results_dir = Config.DIR_PREFIX + f"baselines/{meta_model_mode}/results"
    zip_path = Config.DIR_PREFIX + f"baselines/{meta_model_mode}/results.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(results_dir, 
                        'zip', 
                        root_dir=Config.DIR_PREFIX+f"baselines/{meta_model_mode}", 
                        base_dir="results")
    return


