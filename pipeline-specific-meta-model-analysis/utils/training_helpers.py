from utils.file_helpers import load
from config.config import Config

import json
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    train_test_split,
)

def fetch_training_data(
        dataset_ids: list[int], 
        metafeature_settings: str,
        split_mode: str
        ) -> pd.DataFrame: 
    """
    Fetch training data per model for a list of dataset IDs.
    
    Parameters
    ----------
    dataset_ids : list[int]
        List of OpenML dataset IDs.
    """
    #Load metafeatures
    metafeatures_path = Config.DIR_PREFIX + f"metafeatures/metafeature_store/metafeatures_{metafeature_settings}.pkl"
    metafeatures = load(metafeatures_path) 

    # Load meta-training data from benchmark
    benchmark_path = Config.DIR_PREFIX + "benchmark/pipeline_evaluations.json"
    with open(benchmark_path, 'r') as benchmark_file:
        training_data_json = json.load(benchmark_file) 

    # Asert that all dataset ids are in the benchmark
    for dataset_id in dataset_ids:
        assert str(dataset_id) in training_data_json, f"Dataset ID {dataset_id} not found in benchmark data"

    # Filter the training data to include only successful classification runs    
    training_data = []
    for dataset_id, runs in training_data_json.items():
        for run_id, run_info in runs.items():

            if (run_info.get("run_success", False) and "test_f1" in run_info["run_metrics"]):  
                training_data.append(
                    {
                        "dataset_id": int(dataset_id),
                        "run_id": int(run_id),
                        "f1_score": run_info["run_metrics"]["test_f1"],
                        "accuracy": run_info["run_metrics"]["test_accuracy"],
                        "precision": run_info["run_metrics"]["test_precision"],
                        "fit_time": run_info["run_metrics"]["fit_time"],
                        "embedding": np.array(metafeatures[int(dataset_id)]),
                    }
                )

    train_df = pd.DataFrame(training_data)

    # Keep only rows where both dataset_id and run_id appear together in global_train_df
    if split_mode == "global":
        global_train_path = Config.DIR_PREFIX + "global_split/train.csv"
        global_test_path = Config.DIR_PREFIX + "global_split/test.csv"

        # Load the global train and test sets
        global_train_df = pd.read_csv(global_train_path)
        global_test_df = pd.read_csv(global_test_path)
        
        merged_train = pd.merge(
            train_df,
            global_train_df[["dataset_id", "run_id"]],
            on=["dataset_id", "run_id"],
            how="inner"
        )
        train_set_df = merged_train

        merged_test = pd.merge(
            train_df,
            global_test_df[["dataset_id", "run_id"]],
            on=["dataset_id", "run_id"],
            how="inner"
        )
        test_set_df = merged_test

        return train_set_df, test_set_df
    
    return train_df, None

def bin_target_variable(
    training_data_df: pd.DataFrame,
    target_metric: str,
    n_bins: int = Config.CLASS_BINS,
    display_stats:bool = False
) -> Tuple:
    """
    Bin the target variable into discrete categories.
    
    Parameters
    ----------
    training_data_df : pd.DataFrame
        DataFrame containing the training data.
    target_metric : str
        Target metric to bin.
    n_bins : int
        Number of bins to create.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with the binned target variable.
    """
    
    # Bin the target variable into discrete categories
    # training_data_df[f"{target_metric}_binned"] = pd.qcut(
    #     training_data_df[target_metric], q=n_bins, labels=False, duplicates="drop"
    # )

    labels = [f"bin_{i}" for i in range(1, n_bins + 1)]
    binned_col = f"{target_metric}_binned"
    
    # Get bins from training data only
    binned, bin_edges = pd.qcut(
        training_data_df[target_metric], 
        q=n_bins, 
        labels=labels, 
        retbins=True, 
        duplicates="drop"
    )

    training_data_df[binned_col] = binned

    # Display detailed bin statistics
    
    min_bin_count = training_data_df[binned_col].value_counts().min()
    
    if display_stats:
        print("\nBinning Statistics:")
        for i in range(len(bin_edges) - 1):
            bin_data = training_data_df[
                (training_data_df[target_metric] >= bin_edges[i])
                & (training_data_df[target_metric] <= bin_edges[i + 1])
            ]
            print(f"{labels[i]}:")
            print(f"  Min: {bin_edges[i]:.4f}, Max: {bin_edges[i + 1]:.4f}")
            print(f"  Count: {len(bin_data)}")
            print(f"  Mean: {bin_data[target_metric].mean():.4f}")
            print(f"  Std Dev: {bin_data[target_metric].std():.4f}")
            print(f"  Range: {bin_edges[i + 1] - bin_edges[i]:.4f}")
    
    return (training_data_df, bin_edges, min_bin_count)

def apply_bins_to_test_set(
    test_df: pd.DataFrame,
    target_metric: str,
    bin_edges: list,
    n_bins: int = Config.CLASS_BINS
):
    """
    Apply predefined bin edges to the test set.
    """
    labels = [f"bin_{i}" for i in range(1, n_bins + 1)]
    binned_col = f"{target_metric}_binned"

    epsilon = 1  # small margin to catch values just outside
    bin_edges[0] -= epsilon
    bin_edges[-1] += epsilon

    test_df[binned_col] = pd.cut(
        test_df[target_metric],
        bins=bin_edges,
        labels=labels,
        include_lowest=True
    )
    return test_df


def split_to_train_test(
    training_data_df: pd.DataFrame,
    target_metric: str,
    test_size: float = Config.TEST_SIZE,
    random_state: int = Config.RANDOM_STATE,
) -> Tuple:
    
    """
    Split the training data into training and testing sets.
    """
    # For model-specific or dataset-specific meta-models, we don't need groups    
    X = np.stack(training_data_df["embedding"].values)
    y = training_data_df[target_metric].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train dimensions should be ((1-test_size)*n_samples, n_metafeatures)
    # Test dimensions should be (test_size*n_samples, 1)
    # print(X_train.shape, X_test.shape)
    # print(y_train.shape, y_test.shape)

    return (X_train, X_test, y_train, y_test)


def define_model_and_parameters(
    meta_model_label: str,
    ) -> Tuple:
    """
    Define the model and its parameters for meta-training.
    
    Returns
    Parameters
    ----------
    meta_model : str
        The name of the meta-model to be used (e.g., "RF", "SVC").
    -------
    Tuple
        A tuple containing the model class and its parameters.
    """
    min_samples_split = [2, 5, 10]
    # if min_bin_count >= 10:
    #     min_samples_split = [2, 5, 10]
    # elif min_bin_count >= 5 and min_bin_count < 10:
    #     min_samples_split = [2, 5]
    # elif min_bin_count < 5:
    #     min_samples_split = [2]

    # print(min_samples_split)
    
    if meta_model_label == "RF":  # Random Forest Classifier
        meta_model = RandomForestClassifier(random_state=Config.RANDOM_STATE)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": min_samples_split,
        }
    elif meta_model_label == "SVC":  # Support Vector Classifier
        meta_model = SVC()
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],
        }
    elif meta_model_label == "LR":  # Logistic Regression
        meta_model = LogisticRegression()
        param_grid = {
            "penalty": ["l2"],
            "C": [0.1, 1, 10],
            "solver": ["lbfgs"],
        }
    elif meta_model_label == "RFReg":  # Random Forest Regressor
        meta_model = RandomForestRegressor(random_state=Config.RANDOM_STATE)
        param_grid = {
            "n_estimators": [20, 50, 100, 200],
            "max_depth": [10, 20],
            "min_samples_split": min_samples_split,
        }
    elif meta_model_label == "SVR":  # Support Vector Regressor
        meta_model = SVR()
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],
        }
    elif meta_model_label == "LRReg":  # Linear Regression
        meta_model = LinearRegression()
        param_grid = {
            "fit_intercept": [True, False],
            "normalize": [True, False],
        }
    else:
        raise ValueError(f"Unknown meta-model: {meta_model_label}")
    
    return meta_model, param_grid

def check_if_run_in_test_set(
    run_id: int
) -> bool:
    """
    Check if a given run ID is in the test set.
    
    Parameters
    ----------
    run_id : int
        The run ID to check.
    
    Returns
    -------
    bool
        True if the run ID is in the test set, False otherwise.
    """
    
    # Load the test set from the global split
    test_set_path = Config.DIR_PREFIX + "global_split/test.csv"
    test_set_df = pd.read_csv(test_set_path)
    
    return run_id in test_set_df["run_id"].values
    
def display_label_statistics(y_train: np.ndarray, y_test: np.ndarray, meta_model_label: str):
    """
    Display statistics of the target variable for training and testing sets.
    
    Parameters
    ----------
    y_train : np.ndarray
        The training target variable.
    y_test : np.ndarray
        The testing target variable.
    meta_model_label : str
        The label of the meta-model used.
    """
    
    if meta_model_label in Config.CLASS_META_MODEL_LABELS:  # Classification models
        print("Train target variable value counts:")
        print(pd.Series(y_train).value_counts())
        print("Test target variable value counts:")
        print(pd.Series(y_test).value_counts())
    elif meta_model_label in Config.REG_META_MODEL_LABELS:  # Regression models
        print(f"Train target variable range: {np.min(y_train)} - {np.max(y_train)}")
        print(f"Train target variable mean: {np.mean(y_train)}")
        print(f"Train target variable std: {np.std(y_train)}")
        print(f"Train target variable variance: {np.var(y_train)}")
        print(f"Test target variable range: {np.min(y_test)} - {np.max(y_test)}")
        print(f"Test target variable mean: {np.mean(y_test)}")
        print(f"Test target variable std: {np.std(y_test)}")
        print(f"Test target variable variance: {np.var(y_test)}")

    return