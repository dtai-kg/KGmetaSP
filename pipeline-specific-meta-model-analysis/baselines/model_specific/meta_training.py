from utils.training_helpers import *
from utils.file_helpers import *
from utils.evaluation_helpers import *
from config.config import Config

from sklearn.preprocessing import StandardScaler
import argparse
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold
)
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    accuracy_score,
)


def define_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target_metric",
                    default="f1_score",
                    type=str,
                    choices=Config.TARGET_METRICS,
                    help="Target prediction metric (choose from: f1_score, accuracy, precision)")
    
    parser.add_argument("-m", "--meta_model",
                    default="RF",
                    type=str,
                    choices= Config.CLASS_META_MODEL_LABELS + Config.REG_META_MODEL_LABELS,
                    help="Meta-model to train")
    
    parser.add_argument("-e", "--meta_features",
                    default="all",
                    type=str,
                    choices = Config.METAFEATURE_SETTINGS,
                    help="Metafeature setting to fetch embedding")
    
    parser.add_argument("-n", "--n_meta_models",
                    default=1500, #880 are the runs with over 100 succesful simulations. 1036 with over 10. Less than that, results in meta-models with less training data
                    type=int,
                    help="Number of meta-models to train," \
                    " each one for a different run_id. Runs are chosen based " \
                    "on the number of successful simulations.")
    
    parser.add_argument("-sm", "--split_mode",
                    default="global",
                    type=str,
                    choices=Config.SPLIT_MODES,
                    help="Meta-training mode (choose from: local, global)" \
                    "Local: All meta-models are trained and evaluated, with each one" \
                    "having its own random train-test split." \
                    "Global: Meta-models are trained and evaluated based on a" \
                    "pre-defined global train-test split, for fair comparison with other methods.")

    return parser

def meta_train(
        training_data_df: pd.DataFrame, 
        target_metric: str, 
        meta_model_label: str, 
        split_mode: str,
        min_bin_count: int,
        n_folds: int = Config.N_FOLDS       
        ) -> Tuple:
    """
    Train a model-specific meta-model using the provided training data from benchmark.
    Parameters
    ----------
    training_data_df : pd.DataFrame
        DataFrame containing the training data.
    target_metric : str
        Target metric for the meta-model.
    meta_model : str
        Meta-model to train (e.g., "RF", "SVC", "LR").
    """

    #Get train-test split
    if split_mode == "local":
        # Local split: each meta-model has its own random train-test split
        X_train, X_test, y_train, y_test = split_to_train_test(training_data_df=training_data_df, target_metric=target_metric)
    elif split_mode == "global":
        # Global split: use the entire training_data_df as training set
        X_train = np.stack(training_data_df["embedding"].values)
        y_train = training_data_df[target_metric].values
        # For global split, test set will be fetched in evaluation from the global test set
        X_test, y_test = None, None
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if split_mode == "local" else None

    # Define meta-model and hyperparameters
    meta_model, parameter_grid = define_model_and_parameters(meta_model_label=meta_model_label)

    # Define a scoring metric and cross-val strategy
    if meta_model_label in Config.CLASS_META_MODEL_LABELS:  # Classification models
        scorer = make_scorer(accuracy_score)
        cv = StratifiedKFold(n_splits=min(n_folds, min_bin_count), shuffle=True)
    elif meta_model_label in Config.REG_META_MODEL_LABELS:  # Regression models
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        cv = KFold(n_splits=min(n_folds, min_bin_count), shuffle=True)
      

     # Set up GridSearchCV
    print("Starting GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=meta_model,
        param_grid=parameter_grid,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
        verbose=1,
    )

    # Print label statistics
    if split_mode == "local":
        display_label_statistics(y_train=y_train, y_test=y_test, meta_model_label=meta_model_label)

    # Fit the grid search with data
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    #Get cross-val dataframe
    cv_df = pd.DataFrame(grid_search.cv_results_)

    return (X_test_scaled, y_test, best_model, best_params, cv_df, scaler) 


def main():

    print("Starting meta-training...")
    parser = define_args().parse_args()

    # Parameters initialization
    target_metric = parser.target_metric
    meta_model_label = parser.meta_model
    metafeature_setting = parser.meta_features
    n_meta_models = int(parser.n_meta_models)
    target_col = target_metric if meta_model_label not in Config.CLASS_META_MODEL_LABELS else f"{target_metric}_binned"
    split_mode = parser.split_mode
    meta_training_mode = "model_specific"  # Set the meta-training modes

    # Load dataset IDs excluding benchmark regression datasets and those without embeddings
    dataset_ids_path = Config.DIR_PREFIX + "metafeatures/dataset_ids.pkl"
    dataset_ids = load(dataset_ids_path) 
    excluded_datasets = ['190', '196', '198', '204', '206', '217', '223', '227'] 
    dataset_ids = [dataset_id for dataset_id in dataset_ids if dataset_id not in excluded_datasets]

    # Fetch training data
    print(f"Fetching training data...")
    training_data_df, testing_data_df = fetch_training_data(dataset_ids, metafeature_setting, split_mode)

    # Find most frequent runs
    most_frequent_runs = training_data_df["run_id"].value_counts().head(n_meta_models).index.tolist() 
    
    # Train meta-models
    print(f"Training meta-models...")
    for run_id in most_frequent_runs:
        # Skip if the experiment is already done
        if check_if_results_directory_exists(
            target_metric=target_metric,
            metafeature_setting=metafeature_setting,
            meta_model_label=meta_model_label,
            split_mode=split_mode,
            run_id=run_id
        ):
            print(f"Results for these settings already exist. Skipping...")
            continue

        # If the split mode is global, we need to check if the run ID exists in the test set
        # else training is not meaningful since no evaluation can be done
        if split_mode == "global" and run_id not in testing_data_df["run_id"].values:
            print(f"Run ID {run_id} not found in the test data for global split mode. Skipping...")
            continue

        print(f"\tTraining model-specific meta-model for {run_id}...")

        # Filter the training data for the current run_id
        filtered_training_data_df = training_data_df.loc[training_data_df["run_id"] == run_id].copy()
        if split_mode == "global": filtered_testing_data_df = testing_data_df.loc[testing_data_df["run_id"] == run_id].copy()
        n_simulations = len(filtered_training_data_df)

        # Check if there are enough simulations to train the meta-model
        if n_simulations < Config.CLASS_BINS:
            print(f"Not enough simulations ({n_simulations}) for run ID {run_id} in global split mode. Skipping...")
            continue

        # Bin target variable if meta-model is classification-based
        if meta_model_label in Config.CLASS_META_MODEL_LABELS:
            (filtered_training_data_df, bin_edges, min_bin_count) = bin_target_variable(training_data_df=filtered_training_data_df, 
                                                                           target_metric=target_metric)
            if min_bin_count < Config.MIN_SIMS_FOR_GLOBAL_SPLIT:
                print(f"Not enough bin instances ({min_bin_count}) for run ID {run_id} in global split mode after binning. Skipping...")
                continue
            if split_mode == "global": filtered_testing_data_df = apply_bins_to_test_set(test_df=filtered_testing_data_df, 
                                                                                         target_metric=target_metric,
                                                                                         bin_edges=bin_edges)   
        else:
            min_bin_count = n_simulations

        # Train the meta-model
        (X_test, y_test, best_model, best_params, cv_df, scaler) = meta_train(training_data_df=filtered_training_data_df, 
                                                                              target_metric=target_col, 
                                                                              meta_model_label=meta_model_label, 
                                                                              split_mode=split_mode,
                                                                              min_bin_count=min_bin_count)

        #Evaluate the meta-model
        # If split mode is global, we don't get X_test and y_test from the global test set
        if split_mode == "global":
            X_test, y_test = get_global_test_set(test_df=filtered_testing_data_df,
                                                 target_metric=target_col,
                                                 scaler=scaler)

        
        test_metric1, test_metric2 = eval_meta_model(X_test=X_test, 
                                                     y_test=y_test, 
                                                     best_model=best_model, 
                                                     meta_model_label=meta_model_label)
        
        

        # Save model and results
        save_model_results(
            run_id=run_id,
            target_metric=target_metric,
            meta_model_label=meta_model_label,
            meta_model_mode=meta_training_mode,
            test_metric1=test_metric1,
            test_metric2=test_metric2,
            best_model=best_model,
            best_params=best_params,
            metafeature_setting=metafeature_setting,
            cv_df=cv_df,
            n_simulations=n_simulations,
            split_mode=split_mode
        )
         
    return

if __name__ == "__main__":
    main()