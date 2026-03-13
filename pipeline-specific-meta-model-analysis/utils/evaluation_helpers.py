import numpy as np
from sklearn.model_selection import GridSearchCV
from typing import Tuple
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_squared_error,
    r2_score,
    accuracy_score,
)
import pandas as pd

from config.config import Config

def eval_meta_model(
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        best_model: object,
        meta_model_label: str
        ) -> Tuple[float, float]:
    """
    Evaluate the meta-model using the test set and return the evaluation metrics.
    
    Parameters
    ----------
    X_test : np.ndarray
        The test set features.
    y_test : np.ndarray
        The true labels for the test set.
    grid_search : GridSearchCV
        The grid search object used for hyperparameter tuning.
    best_model : object
        The best model obtained from the grid search.
    
    Returns
    -------
    Tuple[float, float]
        A tuple containing the test metric and the best test metric from the grid search.
    """
    
    # Make predictions on the test set
    print("Evaluating meta-model on test data...")
    y_pred = best_model.predict(X_test)
    
    # Calculate evaluation metrics
    if meta_model_label in Config.CLASS_META_MODEL_LABELS:  # Classification models
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        test_metric1 = accuracy_score(y_test, y_pred) #Accuracy
        test_metric2 = f1_score(y_test, y_pred, average="weighted") #F1 score
        print(f"Test Accuracy: {test_metric1:.4f}")
    elif meta_model_label in Config.REG_META_MODEL_LABELS:  # Regression models
        test_metric1 = mean_squared_error(y_test, y_pred) #MSE
        test_metric2 = r2_score(y_test, y_pred) #R2
        print("\nResults:")
        print(f"MSE = {test_metric1:.4f}, R2 = {test_metric2:.4f}")
    
    return (test_metric1, test_metric2)

def get_global_test_set(
        test_df: pd.DataFrame,
        target_metric:str,
        scaler: object
) -> Tuple:
    """
    Get the global test set for a given run ID.
    
    Parameters
    ----------
    run_id : int
        The run ID to get the test set for.
    
    Returns
    -------
    Tuple
        The global X_test and y_test for the given run ID.
    """ 
    X_test = np.stack(test_df["embedding"].values)
    X_test_scaled = scaler.transform(X_test)
    y_test = test_df[target_metric].values

    return (X_test_scaled, y_test)