from build_pipeline import build_pipeline
import sys
import pandas as pd 
import numpy as np
import time
import multiprocessing

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split, KFold, cross_validate
from sklearn.metrics import (accuracy_score, make_scorer, precision_score, f1_score, 
                             roc_auc_score, mean_absolute_error, r2_score, 
                             mean_squared_error)

def test_pipeline(X, 
                  y, 
                  target, 
                  run_dict, 
                  estimation_proc_dict, 
                  task_type, 
                  run_id, 
                  sim_timeout = None):
    """Function that trains and evaluates an openml pipeline to a given OpenML dataset"""

    #Instantiate estimation procedure object
    estimation_procedure_object = instantiate_estimation(estimation_proc_dict, task_type)

    #Get dataset input and target
    X, y = assess_dataset_target(X, y, target)
    #y = assess_y_type(y)
    numeric_columns = X.select_dtypes(include=['number']).columns.tolist()
    nominal_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_indices = [X.columns.get_loc(col) for col in numeric_columns]
    categorical_indices = [X.columns.get_loc(col) for col in nominal_columns]

    #Build pipeline
    pipeline = build_pipeline(run_dict, numeric_indices, categorical_indices, X)
    print(pipeline)

    scores = fit_model(estimation_proc_dict, estimation_procedure_object, task_type, pipeline, X, y, sim_timeout)
    # print(scores)

    return scores

def fit_model(estimation_procedure, 
              estimation_procedure_object, 
              task_type, 
              pipeline, 
              X, 
              y, 
              sim_timeout):
    """Function that fits the extracted sklearn model to the dataset"""

    try:
        if sim_timeout:
            scores = instantiate_evaluation_w_timeout(estimation_procedure, estimation_procedure_object,
                                                   task_type, pipeline, X, y, sim_timeout)
        else:
            scores = instantiate_evaluation(estimation_procedure, estimation_procedure_object,
                                            task_type, pipeline, X, y)
            
        if scores is not None: 
            #print("\nRun successful!")
            for key in scores:
                scores[key] = round(np.mean(scores[key]), 3)
        return scores
    
    except Exception as error:
        print(f"\nInvalid data found! Skipping simulation! Error: {error}")
        if "could not convert" in str(error) or "not supported between instances" in str(error) or "all columns should be a numeric or convertible" in str(error) or "Got 'continuous' instead" in str(error) :
            print("This data could not be modeled with this pipeline.")
        elif "test_size" in str(error) and "should be greater" in str(error):
            print("Parameters passed not sufficient to cover all classes when splitting data.")
        elif "Found unknown categories" in str(error):
            print("Train/test split not correct. Unseen categories found in the test set.")
        elif "n_splits" in str(error) and "cannot be greater than" in str(error):
            print("n_splits too big.")
        elif "Found array with 0 feature" in str(error):
            print("Invalid preprocessing methods for specified classifier")
        elif "list index out of range" in str(error):
            print("Nested Pipelines can't exist in sklearn.")
        elif "cannot be set if" in str(error):
            print("Invalid combination of hyper-parameters.")
        elif "all features must be in" in str(error):
            print("Wrong index range detected in one of the pipeline steps.")
        #sys.exit("Fit model error: " + str(error))
        return None

def instantiate_estimation(estimation_params, 
                           task_type):
    """Funciton that instantiates the estimation object using OpenML task information"""

    if "stratified_sampling" in estimation_params["parameters"] and estimation_params["parameters"]["stratified_sampling"] == "true":
        n_folds = None
        if "number_folds" in estimation_params['parameters']: n_folds = estimation_params['parameters']['number_folds']
        if n_folds is None or n_folds == "":
            cv = StratifiedKFold(shuffle=True, random_state=1)
            return cv
        else:
            cv = StratifiedKFold(n_splits=int(n_folds), shuffle=True, random_state=1)
            return cv
    else:
        n_folds = None
        if "number_folds" in estimation_params['parameters']: n_folds = estimation_params['parameters']['number_folds']
        if n_folds is None or n_folds == "":
            cv = KFold()
            return cv
        elif task_type == "regression":
            cv = KFold(n_splits=int(n_folds))
        else:
            cv = StratifiedKFold(n_splits=int(n_folds))
            return cv


def instantiate_evaluation(estimation_procedure, 
                           estimation_procedure_object, 
                           task_type, 
                           pipeline, 
                           X, 
                           y, 
                           multip_q = None):
    """Function that instantiates the evaluation object using OpenML task information"""

    if task_type == "regression":
        scoring_metrics = {
            'r2': 'r2',
            'mae': make_scorer(mean_absolute_error),
            'mse': make_scorer(mean_squared_error)
        }
        
    elif task_type == "classification":
        scoring_metrics = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, average='weighted'),
            "f1": make_scorer(f1_score, average='weighted')
            #"roc_auc": make_scorer(roc_auc_score, average="weighted", multi_class="raise")
        }


    if estimation_procedure["type"] == "crossvalidation":
        try:
            scores = cross_validate(pipeline, X, y, cv=estimation_procedure_object, scoring=scoring_metrics, error_score='raise' )
            if multip_q is not None:
                multip_q.put(scores)
            return scores
        except TypeError as type_error:
            print("Cross-val error. Simulation unsuccesful")
            if "keywords must be strings" in str(type_error):
                print("Nested Pipelines can't exist in sklearn.")
            elif "not supported between instances" in str(type_error) or "must be uniformly strings or numbers" in str(type_error) or "Cannot cast array data" in str(type_error):
                print("This data could not be modeled with this pipeline.")
            elif "a bytes-like object is required" in str(type_error):
                print("Input data too sparse for this pipeline")
            elif "can't multiply sequence by non-int" in str(type_error):
                print("Multiplication format misalignement.")

            if multip_q is not None:
                multip_q.put(None)
            return None
        except BrokenPipeError as broken_pipe_error:
            print(f"Broken pipe error occurred: {broken_pipe_error}. Skipping simulation!")
            if multip_q is not None:
                multip_q.put(None)
            return None
        except Exception as exception:
            print(f"\nInvalid data found! Skipping simulation! Error: {exception}")
            if multip_q is not None:
                multip_q.put(None)
            return None
                #sys.exit("Cross-val error: " + str(type_error))
        # except TypeError as type_error:
        #     print(f"Invalid pipeline! Check this run again! Error: {type_error}")
        #     return None
        
    elif estimation_procedure["type"] == "holdout":
        #try:
        test_size = float(estimation_procedure['parameters'].get('percentage', 33)) / 100.0
        stratify = y if estimation_procedure['parameters'].get('stratified_sampling', 'false') == 'true' else None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=42)
        # Fit the pipeline
        start_time = time.time()
        try:
            pipeline.fit(X_train, y_train)
        except Exception as error:
            print(f"\nInvalid data found! Skipping simulation! Error: {error}")
            if multip_q is not None:
                multip_q.put(None)
            return None
        fit_time = time.time() - start_time
        
        # Predict and evaluate on the test set
        y_pred = pipeline.predict(X_test)

        scores = {}
        scores["fit_time"] = fit_time
        scores["test_accuracy"] = accuracy_score(y_test, y_pred)
        scores["test_precision"] = precision_score(y_test, y_pred, average='weighted')
        scores["test_f1"] = f1_score(y_test, y_pred, average='weighted')

        if multip_q is not None:
            multip_q.put(scores)
        return scores
        
        #except TypeError as type_error:
        #print(f"Invalid pipeline! Check this run again! Error: {type_error}")
        #return None
        
    else: print("No estimation instantiated. Problem!")

def instantiate_evaluation_w_timeout(estimation_procedure, 
                                     estimation_procedure_object, 
                                     task_type,
                                     pipeline, 
                                     X, 
                                     y, 
                                     timeout):  
    """Function that instantiates the evaluation object using OpenML task information,
    given an execution timeout"""

    multip_q = multiprocessing.Queue()
    process = multiprocessing.Process(target=instantiate_evaluation, 
                                      args=(estimation_procedure, estimation_procedure_object,
                                            task_type, pipeline, X, y, multip_q))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        print("Pipeline execution timed out.")
        return None
    else:
        res = multip_q.get()
        return res

def assess_dataset_target(X, 
                          y, 
                          target):
    """Function that identifies OpenML dataset target"""

    if y:
        return X, y
    else:
        y = X[target]
        X = X.drop(columns=[target])
        return X, y
    
def assess_y_type(y): 
    """Function that identifies OpenML dataset target type"""

    try:
        # Attempt to convert `y` to numeric values
        y_numeric = pd.to_numeric(y, errors='raise')
        
        # Check if all values are integers
        if y_numeric.equals(y_numeric.astype(int)):
            # If all values are whole numbers, convert to integer type
            return y_numeric.astype(int)
        else:
            # Otherwise, keep as float
            return y_numeric.astype(float)
    
    except ValueError:
        return y
