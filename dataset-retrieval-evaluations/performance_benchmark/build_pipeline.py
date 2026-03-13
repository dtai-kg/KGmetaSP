import sys
import numpy as np

from sklearn.impute import SimpleImputer, MissingIndicator, KNNImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, OrdinalEncoder,
                                   MaxAbsScaler, MinMaxScaler, Normalizer, PolynomialFeatures,
                                   QuantileTransformer, KBinsDiscretizer, PowerTransformer,
                                   RobustScaler)
from sklearn.feature_selection import (VarianceThreshold, GenericUnivariateSelect, SelectKBest,
                                       SelectPercentile, f_classif, mutual_info_classif, chi2)
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                               ExtraTreesClassifier, BaggingClassifier, 
                               GradientBoostingClassifier, RandomForestRegressor,
                               VotingClassifier)
from sklearn.ensemble._weight_boosting import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier
from sklearn.svm import SVC, SVR, NuSVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import FactorAnalysis, FastICA, KernelPCA, PCA, TruncatedSVD
from sklearn.linear_model import (LinearRegression, LogisticRegression, Perceptron, RidgeClassifier,
                                  SGDClassifier, SGDRegressor)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing._target_encoder import TargetEncoder

from sklearn.utils import _param_validation
from sklearn_param_map import *
from sklearn import set_config


def build_pipeline (run_dict, 
                    numeric_columns, 
                    nominal_columns, 
                    X):
    
    """Function for building an sklearn pipeline given a run dictionary"""

    print("\nBuilding the pipeline...")
    pipeline_steps = []

    #Loop through every pipeline component
    for step in run_dict:
        
        component = step['component']
        params = step['params']
        substeps = step.get('steps', [])

        # Instantiate component 
        component, component_role = instantiate_component(step["component"].split('.')[-1],
                                                          params, 
                                                          substeps,
                                                          numeric_columns, 
                                                          nominal_columns, 
                                                          X)
        print(f"Sklearn component: {component}")
        if component_role is not None: pipeline_steps.append((component_role, component))
        else: pipeline_steps.append((component))

    #Create a Pipeline object
    # Check if steps are already an sklearn pipeline
    if len(pipeline_steps) == 1 and isinstance(pipeline_steps[0], Pipeline):
        pipeline = pipeline_steps[0]
    else:
        pipeline = Pipeline(pipeline_steps)
    pipeline.verbose = True

    print(f"\nConstructed pipeline: {pipeline}")


    return pipeline
    

def instantiate_component(component, 
                          params, 
                          substeps, 
                          numeric_columns, 
                          nominal_columns, 
                          X):
    """Function that instantiates sklearn components given pipeline dictionary components"""
    
    if component == 'SimpleImputer' or component == "Imputer":
        if 'verbose' in params:
            del params['verbose']
        if params["missing_values"] == "NaN":
            del params["missing_values"]
        if 'axis' in params:
            del params['axis']
        if 'categorical_features' in params:
            del params["categorical_features"] 
        if 'strategy_nominal' in params:
            del params['strategy_nominal']

        fixed_params = {Imputer_param_map.get(key, key): value for key, value in params.items()}
        return SimpleImputer(**fixed_params), "SimpleImputer"

    elif component == "ConditionalImputer" or component=="ConditionalImputer2":  
        if 'verbose' in params:
            del params['verbose']
        if params["missing_values"] == "NaN":
            del params["missing_values"]
        if 'axis' in params:
            del params['axis']
        if 'categorical_features' in params:
            del params['categorical_features']

        numeric_params = {}
        nominal_params = {}

        for key in params:
            numeric_params[key] = params[key]
            nominal_params[key] = params[key]

        if 'strategy' in numeric_params:
            del numeric_params['strategy_nominal']

        if 'strategy_nominal' in nominal_params:
            nominal_params['strategy'] = params['strategy_nominal']
            del nominal_params['strategy_nominal']

        numeric_component_params = fix_outdated_params(numeric_params, Imputer_param_map)
        nominal_component_params = fix_outdated_params(nominal_params, Imputer_param_map)

        numeric_imputer = SimpleImputer(**numeric_component_params)
        nominal_imputer = SimpleImputer(**nominal_component_params)

        column_transformer = ColumnTransformer(
            transformers=[
                ('numeric_imputer', numeric_imputer, numeric_columns),
                ('nominal_imputer', nominal_imputer, nominal_columns)
            ], 
            remainder='passthrough'
        )
        return column_transformer, 'ConditionalImputer'
    
    elif component == 'ColumnTransformer':
        transformers = []
        try: 
            column_types = params.get('column_type', [])
        except:
            column_types = None

        for idx, sub_step in enumerate(substeps):
            #print(f"Sub-step: {[sub_step]}")
            sub_pipeline = build_pipeline([sub_step], numeric_columns, nominal_columns, X)

            if column_types:

                if column_types[idx] == 'numeric' or column_types[idx] == 'num':
                    col_group = numeric_columns
                else:
                    col_group = nominal_columns

                transformers.append((column_types[idx], sub_pipeline[0], col_group))

            else:
                if isinstance(sub_pipeline[0], tuple):
                    sub_pipeline = sub_pipeline[0][1]
                else:
                    sub_pipeline = sub_pipeline[0]

                col_group = numeric_columns + nominal_columns
                if isinstance(sub_pipeline, OneHotEncoder): col_group = nominal_columns

                transformers.append((f"transformer_step_{idx}", sub_pipeline, col_group))
            
        #print(f"Transformers: {transformers}")
        return ColumnTransformer(transformers=transformers, remainder='passthrough'), component
    
    elif component == 'Pipeline':
        #print(f"Pipeline: {substeps}")
        sub_pipeline = build_pipeline(substeps, numeric_columns, nominal_columns, X)
        if isinstance(sub_pipeline, Pipeline): return sub_pipeline, None
        return Pipeline(sub_pipeline), None
    
    elif component == "VotingClassifier":
        
        estimators = []
        for step in substeps:
            sub_component_name = step['component']
            sub_params = step['params']

            # Instantiate component 
            sub_component, sub_component_role = instantiate_component(sub_component_name.split('.')[-1],
                                                          sub_params, 
                                                          None,
                                                          numeric_columns, 
                                                          nominal_columns, 
                                                          X)
            
            estimators.append((sub_component_role, sub_component))

        return VotingClassifier(estimators=estimators), component
    
    #Turn OneHotEncoder to Transformer to makes sure that only nominal columns are processed
    elif component == "OneHotEncoder" or component == "TargetEncoder": 
        component_cls = globals()[component]
        params = fix_outdated_params(params, globals().get(component+"_param_map"))
        params = validate_param_values(component_cls, params)
        transformers = [(component, component_cls(**params), nominal_columns)]
        return ColumnTransformer(transformers=transformers, remainder='passthrough'), "EncoderColumnTransformer"

    elif component == "OrdinalEncoder":
        component_cls = globals()[component]
        params = validate_param_values(component_cls, params)
        transformers = [(component, component_cls(**params), nominal_columns)]
        return ColumnTransformer(transformers=transformers, remainder='passthrough'), "EncoderColumnTransformer"

    # Update parameter keys to modern versions
    if component+"_param_map" in globals(): 
        if params:
            params = fix_outdated_params(params, globals().get(component+"_param_map"))
            #params = {globals().get(component+"param_map").get(key, key): value for key, value in params.items()}

    # Instantiate component
    try:
        component_cls = globals()[component]

        #Validate that parameter values satisfy sklearn constraints else use default values
        params = validate_param_values(component_cls, params)

        # Return estimator
        return component_cls(**params), component
    except Exception as e:
        print(f"\nNo component instantiated. Problem: {e}")
    
    
def fix_outdated_params (params, 
                         replace_dict):
    """Function that fixes outdated parameters by replacing them when possible,
    else removing them"""

    for old_key, new_key in replace_dict.items():
        if old_key in params:
            if new_key is None:
                # If new_key is None, delete the key-value pair
                del params[old_key]
            else:
                # Rename the key
                params[new_key] = params.pop(old_key)

    return params
    

def validate_param_values(estimator_class, 
                          params):
    """Parameter validation functino that validates dictionary parameter to ensure that they are 
    compatible with sklearn"""

    # Delete unwanted parameters
    if "verbose" in params: 
        del params["verbose"]

    # Extract value from nested dicts
    if "pooling_func" in params and "value" in params["pooling_func"]:
        if "mean" in params["pooling_func"]["value"]: 
            params["pooling_func"] = np.mean
        elif "amax" in params["pooling_func"]["value"]: 
            params["pooling_func"] = np.amax
        elif "median" in params["pooling_func"]["value"]: 
            params["pooling_func"] = np.median
        else: 
            sys.exit("Pooling function value not accounted for.")
    if "dtype" in params: 
        if "value" in params["dtype"]:
            if params["dtype"]["value"] == "np.float64" or params["dtype"]["value"] == "np.float64":
                params["dtype"] = np.float64
            else: 
                sys.exit("Datatype value not accounted for.")
        else:
            if params["dtype"] == "numpy.float64" or params["dtype"] == "np.float64":
                params["dtype"] = np.float64
            else: 
                sys.exit("Datatype value not accounted for.")
    if "score_func" in params and "value" in params["score_func"]:
        if "f_classif" in params["score_func"]["value"]:
            params["score_func"] = f_classif
        elif "mutual_info_classif" in params["score_func"]["value"]:
            params["score_func"] = mutual_info_classif
        elif "univariate_selection" in params["score_func"]["value"]:
            params["score_func"] = chi2
        else: 
            sys.exit("Score function value not accounted for.")
    if "missing_values" in params and params["missing_values"] == "NaN":
        del params["missing_values"]

    #Add parallel processing for all pipelines that it's possible, for faster experiments
    allowed_params = list(estimator_class().get_params().keys())
    if "n_jobs" in params and params["n_jobs"] == -1: pass
    elif "n_jobs" in allowed_params and estimator_class != LogisticRegression and (
        estimator_class != BaggingClassifier
    ):
        params["n_jobs"] = -2 #Use all CPUs but one

    print(f"\nValidating parameters of {str(estimator_class)}")
    validated_params = {}
    for key, value in params.items():
        try:
            _param_validation.validate_parameter_constraints(estimator_class._parameter_constraints, {key:value}, str(estimator_class))
            validated_params[key] = value  # If successful, keep the parameter
        except Exception as e:
            print(f"Invalid value for parameter '{key}': {value}. Removing it. ({e})")

    return validated_params
