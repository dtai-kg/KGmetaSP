#Mapping of older sklearn component parameter to modern ones or None if they are deprecated

OneHotEncoder_param_map = {
    'sparse': 'sparse_output',
    'n_values': None,
    'categorical_features': None
}

Imputer_param_map = {
    'fill_empty': 'fill_value',
}

BaggingClassifier_param_map = {
    'base_estimator': 'estimator'
}

AdaBoostClassifier_param_map = {
    'base_estimator': 'estimator'
}

MLPClassifier_param_map = {
    'algorithm': 'solver'
}

FeatureAgglomeration_param_map = {
    'affinity': 'metric'
}

SGDClassifier_param_map = {
    'n_iter': 'n_iter_no_change'
}

SVC_param_map = {
    'options': None,
    'params': None
}

SVR_param_map = {
    'options': None,
    'params': None
}

LinearRegression_param_map = {
    'normalize': None
}

TargetEncoder_param_map = {
    'drop_invariant': None,
    'cols': None,
    'smoothing': 'smooth',
    'handle_unknown': None,
    'impute_missing': None,
    'min_samples_leaf': None,
    'return_df': None
}
