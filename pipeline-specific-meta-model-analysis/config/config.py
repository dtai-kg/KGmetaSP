class Config:

    IS_LOCAL = False
    N_FOLDS = 10
    RANDOM_STATE = 42
    CLASS_BINS = 3
    TEST_SIZE = 0.1
    METAFEATURE_SETTINGS = [
        "all",
        "simple",
        "statistical",
        "information_theory",
        "landmarkers",
        "mlsea",
    ]
    TARGET_METRICS = [
        "accuracy",
        "precision",
        "f1_score"
    ]
    CLASS_META_MODEL_LABELS = [
        "RF"
        #"SVC",
        #"LR",
    ]
    REG_META_MODEL_LABELS = [
        "RFReg"
        #"SVR",
        #"LRReg",
    ]

    SPLIT_MODES = [
        "global",
        "local"
    ]

    MIN_SIMS_FOR_GLOBAL_SPLIT = 2 #Chosen as the minimum number of splits in set hyper-parameters for grid search

    if IS_LOCAL:
        DIR_PREFIX = ""

    else:
        DIR_PREFIX = "/apollo/ida/Bosch/rdf2vec-meta-model/"
