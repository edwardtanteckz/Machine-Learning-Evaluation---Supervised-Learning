import numpy as np

# ============================================================================
# SETTINGS
# ============================================================================
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = -1
VERBOSE = 1

RANDOM_SEARCH_ITERATIONS = 50

# ============================================================================
# ADULT DATASET CONFIGURATION
# ============================================================================
ADULT_CONFIG = {
    'target': 'class', 'task_type': 'binary',
    'positive_class': '>50K',
    'metrics': ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'pr_auc'],
    'primary_metric': 'f1',
    'numeric_features': ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                         'capital-loss', 'hours-per-week'],
    'categorical_features': ['workclass', 'education', 'marital-status', 
                            'occupation', 'relationship', 'race', 'sex', 
                            'native-country'],
    'scale_features': True, 'handle_rare_categories': True, 'rare_threshold': 0.01
}

# ============================================================================
# WINE DATASET CONFIGURATION
# ============================================================================
WINE_CONFIG = {
    'target': 'quality', 'task_type': 'multiclass',
    'metrics': ['accuracy', 'macro_f1', 'weighted_f1'], 'primary_metric': 'macro_f1',
    'exclude_features': ['class'],
    'numeric_features': ['fixed_acidity', 'volatile_acidity', 'citric_acid', 
                     'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 
                     'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 
                     'alcohol', 'type'],
    'scale_features': True, 'drop_correlated': False
}

# ============================================================================
# HYPERPARAMETER SEARCH SPACES
# ============================================================================
DT_PARAM_GRID = {
    'max_depth': [3, 5, 10, 15, 20, None], 'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8], 'ccp_alpha': [0.0, 0.0001, 0.001, 0.01, 0.1],
    'criterion': ['gini', 'entropy'], 'class_weight': [None, 'balanced']
}
DT_COMPLEXITY_PARAM = {
    'param_name': 'ccp_alpha',
    'param_values': [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    'fixed_params': {'max_depth': 15, 'min_samples_leaf': 2}
}

KNN_PARAM_GRID = {
    'n_neighbors': [3, 5, 7, 11, 15, 21, 31, 51, 75, 101],
    'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan'],
    'p': [1, 2]
}
KNN_COMPLEXITY_PARAM = {
    'param_name': 'n_neighbors', 'param_values': [3, 5, 11, 21, 51, 75, 101],
    'fixed_params': {'weights': 'uniform', 'metric': 'euclidean'}
}

SVM_PARAM_GRID = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0], 'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0], 'class_weight': [None, 'balanced']
}
SVM_COMPLEXITY_PARAM = {
    'param_name': 'C', 'param_values': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    'fixed_params': {'kernel': 'rbf', 'gamma': 'scale'}
}

NN_SKLEARN_PARAM_GRID = {
    'hidden_layer_sizes': [ (50,), (100,), (200,), (50, 50), (100, 50), (100, 100)],
    'alpha': [0.0001, 0.001, 0.01, 0.1], 'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [500], 'early_stopping': [True], 'validation_fraction': [0.1],
    'n_iter_no_change': [10], 'batch_size': ['auto', 32, 64, 128]
}
NN_SKLEARN_COMPLEXITY_PARAM = {
    'param_name': 'hidden_layer_sizes',
    'param_values': [(10,), (25,), (50,), (100,), (200,), (400,)],
    'fixed_params': {'alpha': 0.001, 'learning_rate_init': 0.01, 'max_iter': 500, 'early_stopping': True}
}

NN_PYTORCH_CONFIG = {
    'architectures': [
        {'hidden_sizes': [50], 'name': 'shallow_50'},
        {'hidden_sizes': [100], 'name': 'shallow_100'},
        {'hidden_sizes': [200], 'name': 'shallow_200'},
        {'hidden_sizes': [50, 50], 'name': 'deep_50_50'},
        {'hidden_sizes': [100, 50], 'name': 'deep_100_50'},
    ],
    'learning_rates': [0.001, 0.01, 0.1], 'weight_decay': [0.0, 0.0001, 0.001, 0.01],
    'batch_sizes': [32, 64, 128], 'max_epochs': 50, 'early_stopping_patience': 20,
    'activation': 'relu'
}
NN_PYTORCH_COMPLEXITY_PARAM = {
    'param_name': 'hidden_sizes', 'param_values': [[10], [25], [50], [100], [200], [400]],
    'fixed_params': {'learning_rate': 0.01, 'weight_decay': 0.001, 'batch_size': 64}
}

# ============================================================================
# LEARNING CURVE SETTINGS
# ============================================================================
LEARNING_CURVE_CONFIG = {
    'train_sizes': np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]),
    'cv': CV_FOLDS, 'shuffle': True, 'random_state': RANDOM_SEED,
    'n_jobs': -1
}

def get_dataset_config(dataset_name):
    if dataset_name.lower() == 'adult':
        return ADULT_CONFIG
    elif dataset_name.lower() == 'wine':
        return WINE_CONFIG

def get_param_grid(algo, complexity_focus=False):
    if complexity_focus:
         grids = {
            'dt': DT_COMPLEXITY_PARAM,
            'knn': KNN_COMPLEXITY_PARAM,
            'svm': SVM_COMPLEXITY_PARAM,
            'nn_sklearn': NN_SKLEARN_COMPLEXITY_PARAM,
            'nn_pytorch': NN_PYTORCH_COMPLEXITY_PARAM,
        }
    else:
        grids = {
            'dt': DT_PARAM_GRID,
            'knn': KNN_PARAM_GRID,
            'svm': SVM_PARAM_GRID,
            'nn_sklearn': NN_SKLEARN_PARAM_GRID,
        }
    return grids.get(algo, {})
