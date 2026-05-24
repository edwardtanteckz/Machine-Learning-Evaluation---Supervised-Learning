import json
import numpy as np
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.model_selection import learning_curve, StratifiedKFold, train_test_split, cross_validate
import time

from config.config import *
import model.models as model
from tuner import HyperparameterTuner

import warnings
warnings.filterwarnings('ignore')

def print_metrics(metrics, algo):
    print(f"\n{'='*60}")
    print(f"Evaluation Metrics: {algo}")
    print(f"{'='*60}")
    
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    
    if 'f1' in metrics:
        print(f"F1:        {metrics['f1']:.4f}")
    if 'macro_f1' in metrics:
        print(f"Macro-F1:  {metrics['macro_f1']:.4f}")
        print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    if 'pr_auc' in metrics:
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    
    print(f"\nTiming:")
    print(f"Fit time:     {metrics.get('fit_time', 0):.4f}s")
    print(f"Predict time: {metrics.get('predict_time', 0):.6f}s")

def calculate_metrics(y_true, y_pred, y_pred_proba=None, fit_time=0, predict_time=0, task_type=None, algo=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(
            y_true, y_pred, average='binary' if task_type == 'binary' else 'macro', zero_division = 0
        ),
        'recall': recall_score(
            y_true, y_pred, average='binary' if task_type == 'binary' else 'macro', zero_division = 0
        ),
    }

    if task_type == 'binary':
        metrics['f1'] = f1_score(y_true, y_pred, average = 'binary', zero_division = 0)
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba[:, 1])
    else:
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division = 0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division = 0)
        per_class_f1 = f1_score(y_true, y_pred, average = None, zero_division = 0)
        metrics['per_class_f1'] = per_class_f1.tolist()
    
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    metrics['fit_time'] = fit_time
    metrics['predict_time'] = predict_time

    print_metrics(metrics, algo)
    return metrics

def generate_learning_curve(dataset_name, algo, X_train, y_train, params=None):
    dataset_config = get_dataset_config(dataset_name)
    base_model = model.create_model(algo, **params)
    
    cv = StratifiedKFold(
        n_splits = LEARNING_CURVE_CONFIG['cv'], shuffle = LEARNING_CURVE_CONFIG['shuffle'],
        random_state = LEARNING_CURVE_CONFIG['random_state']
    )
    
    if dataset_config['primary_metric'] == 'f1':
        scoring = 'f1' if dataset_config['task_type'] == 'binary' else 'f1_macro'
    elif dataset_config['primary_metric'] == 'macro_f1':
        scoring = 'f1_macro'
    else:
        scoring = 'accuracy'
    
    print(f"Computing learning curve...")
    print(f"  Train sizes: {LEARNING_CURVE_CONFIG['train_sizes']}")
    print(f"  CV folds: {LEARNING_CURVE_CONFIG['cv']}")
    print(f"  Scoring: {scoring}")
    
    _, train_scores, val_scores = learning_curve(
        base_model.model, X_train, y_train,
        train_sizes = LEARNING_CURVE_CONFIG['train_sizes'],
        cv = cv, scoring = scoring, n_jobs = N_JOBS, verbose = 0
    )
    
    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)
    val_mean = np.mean(val_scores, axis = 1)
    val_std = np.std(val_scores, axis = 1)
    gap = train_mean[-1] - val_mean[-1]

    print(f"\n✓ Learning curve generated")
    print(f"\nFinal scores (max training size):")
    print(f"  Train Result: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
    print(f"  Valuation Result:   {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")
    print(f"  Gap:   {train_mean[-1] - val_mean[-1]:.4f}")
    if gap > 0.15:
        print(f"  → High variance (overfitting)")
    elif train_mean[-1] < 0.6 and val_mean[-1] < 0.6:
        print(f"  → High bias (underfitting)")
    else:
        print(f"  → Balanced")

def generate_complexity_curve(dataset_name, algo, X_train, y_train):
    dataset_config = get_dataset_config(dataset_name)
    config = get_param_grid(algo, complexity_focus=True)
    train_scores = []
    val_scores = []

    cv = StratifiedKFold(n_splits = CV_FOLDS, shuffle = True, random_state = RANDOM_SEED)
    if dataset_config['primary_metric'] == 'f1':
        scoring = 'f1' if dataset_config['task_type'] == 'binary' else 'f1_macro'
    elif dataset_config['primary_metric'] == 'macro_f1':
        scoring = 'f1_macro'
    else:
        scoring = 'accuracy'
    
    for value in config['param_values']:
        print(f"{config['param_name']} = {value}...", end=' ')
        
        model_params = config.get('fixed_params', {}).copy()
        model_params[config['param_name']] = value
        base_model = model.create_model(algo, **model_params)
        
        cv_results = cross_validate(
            base_model.model, X_train, y_train, cv=cv,
            scoring=scoring, return_train_score=True, n_jobs=N_JOBS
        )
        
        train_score = np.mean(cv_results['train_score'])
        val_score = np.mean(cv_results['test_score'])
        
        train_scores.append(train_score)
        val_scores.append(val_score)        
        print(f"Train: {train_score:.4f}, Val: {val_score:.4f}")
    
    best_idx = np.argmax(val_scores)
    best_param_value = config['param_values'][best_idx]
    best_val_score = val_scores[best_idx]
    
    print(f"\n✓ Complexity curve generated")
    print(f"\nBest {config['param_name']}: {best_param_value}")
    print(f"Best validation score: {best_val_score:.4f}")

def train_and_evaluate(dataset_name, algo, X_train, X_test, y_train, y_test, class_weights, params={}, results={}):
    dataset_config = get_dataset_config(dataset_name)
    if class_weights is not None and algo in ['dt', 'svm']:
        params['class_weight'] = class_weights
    base_model = model.create_model(algo, **params)

    start_time = time.time()
    base_model.fit(X_train, y_train)
    fit_time = time.time() - start_time

    start_time = time.time()
    y_pred = base_model.predict(X_test)
    predict_time = time.time() - start_time
    
    y_pred_proba = base_model.predict_proba(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, fit_time, predict_time, dataset_config['task_type'], algo)

    results[algo] = {'params': params, 'metrics': metrics, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba, 'best_model': base_model}
    return results

def tune_hyperparameters(dataset_name, algo, X_train, y_train, search_method='random', n_iter=30):
    print(f"\n{'='*60}")
    print(f"Tuning hyperparameters for {algo} using Search={search_method} and Interation={n_iter}.")
    print(f"{'='*60}")

    param_grid = get_param_grid(algo)
    dataset_config = get_dataset_config(dataset_name)

    tuner = HyperparameterTuner(
        algo, param_grid, X_train, y_train,
        primary_metric = dataset_config['primary_metric'],
        task_type = dataset_config['task_type']
    )
    if algo == 'svm':
        results, best_model = tuner.random_search(n_iter = n_iter, cv = 3)
    else:
        results, best_model = tuner.random_search(n_iter = n_iter, cv = CV_FOLDS)

    tuning_results_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        f'output/{dataset_name}_{algo}_tuning.json'
    )
    tuner.save_results(tuning_results_path)

    return {
        'best_params': results['best_params'], 'best_cv_score': results['best_score'],
        'model': best_model, 'tuning_time': results['search_time']
    }

def cross_validate_nnpytorch(dataset_name, X_train, y_train, hidden_sizes, learning_rate, weight_decay, 
                    batch_size, cv=CV_FOLDS):
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    val_scores = []
    dataset_config = get_dataset_config(dataset_name)
    y_train = np.array(y_train)

    for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train)):
        X_train_fold =  X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        base_model = model.NeuralNetworkPyTorch(
            input_size = X_train.shape[1], output_size = len(np.unique(y_train)),
            hidden_sizes = hidden_sizes, learning_rate = learning_rate,
            weight_decay = weight_decay, batch_size = batch_size,
            max_epochs = NN_PYTORCH_CONFIG['max_epochs'],
            early_stopping_patience = NN_PYTORCH_CONFIG['early_stopping_patience']
        )
        base_model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        y_val_pred, _ = base_model.predict(X_val_fold)
        
        if dataset_config['task_type'] == 'binary':
            score = f1_score(y_val_fold, y_val_pred, average = 'binary', zero_division = 0)
        else:
            score = f1_score(y_val_fold, y_val_pred, average = 'macro', zero_division = 0)
        
        val_scores.append(score)
        print(f"    Fold {fold + 1}/{cv}: {score:.4f}")

    mean_score = np.mean(val_scores)
    std_score = np.std(val_scores)
    return mean_score, std_score

def tune_NNpytorch_hyperparameters(dataset_name, X_train, y_train,n_configs=20):
    print(f"\n{'='*60}")
    print(f"Tuning hyperparameters for NN Pytorch using Configuration={n_configs}.")
    print(f"{'='*60}")

    np.random.seed(RANDOM_SEED)
    configs = []
    tuning_results = []
    best_val_score = -np.inf

    for i in range(n_configs):
        arch = NN_PYTORCH_CONFIG['architectures'][np.random.randint(len(NN_PYTORCH_CONFIG['architectures']))]
        config = {
            'hidden_sizes': arch['hidden_sizes'],
            'learning_rate': NN_PYTORCH_CONFIG['learning_rates'][np.random.randint(len(NN_PYTORCH_CONFIG['learning_rates']))],
            'weight_decay': NN_PYTORCH_CONFIG['weight_decay'][np.random.randint(len(NN_PYTORCH_CONFIG['weight_decay']))],
            'batch_size': NN_PYTORCH_CONFIG['batch_sizes'][np.random.randint(len(NN_PYTORCH_CONFIG['batch_sizes']))]
        }
        configs.append(config)
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{n_configs}] Testing configuration:")
        print(f"  Hidden layers: {config['hidden_sizes']}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Weight decay: {config['weight_decay']}")
        print(f"  Batch size: {config['batch_size']}")

        mean_score, std_score = cross_validate_nnpytorch(dataset_name, X_train, y_train,
            config['hidden_sizes'], config['learning_rate'],
            config['weight_decay'], config['batch_size']
        )        
        print(f"  → CV Score: {mean_score:.4f} ± {std_score:.4f}")

        result = config.copy()
        result['mean_score'] = mean_score
        result['std_score'] = std_score
        tuning_results.append(result)
        
        if mean_score > best_val_score:
            best_val_score = mean_score
            best_params = config.copy()
    
    print(f"\nBest configuration:")
    print(f"  Hidden layers: {best_params['hidden_sizes']}")
    print(f"  Learning rate: {best_params['learning_rate']}")
    print(f"  Weight decay: {best_params['weight_decay']}")
    print(f"  Batch size: {best_params['batch_size']}")
    print(f"  CV Score: {best_val_score:.4f}")
    
    return best_params, tuning_results, best_val_score

def train_best_model(X_train, y_train, best_params):
    print(f"\n{'='*60}")
    print(f"TRAINING BEST MODEL")
    print(f"{'='*60}")
    
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_SEED, stratify=y_train
    )

    best_model = model.NeuralNetworkPyTorch(
        input_size = X_train.shape[1], output_size = len(np.unique(y_train)),
        hidden_sizes = best_params['hidden_sizes'], learning_rate = best_params['learning_rate'],
        weight_decay = best_params['weight_decay'], batch_size = best_params['batch_size'],
        max_epochs = NN_PYTORCH_CONFIG['max_epochs'], early_stopping_patience = NN_PYTORCH_CONFIG['early_stopping_patience']
    )
    fit_time = best_model.fit(X_train_sub, np.array(y_train_sub), X_val_sub, np.array(y_val_sub))
    
    curves = best_model.get_training_curves()
    print(f"\nTraining info:")
    print(f"  Epochs trained: {len(curves['train_losses'])}")
    print(f"  Best epoch: {curves['best_epoch']}")
    print(f"  Final train loss: {curves['train_losses'][-1]:.4f}")
    print(f"  Final val loss: {curves['val_losses'][-1]:.4f}")
    
    return fit_time, curves, best_model

def evaluate_on_test(dataset_name, best_model, X_test_scaled, y_test):
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION FOR Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    dataset_config = get_dataset_config(dataset_name)
    y_pred, predict_time = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(
            y_test, y_pred,
            average = 'binary' if dataset_config['task_type'] == 'binary' else 'macro',
            zero_division = 0
        ),
        'recall': recall_score(
            y_test, y_pred,
            average = 'binary' if dataset_config['task_type'] == 'binary' else 'macro',
            zero_division = 0
        ),
        'predict_time': predict_time
    }
    if dataset_config['task_type'] == 'binary':
        metrics['f1'] = f1_score(y_test, y_pred, average = 'binary', zero_division = 0)
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        metrics['pr_auc'] = average_precision_score(y_test, y_pred_proba[:, 1])
    else:
        metrics['macro_f1'] = f1_score(y_test, y_pred, average = 'macro', zero_division = 0)
        metrics['weighted_f1'] = f1_score(y_test, y_pred, average = 'weighted', zero_division = 0)
        per_class_f1 = f1_score(y_test, y_pred, average = None, zero_division = 0)
        metrics['per_class_f1'] = per_class_f1.tolist()
    
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    
    if 'f1' in metrics:
        print(f"  F1:        {metrics['f1']:.4f}")
    if 'macro_f1' in metrics:
        print(f"  Macro-F1:  {metrics['macro_f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    if 'pr_auc' in metrics:
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    return metrics

def save_results(dataset_name, fit_time, curves, metrics, best_model, best_params, tuning_results, best_val_score):
    print(f"\n{'='*60}")
    print(f"SAVING RESULTS")
    print(f"{'='*60}")
    
    params_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        f'output/{dataset_name}_nn_pytorch_best_params.json'
    )
    save_params = best_params.copy()
    save_params['cv_score'] = float(best_val_score)
    
    with open(params_path, 'w') as f:
        json.dump(save_params, f, indent=2)
    print(f"✓ Best parameters saved to {params_path}")
    
    tuning_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        f'output/{dataset_name}_nn_pytorch_tuning_results.csv'
    )
    pd.DataFrame(tuning_results).to_csv(tuning_path, index=False)
    print(f"✓ Tuning results saved to {tuning_path}")
    
    curves_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        f'output/{dataset_name}_nn_pytorch_training_curves.json'
    )
    save_curves = {
        'train_losses': [float(x) for x in curves['train_losses']],
        'val_losses': [float(x) for x in curves['val_losses']],
        'train_accuracies': [float(x) for x in curves['train_accuracies']],
        'val_accuracies': [float(x) for x in curves['val_accuracies']],
        'best_epoch': int(curves['best_epoch'])
    }
    with open(curves_path, 'w') as f:
        json.dump(save_curves, f, indent=2)
    print(f"✓ Training curves saved to {curves_path}")
    
    metrics_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        f'output/{dataset_name}_nn_pytorch_test_metrics.json'
    )
    
    save_metrics = {
        k: float(v) if isinstance(v, (int, float, np.number)) else v
        for k, v in metrics.items()
        if k != 'confusion_matrix'
    }
    save_metrics['fit_time'] = float(fit_time)
    
    with open(metrics_path, 'w') as f:
        json.dump(save_metrics, f, indent = 2)
    print(f"✓ Test metrics saved to {metrics_path}")
    
    model_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        f'output/{dataset_name}_nn_pytorch_best_model.pth'
    )
    best_model.save_model(model_path)

    print(f"✓ Model saved to {model_path}")
    print(f"\n✓ All PyTorch results saved")

def execute_experiments(dataset_name, X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test, class_weights):
    
    algorithms = ['dt', 'knn','svm', 'nn_sklearn']
    results = {'dt': {}, 'knn': {}, 'svm': {}, 'nn_sklearn': {}, 'nn_pytorch': {}}
    
    for algo in algorithms:
        start_time = time.time()
        if algo == 'dt':
            X = X_train
            X_test = X_test
        else:
            X = X_train_scaled
            X_test = X_test_scaled

        if algo == 'svm':
            n_iter = 10
        else:
            n_iter = 30

        tuning_results = tune_hyperparameters(
            dataset_name, algo, X, y_train
            , search_method='random', n_iter=n_iter
        )
        results = train_and_evaluate(dataset_name, algo, X, X_test
                                        , y_train, y_test, class_weights
                                        , tuning_results['best_params'], results)
        
        print(f"\n{'='*60}")
        print(f"Generating Curves: {algo}")
        print(f"{'='*60}")
        generate_learning_curve(dataset_name, algo, X, y_train, tuning_results['best_params'])
        generate_complexity_curve(dataset_name, algo, X, y_train)

        Complete_time = time.time() - start_time
        print(f"Complete time for {algo}: {Complete_time:.2f}s")
    
    best_param, tuning_results, best_val_score = tune_NNpytorch_hyperparameters(dataset_name, X_train_scaled, y_train, n_configs=10)
    fit_time, curves, best_model = train_best_model(X_train_scaled, y_train, best_param)
    metrics = evaluate_on_test(dataset_name, best_model, X_test_scaled, y_test)
    save_results(dataset_name, fit_time, curves, metrics, best_model, best_param, tuning_results, best_val_score)
