import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import time
import json

from config.config import *
import model.models as model

class HyperparameterTuner:
    def __init__(self, algo, param_grid, X_train, y_train, primary_metric='f1', task_type='binary', n_jobs=-1):
        self.algorithm = algo
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.primary_metric = primary_metric
        self.task_type = task_type
        self.n_jobs = n_jobs
        
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.cv_results = None
        
    def _get_score(self):
        if (self.primary_metric == 'f1') and (self.task_type == 'binary'):
            return make_scorer(f1_score, average='binary')
        elif self.primary_metric == 'macro_f1':
            return make_scorer(f1_score, average='macro', zero_division=0)
        elif self.primary_metric == 'accuracy':
            return make_scorer(accuracy_score)
    
    def random_search(self, n_iter=RANDOM_SEARCH_ITERATIONS, cv=CV_FOLDS, verbose=1):
        print(f"Parameter samples: {n_iter}")
        print(f"CV folds: {cv}")
        print(f"Primary metric: {self.primary_metric}")
        
        base_model = model.create_model(self.algorithm)
        scorer = self._get_score()
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        
        random_search = RandomizedSearchCV(
            estimator = base_model.model, param_distributions = self.param_grid,
            n_iter = n_iter, scoring = scorer, cv = cv_strategy, n_jobs = self.n_jobs,
            verbose = verbose, random_state = RANDOM_SEED, return_train_score = True
            , error_score='raise'
        )
        start_time = time.time()
        random_search.fit(self.X_train, self.y_train)
        search_time = time.time() - start_time
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.cv_results = pd.DataFrame(random_search.cv_results_)
        self.best_model = model.create_model(self.algorithm, **self.best_params)
        
        print(f"\n✓ Random search completed in {search_time:.2f}s")
        print(f"\nBest parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV score: {self.best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'search_time': search_time,
            'cv_results': self.cv_results
        }, self.best_model
    
    def save_results(self, filepath):
        result = {
            'algorithm': self.algorithm, 'best_params': self.best_params,
            'best_score': float(self.best_score) if self.best_score else None,
            'primary_metric': self.primary_metric, 'task_type': self.task_type
        }
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        if self.cv_results is not None:
            cv_filepath = filepath.replace('.json', '_cv_results.csv')
            self.cv_results.to_csv(cv_filepath, index=False)
            print(f"✓ CV results saved to {cv_filepath}")
