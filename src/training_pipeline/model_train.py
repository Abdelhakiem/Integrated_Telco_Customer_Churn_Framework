
from functools import partial
import os
import pickle
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from omegaconf import DictConfig
import hydra
import importlib
from sklearn.model_selection import cross_validate
import json


def get_model_class(class_path: str):
    """Dynamically import model class from string"""
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def build_hyperopt_space(space_config: dict):
    """Convert YAML space config to Hyperopt space"""
    space = {}
    for param, config in space_config.items():
        if config['type'] == 'choice':
            space[param] = hp.choice(param, config['options'])
        elif config['type'] == 'loguniform':
            space[param] = hp.loguniform(param, config['low'], config['high'])
        elif config['type'] == 'uniform':
            space[param] = hp.uniform(param, config['low'], config['high'])
        elif config['type'] == 'quniform':
            space[param] = scope.int(hp.quniform(param, config['low'], config['high'], config['q']))
    return space

def objective(params, model_class, fixed_params, X, y, n_folds, logger):
    """Optimization objective using F1-score"""
    try:
        # Combine fixed and tuned parameters
        all_params = {**fixed_params, **params}
        
        # Handle special cases
        if 'penalty' in all_params and all_params['penalty'] == 'none':
            all_params.pop('l1_ratio', None)
            
        # Initialize model
        model = model_class(**all_params)
        
        # Custom scorer for churn prediction
        f1_scorer = make_scorer(f1_score, average='macro')
        
        scores = cross_validate(
            model,
            X,
            y,
            cv=n_folds,
            scoring=f1_scorer,
            n_jobs=-1,
            error_score="raise",
        )
        
        return {
            "loss": -np.mean(scores["test_score"]),  # Minimize negative F1
            "params": all_params,
            "status": STATUS_OK,
        }
        
    except Exception as e:
        logger.error(f"Objective function failed: {str(e)}")
        return {"loss": 0, "exception": str(e), "status": STATUS_FAIL}

def train_model(X_train, y_train, model_cfg: DictConfig, model_name: str, logger, cfg: DictConfig):
    """Complete training workflow for churn prediction"""
    try:
        # Get model class
        model_class = get_model_class(model_cfg.class_name)
        
        # Build hyperopt space
        space = build_hyperopt_space(model_cfg.space)
        
        # Hyperparameter optimization
        logger.info(f"Starting Bayesian optimization for {model_name}")
        trials = Trials()
        
        best = fmin(
            fn=partial(
                objective, 
                model_class=model_class,
                fixed_params=model_cfg.fixed_params,
                X=X_train, 
                y=y_train,
                n_folds=model_cfg.n_folds,
                logger=logger
            ),
            space=space,
            algo=tpe.suggest,
            max_evals=model_cfg.max_evals,
            trials=trials,
            show_progressbar=False,
        )
        
        # Get best parameters
        best_params = trials.best_trial["result"]["params"]
        logger.info(f"Best parameters for {model_name}: {best_params}")
        
        # Train final model
        logger.info(f"Training final {model_name} model")
        final_model = model_class(**best_params)
        final_model.fit(X_train, y_train)
        
        # Save model
        model_path = os.path.join(cfg.data.artifacts, model_name)
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, "model.pkl"), "wb") as pkl:
            pickle.dump(final_model, pkl)
            
        # Save best parameters
        with open(os.path.join(model_path, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)
            
        logger.info(f"{model_name} model saved successfully")
        return final_model
        
    except Exception as e:
        logger.error(f"{model_name} training failed: {str(e)}")
        raise
