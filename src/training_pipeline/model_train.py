from functools import partial
import os
import pickle
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, make_scorer

# Configuration - should eventually move to Hydra config
SOURCE = os.path.join("data", "processed")
MODEL_PATH = "models"
N_FOLDS = 5
MAX_EVALS = 50
TARGET_COL = "Churn"

# Hyperparameter space optimized for churn prediction
SPACE = {
    "penalty": hp.choice("penalty", ["l1", "l2"]),
    "C": hp.loguniform("C", -4, 4),
    "solver": hp.choice("solver", ["saga", "liblinear"]),
    "l1_ratio": hp.uniform("l1_ratio", 0, 1)}

def encode_target(file_name_train: str, file_name_test: str, target_col: str, model_name: str, logger):
    """Load and prepare data with target encoding"""
    logger.info("Loading processed data")
    try:
        df_train = pd.read_parquet(os.path.join(SOURCE, f"{file_name_train}.parquet"))
        df_test = pd.read_parquet(os.path.join(SOURCE, f"{file_name_test}.parquet"))
        
        # Split features and target
        X_train, y_train = df_train.drop(columns=[TARGET_COL]), df_train[TARGET_COL]
        X_test, y_test = df_test.drop(columns=[TARGET_COL]), df_test[TARGET_COL]

        # Create and save label encoder
        logger.info("Encoding target variable")
        encoder = LabelEncoder()
        y_train_enc = encoder.fit_transform(y_train)
        y_test_enc = encoder.transform(y_test)
        
        # Create inverse mapping
        decoder = {i: cls for i, cls in enumerate(encoder.classes_)}
        target_translator = {"encoder": encoder, "decoder": decoder}
        
        # Save artifacts
        os.makedirs(os.path.join(MODEL_PATH, model_name), exist_ok=True)
        with open(os.path.join(MODEL_PATH, model_name, "target_encoder.pkl"), "wb") as f:
            pickle.dump(target_translator, f)
            
        return X_train, y_train_enc, X_test, y_test_enc
    
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def objective(params, X, y, n_folds=N_FOLDS):
    """Optimization objective using F1-score"""
    try:
        # Handle penalty-specific parameters
        if params["penalty"] == "elasticnet":
            params["l1_ratio"] = params.get("l1_ratio", 0.5)
        else:
            params.pop("l1_ratio", None)
            
        # Initialize model with params
        model = LogisticRegression(**params, max_iter=1000, random_state=42)
        
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
            "params": params,
            "status": STATUS_OK,
        }
        
    except Exception as e:
        return {"loss": 0, "exception": str(e)}

def train_model(X_train, y_train , model_name: str, logger):
    """Complete training workflow for churn prediction"""
    try:
       
        # Hyperparameter optimization
        logger.info("Starting Bayesian optimization")
        trials = Trials()
        
        best = fmin(
            fn=partial(objective, X=X_train, y=y_train),
            space=SPACE,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=trials,
            show_progressbar=False,
        )
        
        # Get best parameters
        best_params = trials.best_trial["result"]["params"]
        logger.info(f"Best parameters: {best_params}")
        
        # Train final model
        logger.info("Training final model")
        final_model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
        final_model.fit(X_train, y_train)
        
        os.makedirs(os.path.join(MODEL_PATH, model_name), exist_ok=True)
        with open(os.path.join(MODEL_PATH, model_name, "final_model.pkl"), "wb") as pkl:
            pickle.dump(final_model, pkl)
        logger.info("Model trained and saved successfully")
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise