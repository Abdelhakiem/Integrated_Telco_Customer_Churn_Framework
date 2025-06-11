from src.logger import ExecutorLogger
from src.training_pipeline.proc_data import read_process_data
from src.training_pipeline.model_evaluate import evaluate
from src.training_pipeline.model_train import train_model
import os
import pickle
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
import pickle
from pathlib import Path
MLFLOW_TRACKING_URI = './models/mlruns'
MLFLOW_EXPERIMENT_NAME = "churn_prediction"

@hydra.main(config_path="conf", config_name="config", version_base=None)
def train_pipeline(cfg: DictConfig):
    logger = ExecutorLogger("train pipeline")
    logger.info("Starting training pipeline")
    # read_process_data(
    #     logger=logger,
    #     cfg=cfg,
    #     file_name=cfg.data.raw_file,
    #     IS_PARAMETRIC=False  
    # )
    encoder_path = os.path.join(cfg.data.artifacts, "target_encoder.pkl")
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    
    # Initialize client and experiment
    Path(MLFLOW_TRACKING_URI).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    client = MlflowClient()
    # Get or create experiment
    try:
        exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if exp is None:
            logger.info(f"Creating new experiment: {MLFLOW_EXPERIMENT_NAME}")
            exp_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
            exp = mlflow.get_experiment(exp_id)
        else:
            exp_id = exp.experiment_id
    except Exception as e:
        logger.error(f"MLflow experiment setup failed: {str(e)}")
        raise

    # Train and evaluate all models
    for model_name, model_cfg in cfg.models.items():
        logger.info(f"Starting pipeline for {model_name}")
        
        # Determine data type
        is_parametric = "LogisticRegression" in model_cfg.class_name or "SVC" in model_cfg.class_name or "GaussianNB" in model_cfg.class_name
        suffix = cfg.data.parametric_suffix if is_parametric else cfg.data.tree_suffix
        
        # Load processed data
        train_path = os.path.join(
            cfg.data.destination, 
            f"{cfg.data.raw_file}{cfg.data.train_suffix}{suffix}.parquet"
        )
        test_path = os.path.join(
            cfg.data.destination, 
            f"{cfg.data.raw_file}{cfg.data.test_suffix}{suffix}.parquet"
        )
        
        df_train = pd.read_parquet(train_path)
        df_test = pd.read_parquet(test_path)
        
        X_train = df_train.drop(columns=[cfg.processing.target])
        y_train_enc = df_train[cfg.processing.target]
        X_test = df_test.drop(columns=[cfg.processing.target])
        y_test_enc = df_test[cfg.processing.target]
                # Train model
        model = train_model(
            X_train=X_train,
            y_train=y_train_enc,
            model_cfg=model_cfg,
            model_name=model_name,
            logger=logger,
            cfg=cfg
        )
        logger.info("Evaluation started: ")
        # Evaluate model
        metrics = evaluate(
            X_test=X_test,
            y_test=y_test_enc,
            model=model,
            model_name=model_name,
            encoder=encoder,
            logger=logger,
            cfg=cfg
        )
                
        # Start MLflow run
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
            try:
                # ... [training and evaluation code] ...
                
                # Log parameters
                mlflow.log_params(model_cfg.fixed_params)
                
                # Log metrics
                flat_metrics = {k: v for k, v in metrics.items() 
                               if not isinstance(v, dict)}
                mlflow.log_metrics(flat_metrics)
                
                # Log artifacts
                model_path = os.path.join(cfg.data.artifacts, model_name)
                mlflow.log_artifacts(model_path, "model")
                
                # Log evaluation report
                report_path = os.path.join(cfg.data.reports, model_name)
                mlflow.log_artifacts(report_path, "evaluation")
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
            except Exception as e:
                logger.error(f"MLflow logging failed for {model_name}: {str(e)}")
                mlflow.end_run(status="FAILED")
                continue
                
        logger.info(f"MLflow run completed for {model_name}")
    
    logger.info("Training Completed...")
    
    
if __name__ == "__main__":
    train_pipeline()