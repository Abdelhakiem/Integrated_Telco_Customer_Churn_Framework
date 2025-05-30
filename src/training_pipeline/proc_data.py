import os
import pandas as pd
import numpy as np 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_engine.selection import DropFeatures
from feature_engine.encoding import OneHotEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder



def split_data_stratified(df, target_col, test_size=0.15, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def handle_class_imbalance(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled



def bool_to_int(x):
    return x.astype(int)

def to_python_list(omegaconf_list):
    """Convert OmegaConf ListConfig to native Python list"""
    return [item for item in omegaconf_list] if omegaconf_list else []

def create_regression_cleaning_pipeline(cfg, logger=None):
    """Create pipeline for parametric models with dynamic column handling"""
    # Convert config to native Python lists
    parametric_drop = to_python_list(cfg.processing.parametric_drop)
    bool_columns = to_python_list(cfg.processing.bool_columns)
    category_columns = to_python_list(cfg.processing.category_columns)
    int_columns = to_python_list(cfg.processing.int_columns)
    float_columns = to_python_list(cfg.processing.float_columns)
    
    # Remove target from bool columns if present
    bool_columns = [col for col in bool_columns if col != cfg.processing.target]
    
    # Calculate columns to actually process
    bool_to_convert = [col for col in bool_columns if col not in parametric_drop]
    scaler_cols = [col for col in int_columns + float_columns if col not in parametric_drop]
    onehot_cols = [col for col in category_columns if col not in parametric_drop]
    
    if logger:
        logger.info(f"Parametric pipeline processing:")
        logger.info(f"  - Dropping: {parametric_drop}")
        logger.info(f"  - Converting bools: {bool_to_convert}")
        logger.info(f"  - Scaling: {scaler_cols}")
        logger.info(f"  - One-hot encoding: {onehot_cols}")
    
    return Pipeline([
        ('drop_features', DropFeatures(features_to_drop=parametric_drop)),
        ('bool_to_int', SklearnTransformerWrapper(
            transformer=FunctionTransformer(bool_to_int),  # Use named function
            variables=bool_to_convert
        )),
        ('scaler', SklearnTransformerWrapper(
            transformer=StandardScaler(),
            variables=scaler_cols
        )),
        ('onehot', OneHotEncoder(
            variables=onehot_cols,
            drop_last=False
        ))
    ])

def create_tree_cleaning_pipeline(cfg, logger=None):
    """Create pipeline for tree-based models with dynamic column handling"""
    # Convert config to native Python lists
    tree_drop = to_python_list(cfg.processing.tree_drop)
    additional_drop = to_python_list(cfg.processing.drop_cols) if cfg.processing.drop_cols else []
    to_drop = tree_drop + additional_drop
    
    bool_columns = to_python_list(cfg.processing.bool_columns)
    category_columns = to_python_list(cfg.processing.category_columns)
    
    # Remove target from bool columns if present
    bool_columns = [col for col in bool_columns if col != cfg.processing.target]
    
    # Calculate columns to actually process
    bool_to_convert = [col for col in bool_columns if col not in to_drop]
    onehot_cols = [col for col in category_columns if col not in to_drop]
    
    if logger:
        logger.info(f"Tree pipeline processing:")
        logger.info(f"  - Dropping: {to_drop}")
        logger.info(f"  - Converting bools: {bool_to_convert}")
        logger.info(f"  - One-hot encoding: {onehot_cols}")
    
    return Pipeline([
        ('drop_features', DropFeatures(features_to_drop=to_drop)),
        ('bool_to_int', SklearnTransformerWrapper(
            transformer=FunctionTransformer(bool_to_int),
            variables=bool_to_convert
        )),
        ('onehot', OneHotEncoder(
            variables=onehot_cols,
            drop_last=True
        ))
    ])
import pickle
def encode_target(y_train, y_test, logger, cfg: DictConfig):
    """Encode target variable and save encoder"""
    try:
        logger.info("Encoding target variable")
        encoder = LabelEncoder()
        y_train_enc = encoder.fit_transform(y_train)
        y_test_enc = encoder.transform(y_test)
        
        # Save encoder
        encoder_path = os.path.join(cfg.data.artifacts, "target_encoder.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)
            
        return y_train_enc, y_test_enc, encoder
        
    except Exception as e:
        logger.error(f"Target encoding failed: {str(e)}")
        raise

def read_process_data(logger, cfg: DictConfig, file_name: str, IS_PARAMETRIC: bool):
    """Data processing pipeline with enhanced logging and error handling"""
    logger.info("Starting data processing")
    
    try:
        # 1. Load data
        df = pd.read_pickle(os.path.join(cfg.data.source, f"{file_name}.pkl"))
        logger.info(f"Raw data loaded: {df.shape}, columns: {list(df.columns)}")
        
        # 2. Split data
        X_train, X_test, y_train, y_test = split_data_stratified(
            df, 
            cfg.processing.target, 
            test_size=cfg.processing.test_size,
            random_state=cfg.processing.random_state
        )
        logger.info(f"Train/Test split: Train={X_train.shape}, Test={X_test.shape}")

        # 3. Create processing pipeline
        if IS_PARAMETRIC:
            pipeline = create_regression_cleaning_pipeline(cfg, logger)
            suffix = cfg.data.parametric_suffix
        else:
            pipeline = create_tree_cleaning_pipeline(cfg, logger)
            suffix = cfg.data.tree_suffix
        
        # 4. Process data
        logger.info("Fitting pipeline on training data")
        X_train_cleaned = pipeline.fit_transform(X_train)
        logger.info("Transforming test data")
        X_test_cleaned = pipeline.transform(X_test)
        logger.info(f"Cleaned data shapes: Train={X_train_cleaned.shape}, Test={X_test_cleaned.shape}")
        
        # 5. Handle class imbalance
        logger.info("Applying SMOTE for class imbalance")
        X_train_resampled, y_train_resampled = handle_class_imbalance(X_train_cleaned, y_train)
        logger.info(f"Resampled data shape: {X_train_resampled.shape}")

        # 6. Save artifacts
        train_path = os.path.join(cfg.data.destination, f"{file_name}{cfg.data.train_suffix}{suffix}.parquet")
        test_path = os.path.join(cfg.data.destination, f"{file_name}{cfg.data.test_suffix}{suffix}.parquet")
        pipe_path = os.path.join(cfg.data.destination, f"pipeline{suffix}.pkl")
        
        # Combine with target
        y_train_enc, y_test_enc, _ = encode_target(y_train_resampled, y_test, logger, cfg)  # Encode target and save encoder
        df_train = X_train_resampled.copy()
        df_train [cfg.processing.target] = y_train_enc
        df_test = X_test_cleaned.copy()
        df_test [cfg.processing.target] = y_test_enc
        
        df_train.to_parquet(train_path)
        df_test.to_parquet(test_path)
        joblib.dump(pipeline, pipe_path)

        logger.info(f"Saved processed data:\n- Train: {train_path}\n- Test: {test_path}\n- Pipeline: {pipe_path}")
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise