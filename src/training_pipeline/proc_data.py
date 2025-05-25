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



SOURCE = os.path.join('data', 'interim')
DESTINATION = os.path.join('data','processed')

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

# Then modify your pipeline creation:

def create_regression_cleaning_pipeline():
    return Pipeline([
        # 1. Drop columns (corrected names)
        ('drop_features', DropFeatures(
            features_to_drop=['gender', 'PhoneService', 'TotalCharges'])),

        # 2. Convert bool columns to integers
        ('bool_to_int', SklearnTransformerWrapper(
            transformer=FunctionTransformer(bool_to_int),  # Use named function
            variables=['Partner', 'Dependents', 'PaperlessBilling', 'SeniorCitizen']
        )),
        
        # 3. Normalize numerical features
        ('scaler', SklearnTransformerWrapper(
            transformer=StandardScaler(),
            variables=['tenure', 'MonthlyCharges'])),
        
        # 4. One-hot encode remaining categoricals
        ('onehot', OneHotEncoder(
            variables=['MultipleLines', 'InternetService', 'OnlineSecurity',
                      'OnlineBackup', 'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'],
            drop_last=False))
    ])
def create_tree_cleaning_pipeline(drop_additional=None):
    to_drop = ['TotalCharges']
    if drop_additional:
        to_drop.extend(drop_additional)
    
    return Pipeline([
        # Step 1: Drop specified columns
        ('drop_features', DropFeatures(features_to_drop=to_drop)),
        
        # Step 2: Convert bool columns to integers
        ('bool_to_int', SklearnTransformerWrapper(
            transformer=FunctionTransformer(bool_to_int),
            variables=['Partner', 'Dependents', 'PhoneService', 
                     'PaperlessBilling', 'SeniorCitizen']
        )),
        
        # Step 3: One-hot encode categorical features
        ('onehot', OneHotEncoder(
            variables=[
                'gender', 'MultipleLines', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'InternetService',
                'Contract', 'PaymentMethod'
            ],
            drop_last=True
        ))
    ])
  

def read_process_data(logger,
    file_name: str, target: str, IS_PARAMETRIC : bool,  num_cols: list, cat_cols: list, bool_cols: list,drop_cols: list, test_size=0.15
):
    """Data processing pipeline"""
    logger.info("Starting data processing")

    try:
        # 1. Load data
        df = pd.read_pickle(os.path.join(SOURCE, f"{file_name}.pkl"))
        logger.info(f"Raw data loaded: {df.shape}")

        # 3. Split data
        X_train, X_test, y_train, y_test = split_data_stratified(df, target, test_size)

        logger.info(f"Train/Test split: {X_train.shape}/{y_train.shape}")

        # 4. Create processing pipeline
        if(IS_PARAMETRIC):
            pipeline = create_regression_cleaning_pipeline()
        else:
            pipeline = create_tree_cleaning_pipeline()

        
        X_train_cleaned = pipeline.fit_transform(X_train)
        X_test_cleaned = pipeline.transform(X_test)
        
        logger.info("Cleaned Traing, Test Dataframes")
        
        X_train_resampled, y_train_resampled = handle_class_imbalance(X_train_cleaned, y_train)

        logger.info("Applied Upsamping with SMOTE")

        
        # 6. Combine with target (critical fix)
        df_train = X_train_resampled.copy()
        df_train['Churn'] = y_train_resampled
        df_test = X_test_cleaned.copy()
        df_test['Churn'] = y_test

        # 8. Save artifacts
        if IS_PARAMETRIC:   
            df_train.to_parquet(os.path.join(DESTINATION, f"{file_name}-train-parametric_models.parquet"))
            df_test.to_parquet(os.path.join(DESTINATION, f"{file_name}-test-parametric_models.parquet"))
            joblib.dump(pipeline, os.path.join(DESTINATION, "pipeline_parametric_models.pkl"))

        else:
            df_train.to_parquet(os.path.join(DESTINATION, f"{file_name}-train-tree_models.parquet"))
            df_test.to_parquet(os.path.join(DESTINATION, f"{file_name}-test-tree_models.parquet"))
            joblib.dump(pipeline, os.path.join(DESTINATION, "pipeline_tree_models.pkl"))

        logger.info(
            f"Processing complete. Final shapes: Train {df_train.shape}, Test {df_test.shape}"
        )
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise