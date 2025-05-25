from src.logger import ExecutorLogger
from src.training_pipeline.proc_data import read_process_data
from src.training_pipeline.model_evaluate import evaluate
from src.training_pipeline.model_train import train_model, encode_target


def train_pipeline():
    logger = ExecutorLogger("train pipeline")

    # process pipeline
    bool_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'SeniorCitizen']
    category_columns = [
        'gender', 
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'InternetService', 'Contract', 'PaymentMethod'
    ]
    int_columns = ['tenure']
    float_columns = ['MonthlyCharges', 'TotalCharges']

    # read_process_data(
    #     logger,
    #     file_name="telco_customer_churn_preprocessed",
    #     target="Churn",
    #     num_cols=int_columns + float_columns,
    #     bool_cols=bool_columns,
    #     IS_PARAMETRIC=True,
    #     cat_cols=category_columns,
    #     drop_cols=None)
    
    X_train, y_train, X_test, y_test = encode_target(
        file_name_train="telco_customer_churn_preprocessed-train-parametric_models",
        file_name_test="telco_customer_churn_preprocessed-test-parametric_models",
        target_col="Churn",
        model_name="basemodel",
        logger=logger,
    )
    # train_model(X_train= X_train, y_train= y_train, model_name="basemodel", logger=logger)
    evaluate(X_test, y_test, "basemodel", logger)
    logger.info("Training Completed...")


if __name__ == "__main__":
    train_pipeline()