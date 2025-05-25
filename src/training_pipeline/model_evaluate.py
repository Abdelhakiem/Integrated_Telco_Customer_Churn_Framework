import json
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

MODEL_PATH = "models"
REPORT_PATH = "reports"
SOURCE = os.path.join("data", "processed")
TARGET_COL = "Churn"

def evaluate(X_test, y_test, model_name: str, logger):
    """Model evaluation with required metrics"""
    logger.info("Starting model evaluation")
    
    try:
        # Load model artifacts
        with open(os.path.join(MODEL_PATH, model_name, "final_model.pkl"), "rb") as f:
            model = pickle.load(f)
            
        with open(os.path.join(MODEL_PATH, model_name, "target_encoder.pkl"), "rb") as f:
            target_encoder = pickle.load(f)        
        # Encode test labels using the saved encoder
        y_test_enc = target_encoder["encoder"].transform(y_test)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test_enc, y_pred),
            "precision": precision_score(y_test_enc, y_pred),
            "recall": recall_score(y_test_enc, y_pred),
            "f1_score": f1_score(y_test_enc, y_pred),
            "roc_auc": roc_auc_score(y_test_enc, y_proba),
            "classification_report": classification_report(y_test_enc, y_pred, output_dict=True)
        }

        # Save results
        os.makedirs(os.path.join(REPORT_PATH, model_name), exist_ok=True)
        
        # Save metrics
        with open(os.path.join(REPORT_PATH, model_name, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # Generate ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test_enc, y_proba)
        plt.plot(fpr, tpr, label=f'AUC = {metrics["roc_auc"]:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(REPORT_PATH, model_name, 'roc_curve.png'))
        plt.close()

        logger.info(f"""
        Evaluation Results:
        - Accuracy: {metrics['accuracy']:.4f}
        - Precision: {metrics['precision']:.4f}
        - Recall: {metrics['recall']:.4f}
        - F1 Score: {metrics['f1_score']:.4f}
        - ROC AUC: {metrics['roc_auc']:.4f}
        """)

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise