import json
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix
)
from omegaconf import DictConfig

def plot_confusion_matrix(y_true, y_pred, model_name, classes, cfg: DictConfig):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(cfg.data.reports, model_name, 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()

def evaluate(X_test, y_test, model, model_name: str, encoder, logger, cfg: DictConfig):
    """Model evaluation with required metrics"""
    logger.info(f"Starting evaluation for {model_name}")
    
    try:
        # Create report directory
        report_path = os.path.join(cfg.data.reports, model_name)
        os.makedirs(report_path, exist_ok=True)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        # Generate classification report and convert to string representation
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Convert boolean keys to strings
        fixed_report = {}
        for key, value in class_report.items():
            if isinstance(key, bool):
                fixed_report[str(key)] = value
            else:
                fixed_report[key] = value
                
        metrics["classification_report"] = fixed_report

        # Get class names as strings
        class_names = [str(cls) for cls in encoder.classes_]
        
        # Confusion matrix
        plot_confusion_matrix(
            y_test, 
            y_pred, 
            model_name, 
            class_names,
            cfg
        )
        
        # Save metrics
        with open(os.path.join(report_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # Generate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {metrics["roc_auc"]:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(report_path, 'roc_curve.png'))
        plt.close()
        
        # Feature importance (if available)
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feat_imp = pd.Series(importances, index=X_test.columns)
                feat_imp.nlargest(10).plot(kind='barh')
                plt.title('Top 10 Feature Importances')
                plt.savefig(os.path.join(report_path, 'feature_importances.png'))
                plt.close()
        except Exception as e:
            logger.warning(f"Feature importance not available: {str(e)}")
        
        logger.info(f"""
        {model_name} Evaluation Results:
        - Accuracy: {metrics['accuracy']:.4f}
        - Precision: {metrics['precision']:.4f}
        - Recall: {metrics['recall']:.4f}
        - F1 Score: {metrics['f1_score']:.4f}
        - ROC AUC: {metrics['roc_auc']:.4f}
        """)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Evaluation for {model_name} failed: {str(e)}")
        raise