
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_predictions(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'report': classification_report(y_true, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

def pretty_metrics(metrics: dict) -> str:
    return json.dumps(metrics, indent=2)
