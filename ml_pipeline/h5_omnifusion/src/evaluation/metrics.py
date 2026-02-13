
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from typing import Tuple, Dict

def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    """Find the classification threshold that maximizes F1 score."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_value = 0.0
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        if metric == "f1":
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "accuracy":
            value = accuracy_score(y_true, y_pred)
        
        if value > best_value:
            best_value = value
            best_threshold = thresh
            
    return best_threshold, best_value

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = None) -> Dict[str, float]:
    """Compute comprehensive metrics given a threshold (or optimizing it)."""
    if threshold is None:
        threshold, _ = find_optimal_threshold(y_true, y_proba, metric="f1")
        
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }
