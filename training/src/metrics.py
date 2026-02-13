"""
H5-OmniFusion Metrics Module
============================
Evaluation metrics for depression detection
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)


class MetricsTracker:
    """
    Track and compute metrics for H5-OmniFusion training.
    
    Primary metrics: F1-Score, AUC-ROC
    Secondary metrics: Accuracy, Precision, Recall, PHQ8 MAE/RMSE
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated values"""
        self.binary_preds = []
        self.binary_probs = []
        self.binary_labels = []
        self.phq8_preds = []
        self.phq8_labels = []
    
    def update(
        self,
        binary_logits: torch.Tensor,
        binary_labels: torch.Tensor,
        phq8_preds: torch.Tensor,
        phq8_labels: torch.Tensor
    ):
        """
        Update tracker with batch predictions.
        
        Args:
            binary_logits: Classification logits (batch, 2) or (batch,)
            binary_labels: True binary labels (batch,)
            phq8_preds: Predicted PHQ8 scores (batch,)
            phq8_labels: True PHQ8 scores (batch,)
        """
        if binary_logits.dim() == 2 and binary_logits.size(1) == 2:
            probs = torch.softmax(binary_logits, dim=1)[:, 1]
            preds = binary_logits.argmax(dim=1)
        else:
            probs = torch.sigmoid(binary_logits.squeeze())
            preds = (probs >= 0.5).long()
        
        self.binary_preds.extend(preds.cpu().numpy().tolist())
        self.binary_probs.extend(probs.cpu().numpy().tolist())
        self.binary_labels.extend(binary_labels.cpu().numpy().tolist())
        self.phq8_preds.extend(phq8_preds.cpu().numpy().tolist())
        self.phq8_labels.extend(phq8_labels.cpu().numpy().tolist())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dict with computed metrics
        """
        y_true = np.array(self.binary_labels)
        y_pred = np.array(self.binary_preds)
        y_prob = np.array(self.binary_probs)
        phq_true = np.array(self.phq8_labels)
        phq_pred = np.array(self.phq8_preds)
        
        metrics = {}
        
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = 0.5
        
        metrics["phq8_mae"] = mean_absolute_error(phq_true, phq_pred)
        metrics["phq8_rmse"] = np.sqrt(mean_squared_error(phq_true, phq_pred))
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        y_true = np.array(self.binary_labels)
        y_pred = np.array(self.binary_preds)
        return confusion_matrix(y_true, y_pred, labels=[0, 1])


def check_targets_achieved(metrics: Dict[str, float], targets: "TargetMetrics") -> Dict[str, bool]:
    """
    Check if target metrics are achieved.
    
    Returns:
        Dict mapping metric name to whether target is met
    """
    return {
        "f1_score": metrics.get("f1", 0) >= targets.f1_score,
        "auc_roc": metrics.get("auc_roc", 0) >= targets.auc_roc,
        "accuracy": metrics.get("accuracy", 0) >= targets.accuracy,
        "phq8_mae": metrics.get("phq8_mae", float('inf')) <= targets.phq8_mae,
        "phq8_rmse": metrics.get("phq8_rmse", float('inf')) <= targets.phq8_rmse,
    }


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics for logging"""
    lines = []
    lines.append(f"{prefix}F1: {metrics.get('f1', 0):.4f}")
    lines.append(f"{prefix}AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
    lines.append(f"{prefix}Accuracy: {metrics.get('accuracy', 0):.4f}")
    lines.append(f"{prefix}Precision: {metrics.get('precision', 0):.4f}")
    lines.append(f"{prefix}Recall: {metrics.get('recall', 0):.4f}")
    lines.append(f"{prefix}PHQ8 MAE: {metrics.get('phq8_mae', 0):.4f}")
    lines.append(f"{prefix}PHQ8 RMSE: {metrics.get('phq8_rmse', 0):.4f}")
    return " | ".join(lines)
