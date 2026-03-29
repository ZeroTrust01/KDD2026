"""
Evaluation Metrics for CTR prediction.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def compute_auc(y_true, y_pred):
    """Compute Area Under ROC Curve."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(np.unique(y_true)) < 2:
        return 0.5  # undefined AUC
    return roc_auc_score(y_true, y_pred)


def compute_logloss(y_true, y_pred, eps=1e-7):
    """Compute binary cross-entropy log loss."""
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), eps, 1 - eps)
    return log_loss(y_true, y_pred)
