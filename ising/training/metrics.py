"""
ising/training/metrics.py

Evaluation metrics for IsingCNN:
  - Phase classification: accuracy, per-class accuracy
  - Temperature regression: RMSE, MAE
  - Critical region accuracy (hardest test)
"""

import numpy as np


def compute_metrics(
    preds:   np.ndarray,
    labels:  np.ndarray,
    T_pred:  np.ndarray,
    T_true:  np.ndarray,
    Tc:      float = 2.2692,
    window:  float = 0.3,
) -> dict:
    """
    Compute classification and regression metrics.

    Parameters
    ----------
    preds   : (N,) predicted phase labels {0, 1}
    labels  : (N,) true phase labels {0, 1}
    T_pred  : (N,) predicted temperatures
    T_true  : (N,) true temperatures
    Tc      : critical temperature reference
    window  : |T - Tc| < window defines the critical region

    Returns
    -------
    dict with metric values
    """
    # Overall accuracy
    accuracy = float((preds == labels).mean())

    # Per-class accuracy
    ferro_mask = labels == 0
    para_mask  = labels == 1
    acc_ferro  = float((preds[ferro_mask] == 0).mean()) if ferro_mask.any() else float("nan")
    acc_para   = float((preds[para_mask]  == 1).mean()) if para_mask.any()  else float("nan")

    # Critical region accuracy
    crit_mask  = np.abs(T_true - Tc) < window
    acc_crit   = (
        float((preds[crit_mask] == labels[crit_mask]).mean())
        if crit_mask.any() else float("nan")
    )

    # Temperature regression
    T_rmse = float(np.sqrt(np.mean((T_pred - T_true) ** 2)))
    T_mae  = float(np.mean(np.abs(T_pred - T_true)))

    return {
        "accuracy":  accuracy,
        "acc_ferro": acc_ferro,
        "acc_para":  acc_para,
        "acc_crit":  acc_crit,
        "T_rmse":    T_rmse,
        "T_mae":     T_mae,
    }
