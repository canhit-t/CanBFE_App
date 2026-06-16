"""Regression metrics for binding-affinity prediction.

All functions accept plain Python lists or numpy arrays; they also
accept torch tensors (converted internally).

Metrics
-------
rmse       root-mean-square error  (kcal mol⁻¹)
mae        mean absolute error     (kcal mol⁻¹)
pearson_r  Pearson correlation coefficient
spearman_r Spearman rank correlation coefficient

``compute_metrics(y_true, y_pred)`` returns all four as a dict.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _to_array(x) -> np.ndarray:
    """Convert list / torch.Tensor / numpy array to 1-D float64 array."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float64).ravel()


def rmse(y_true: Sequence, y_pred: Sequence) -> float:
    """Root-mean-square error."""
    t, p = _to_array(y_true), _to_array(y_pred)
    return float(math.sqrt(np.mean((t - p) ** 2)))


def mae(y_true: Sequence, y_pred: Sequence) -> float:
    """Mean absolute error."""
    t, p = _to_array(y_true), _to_array(y_pred)
    return float(np.mean(np.abs(t - p)))


def pearson_r(y_true: Sequence, y_pred: Sequence) -> float:
    """Pearson correlation coefficient.  Returns 0.0 if std == 0."""
    t, p = _to_array(y_true), _to_array(y_pred)
    if t.std() == 0 or p.std() == 0:
        return 0.0
    return float(np.corrcoef(t, p)[0, 1])


def spearman_r(y_true: Sequence, y_pred: Sequence) -> float:
    """Spearman rank correlation coefficient.  Returns 0.0 if std == 0."""
    from scipy.stats import spearmanr
    t, p = _to_array(y_true), _to_array(y_pred)
    if len(t) < 3:
        return 0.0
    corr, _ = spearmanr(t, p)
    return float(corr) if not math.isnan(corr) else 0.0


def compute_metrics(
    y_true: Sequence,
    y_pred: Sequence,
) -> dict[str, float]:
    """Return all four metrics as a dict with float values."""
    return {
        "rmse":      rmse(y_true, y_pred),
        "mae":       mae(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
        "spearman_r": spearman_r(y_true, y_pred),
    }
