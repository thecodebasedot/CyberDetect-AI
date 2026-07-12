"""Evaluation utilities.

Because the synthetic dataset carries ground-truth labels, we can measure how
well the unsupervised detector recovers the injected attacks using standard
classification metrics plus ROC-AUC over the continuous anomaly score.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def as_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        return (
            "Detection performance\n"
            "---------------------\n"
            f"  Accuracy   : {self.accuracy:6.3f}\n"
            f"  Precision  : {self.precision:6.3f}\n"
            f"  Recall     : {self.recall:6.3f}\n"
            f"  F1-score   : {self.f1:6.3f}\n"
            f"  ROC-AUC    : {self.roc_auc:6.3f}\n"
            "\nConfusion matrix\n"
            "----------------\n"
            f"  TN={self.true_negatives:<6} FP={self.false_positives:<6}\n"
            f"  FN={self.false_negatives:<6} TP={self.true_positives:<6}\n"
        )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Metrics:
    """Compute detection metrics against ground-truth labels."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        # Only one class present in y_true — AUC is undefined.
        auc = float("nan")

    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=auc,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )
