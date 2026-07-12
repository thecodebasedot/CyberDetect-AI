"""Batch detection over a CSV of network flows using a trained detector."""

from __future__ import annotations

import pandas as pd

from .config import LABEL_COLUMN
from .model import IntrusionDetector


def detect_from_csv(csv_path: str, top: int | None = None) -> pd.DataFrame:
    """Score every flow in ``csv_path`` and return a ranked results frame.

    The returned DataFrame preserves the input columns and adds:
      * ``anomaly_score`` — higher means more anomalous
      * ``prediction``    — 1 = attack, 0 = normal
    sorted so the most suspicious flows come first.
    """
    detector = IntrusionDetector.load()
    df = pd.read_csv(csv_path)

    detection = detector.score(df)
    result = df.copy()
    result["anomaly_score"] = detection.scores
    result["prediction"] = detection.predictions

    result = result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    if top is not None:
        result = result.head(top)
    return result


def summarize(result: pd.DataFrame) -> str:
    """Human-readable one-paragraph summary of a detection result frame."""
    total = len(result)
    flagged = int(result["prediction"].sum())
    lines = [
        f"Scanned {total:,} flows — flagged {flagged:,} as suspicious "
        f"({flagged / total:.1%} of traffic)." if total else "No flows to scan.",
    ]
    if LABEL_COLUMN in result.columns and total:
        actual = int(result[LABEL_COLUMN].sum())
        lines.append(f"Ground-truth attacks present in input: {actual:,}.")
    return "\n".join(lines)
