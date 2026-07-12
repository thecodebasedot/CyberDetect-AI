"""Training pipeline: generate/load data, fit the detector, evaluate, persist."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATASET_PATH, LABEL_COLUMN, RANDOM_STATE
from .data_generator import generate_dataset
from .evaluate import Metrics, evaluate
from .model import IntrusionDetector


def load_or_generate_dataset(n_samples: int, attack_ratio: float) -> pd.DataFrame:
    """Load the cached dataset if present, otherwise synthesize and cache it."""
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    df = generate_dataset(n_samples=n_samples, attack_ratio=attack_ratio)
    df.to_csv(DATASET_PATH, index=False)
    return df


def train(
    n_samples: int = 10_000,
    attack_ratio: float = 0.08,
    test_size: float = 0.3,
    save: bool = True,
) -> tuple[IntrusionDetector, Metrics]:
    """Run the full training + evaluation pipeline.

    Returns the fitted detector and its metrics on the held-out test split.
    """
    df = load_or_generate_dataset(n_samples, attack_ratio)

    # Stratify on the label so both splits keep the same attack proportion.
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COLUMN] if LABEL_COLUMN in df.columns else None,
    )

    detector = IntrusionDetector().fit(train_df)

    detection = detector.score(test_df)
    y_true = test_df[LABEL_COLUMN].to_numpy()
    metrics = evaluate(y_true, detection.predictions, detection.scores)

    if save:
        detector.save()

    return detector, metrics


if __name__ == "__main__":
    _, metrics = train()
    print(metrics.pretty())
