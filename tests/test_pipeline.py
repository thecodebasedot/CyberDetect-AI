"""Unit tests for the CyberDetect AI pipeline.

Run with:  python -m pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LABEL_COLUMN, NUMERIC_FEATURES
from src.data_generator import generate_dataset
from src.evaluate import evaluate
from src.model import IntrusionDetector


def test_dataset_schema_and_ratio():
    df = generate_dataset(n_samples=2_000, attack_ratio=0.1, random_state=1)
    assert len(df) == 2_000
    for col in NUMERIC_FEATURES + [LABEL_COLUMN]:
        assert col in df.columns
    # Labels are binary.
    assert set(df[LABEL_COLUMN].unique()).issubset({0, 1})
    # Attack ratio is respected (within rounding of the 3-way family split).
    attack_frac = df[LABEL_COLUMN].mean()
    assert abs(attack_frac - 0.1) < 0.02


def test_generator_is_reproducible():
    a = generate_dataset(n_samples=500, random_state=7)
    b = generate_dataset(n_samples=500, random_state=7)
    pd.testing.assert_frame_equal(a, b)


def test_detector_fit_predict_shapes():
    df = generate_dataset(n_samples=1_500, attack_ratio=0.08, random_state=2)
    detector = IntrusionDetector().fit(df)
    detection = detector.score(df)

    assert detection.predictions.shape == (len(df),)
    assert detection.scores.shape == (len(df),)
    assert set(np.unique(detection.predictions)).issubset({0, 1})


def test_detector_unfit_raises():
    detector = IntrusionDetector()
    df = generate_dataset(n_samples=10)
    try:
        detector.predict(df)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError when scoring an unfit detector")


def test_detector_recovers_attacks_reasonably():
    """The detector should beat random on the injected attacks."""
    df = generate_dataset(n_samples=4_000, attack_ratio=0.08, random_state=3)
    detector = IntrusionDetector().fit(df)
    detection = detector.score(df)
    metrics = evaluate(df[LABEL_COLUMN].to_numpy(),
                       detection.predictions, detection.scores)
    # Separable synthetic attacks -> AUC well above chance.
    assert metrics.roc_auc > 0.8
    assert metrics.recall > 0.5


def test_save_and_load_roundtrip(tmp_path):
    df = generate_dataset(n_samples=800, random_state=4)
    detector = IntrusionDetector().fit(df)

    model_path = tmp_path / "if.joblib"
    scaler_path = tmp_path / "scaler.joblib"
    detector.save(model_path, scaler_path)

    reloaded = IntrusionDetector.load(model_path, scaler_path)
    original = detector.score(df).predictions
    restored = reloaded.score(df).predictions
    np.testing.assert_array_equal(original, restored)
