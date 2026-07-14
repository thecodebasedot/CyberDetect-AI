"""Tests for the Customer Intelligence module (purchase / churn / CLV)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from growthmind.data import generate_users
from growthmind.models import CustomerIntelligence
from growthmind.recommend import customer_recommendations


@pytest.fixture(scope="module")
def users():
    return generate_users(n=4000, random_state=31)


def test_generator_has_churn_and_mixed_buyers(users):
    assert "churned" in users.columns
    assert set(users["churned"].unique()).issubset({0, 1})
    # Buyers should appear in more than one latent segment (propensity-based).
    buyer_segments = users.loc[users["num_orders"] > 0, "_latent_segment"].nunique()
    assert buyer_segments >= 2
    # Non-trivial purchase and churn base rates.
    assert 0.05 < (users["num_orders"] > 0).mean() < 0.8
    assert 0.05 < users["churned"].mean() < 0.8


def test_customer_models_learn(users):
    ci = CustomerIntelligence()
    metrics = ci.fit(users)
    # Realistic, better-than-chance performance (and not a suspicious 1.0).
    assert 0.7 < metrics.purchase_auc < 0.999
    assert metrics.churn_auc > 0.65
    assert metrics.clv_r2 > 0.2


def test_score_columns_and_ranges(users):
    ci = CustomerIntelligence()
    ci.fit(users)
    scored = ci.score(users)
    for col in ("purchase_prob", "churn_risk", "predicted_clv"):
        assert col in scored.columns
    assert scored["purchase_prob"].between(0, 1).all()
    assert scored["churn_risk"].between(0, 1).all()
    assert (scored["predicted_clv"] >= 0).all()


def test_summary_keys_and_values(users):
    ci = CustomerIntelligence()
    ci.fit(users)
    summary = ci.summary(users)
    for key in ("customers", "avg_churn_risk", "high_churn_customers",
                "revenue_at_risk", "conversion_opportunities", "avg_predicted_clv"):
        assert key in summary
    assert summary["customers"] == len(users)
    assert summary["revenue_at_risk"] >= 0


def test_missing_columns_raise():
    ci = CustomerIntelligence()
    with pytest.raises(ValueError):
        ci.fit(pd.DataFrame({"sessions": [1, 2, 3]}))


def test_unfit_raises(users):
    with pytest.raises(RuntimeError):
        CustomerIntelligence().score(users)


def test_save_load_roundtrip(users, tmp_path):
    ci = CustomerIntelligence()
    ci.fit(users)
    p = tmp_path / "ci.joblib"
    ci.save(p)
    restored = CustomerIntelligence.load(p)
    np.testing.assert_allclose(
        ci.score(users)["churn_risk"].to_numpy(),
        restored.score(users)["churn_risk"].to_numpy(),
    )


def test_customer_recommendations_from_summary():
    summary = {"high_churn_customers": 120, "revenue_at_risk": 5400.0,
               "conversion_opportunities": 30}
    recs = customer_recommendations(summary)
    areas = {r.area for r in recs}
    assert "Retention" in areas and "Conversion" in areas
    # Empty summary -> no recommendations.
    assert customer_recommendations({}) == []
