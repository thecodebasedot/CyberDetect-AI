"""Customer Intelligence — purchase propensity, churn, and lifetime value.

Segmentation (``segmentation.py``) tells you *who* your visitors are today. This
module predicts what they'll *do*:

* **Purchase propensity** — probability a visitor converts, from behaviour alone
  (no purchase columns as inputs — that would be leakage). Classifier → ROC-AUC.
* **Churn** — probability an existing customer lapses, using the ground-truth
  ``churned`` label from the data generator. Classifier → ROC-AUC.
* **Customer Lifetime Value (CLV)** — expected monetary value predicted from
  engagement behaviour. Regressor → R².

Together these drive retention and targeting decisions: whom to nurture toward a
first purchase, whom to win back before they churn, and where the high-value
customers are. ``score`` returns all three per user; ``summary`` rolls them up
into revenue-at-risk and opportunity numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from ..config import MODELS_DIR, RANDOM_STATE

CUSTOMER_MODEL = MODELS_DIR / "customer_intelligence.joblib"

# Behaviour-only features (deliberately exclude num_orders / total_spent so the
# purchase & CLV models learn from behaviour, not from the outcome itself).
BEHAVIOUR_FEATURES = ["sessions", "pageviews", "avg_time_on_site", "recency_days"]

_XGB_CLF = {
    "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.9, "colsample_bytree": 0.9,
    "random_state": RANDOM_STATE, "n_jobs": -1, "eval_metric": "logloss",
}
_XGB_REG = {
    "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.9, "colsample_bytree": 0.9,
    "random_state": RANDOM_STATE, "n_jobs": -1,
}


@dataclass
class CustomerMetrics:
    purchase_auc: float
    churn_auc: float
    clv_r2: float

    def pretty(self) -> str:
        return ("Customer intelligence — "
                f"purchase AUC={self.purchase_auc:.3f}  "
                f"churn AUC={self.churn_auc:.3f}  "
                f"CLV R²={self.clv_r2:.3f}")


class CustomerIntelligence:
    """Three-in-one customer predictive suite (purchase / churn / CLV)."""

    def __init__(self):
        self.purchase_model: XGBClassifier | None = None
        self.churn_model: XGBClassifier | None = None
        self.clv_model: XGBRegressor | None = None
        self.features = BEHAVIOUR_FEATURES

    # -- training ----------------------------------------------------------
    def fit(self, users: pd.DataFrame, test_size: float = 0.25) -> CustomerMetrics:
        self._check_columns(users)
        X = users[self.features]

        y_purchase = (users["num_orders"] > 0).astype(int)
        y_churn = users["churned"].astype(int)
        y_clv = users["total_spent"].astype(float)

        Xtr, Xte, ptr, pte, ctr, cte, vtr, vte = train_test_split(
            X, y_purchase, y_churn, y_clv,
            test_size=test_size, random_state=RANDOM_STATE, stratify=y_purchase,
        )

        self.purchase_model = XGBClassifier(**_XGB_CLF).fit(Xtr, ptr)
        self.churn_model = XGBClassifier(**_XGB_CLF).fit(Xtr, ctr)
        self.clv_model = XGBRegressor(**_XGB_REG).fit(Xtr, vtr)

        return CustomerMetrics(
            purchase_auc=_safe_auc(pte, self.purchase_model.predict_proba(Xte)[:, 1]),
            churn_auc=_safe_auc(cte, self.churn_model.predict_proba(Xte)[:, 1]),
            clv_r2=float(r2_score(vte, self.clv_model.predict(Xte))),
        )

    # -- inference ---------------------------------------------------------
    def score(self, users: pd.DataFrame) -> pd.DataFrame:
        self._check_ready()
        X = users[self.features]
        out = users.copy()
        out["purchase_prob"] = self.purchase_model.predict_proba(X)[:, 1].round(4)
        out["churn_risk"] = self.churn_model.predict_proba(X)[:, 1].round(4)
        out["predicted_clv"] = np.clip(self.clv_model.predict(X), 0, None).round(2)
        return out

    def summary(self, users: pd.DataFrame) -> dict:
        """Roll per-user predictions up into headline retention/opportunity KPIs."""
        scored = self.score(users)
        high_churn = scored[scored["churn_risk"] >= 0.5]
        # Revenue at risk: expected CLV weighted by churn probability.
        revenue_at_risk = float((scored["churn_risk"] * scored["predicted_clv"]).sum())
        # Non-buyers with high purchase propensity = conversion opportunities.
        non_buyers = scored[scored["num_orders"] == 0]
        opportunities = int((non_buyers["purchase_prob"] >= 0.5).sum())
        return {
            "customers": int(len(scored)),
            "avg_purchase_prob": round(float(scored["purchase_prob"].mean()), 4),
            "avg_churn_risk": round(float(scored["churn_risk"].mean()), 4),
            "high_churn_customers": int(len(high_churn)),
            "revenue_at_risk": round(revenue_at_risk, 2),
            "avg_predicted_clv": round(float(scored["predicted_clv"].mean()), 2),
            "conversion_opportunities": opportunities,
        }

    def feature_importance(self) -> pd.DataFrame:
        self._check_ready()
        return pd.DataFrame({
            "feature": self.features,
            "purchase": self.purchase_model.feature_importances_,
            "churn": self.churn_model.feature_importances_,
            "clv": self.clv_model.feature_importances_,
        })

    # -- persistence -------------------------------------------------------
    def save(self, path=CUSTOMER_MODEL) -> None:
        self._check_ready()
        joblib.dump({
            "purchase": self.purchase_model,
            "churn": self.churn_model,
            "clv": self.clv_model,
            "features": self.features,
        }, path)

    @classmethod
    def load(cls, path=CUSTOMER_MODEL) -> "CustomerIntelligence":
        obj = cls()
        blob = joblib.load(path)
        obj.purchase_model = blob["purchase"]
        obj.churn_model = blob["churn"]
        obj.clv_model = blob["clv"]
        obj.features = blob["features"]
        return obj

    # -- internals ---------------------------------------------------------
    def _check_columns(self, users: pd.DataFrame) -> None:
        required = set(self.features) | {"num_orders", "total_spent", "churned"}
        missing = required - set(users.columns)
        if missing:
            raise ValueError(f"users is missing required columns: {sorted(missing)}")

    def _check_ready(self) -> None:
        if self.purchase_model is None or self.churn_model is None or self.clv_model is None:
            raise RuntimeError("CustomerIntelligence is not trained. Call fit() or load().")


def _safe_auc(y_true, y_score) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")
