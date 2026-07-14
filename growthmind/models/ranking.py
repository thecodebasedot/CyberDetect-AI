"""Keyword Ranking — LightGBM learning-to-rank (LambdaMART).

Predicts how a page will rank for a keyword from on-page + off-page signals,
using LightGBM's ``LGBMRanker`` (lambdarank objective). Unlike a plain
regressor, a ranker optimizes the *order* of candidate pages within each
keyword and is evaluated with **NDCG** — the standard ranking metric.

Beyond raw ranking, ``opportunities`` surfaces *striking-distance* keywords —
where our page sits just off page one and a small push (backlinks, relevance,
title) could win a top spot — prioritized by search volume. That's the
practical, revenue-oriented output an SEO team acts on.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score

from ..config import RANDOM_STATE, RANKING_FEATURES, RANKING_MODEL

GROUP_COL = "keyword_id"
LABEL_COL = "relevance_label"

_LGBM_PARAMS = {
    "objective": "lambdarank",
    "n_estimators": 300,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}


@dataclass
class RankingMetrics:
    ndcg_at_10: float
    ndcg_at_5: float
    n_test_keywords: int

    def pretty(self) -> str:
        return (f"Keyword ranker — NDCG@10={self.ndcg_at_10:.3f}  "
                f"NDCG@5={self.ndcg_at_5:.3f}  "
                f"({self.n_test_keywords} test keywords)")


def _group_sizes(df: pd.DataFrame) -> list[int]:
    """Contiguous group sizes per keyword (order preserved)."""
    return df.groupby(GROUP_COL, sort=False).size().tolist()


class KeywordRankingModel:
    """LightGBM LambdaMART ranker over keyword→page candidates."""

    def __init__(self, params: dict | None = None):
        self.params = params or dict(_LGBM_PARAMS)
        self.model: LGBMRanker | None = None
        self.features = RANKING_FEATURES

    # -- training ----------------------------------------------------------
    def fit(self, keywords: pd.DataFrame, test_fraction: float = 0.2) -> RankingMetrics:
        """Fit the ranker; evaluate NDCG on a held-out set of *keywords*.

        The split is by keyword group (not row) so no keyword appears in both
        train and test — the honest way to evaluate a ranker.
        """
        df = keywords.sort_values(GROUP_COL).reset_index(drop=True)
        kids = df[GROUP_COL].unique()
        rng = np.random.default_rng(RANDOM_STATE)
        rng.shuffle(kids)
        n_test = max(1, int(len(kids) * test_fraction))
        test_ids = set(kids[:n_test].tolist())

        train = df[~df[GROUP_COL].isin(test_ids)].reset_index(drop=True)
        test = df[df[GROUP_COL].isin(test_ids)].reset_index(drop=True)

        self.model = LGBMRanker(**self.params)
        self.model.fit(
            train[self.features], train[LABEL_COL],
            group=_group_sizes(train),
        )

        ndcg10, ndcg5 = self._evaluate(test)
        return RankingMetrics(ndcg_at_10=ndcg10, ndcg_at_5=ndcg5,
                              n_test_keywords=len(test_ids))

    def _evaluate(self, test: pd.DataFrame) -> tuple[float, float]:
        n10, n5 = [], []
        for _, grp in test.groupby(GROUP_COL, sort=False):
            if len(grp) < 2:
                continue
            true = grp[LABEL_COL].to_numpy().reshape(1, -1)
            pred = self.model.predict(grp[self.features]).reshape(1, -1)
            n10.append(ndcg_score(true, pred, k=10))
            n5.append(ndcg_score(true, pred, k=5))
        return (float(np.mean(n10)) if n10 else float("nan"),
                float(np.mean(n5)) if n5 else float("nan"))

    # -- inference ---------------------------------------------------------
    def predict_ranking(self, keywords: pd.DataFrame) -> pd.DataFrame:
        """Add a model ``rank_score`` and predicted ``predicted_position`` per
        keyword (1 = best predicted)."""
        self._check_ready()
        df = keywords.copy()
        df["rank_score"] = self.model.predict(df[self.features])
        df["predicted_position"] = (
            df.groupby(GROUP_COL)["rank_score"]
            .rank(ascending=False, method="first").astype(int)
        )
        return df

    def opportunities(self, keywords: pd.DataFrame, top: int = 15) -> pd.DataFrame:
        """Striking-distance wins: *our* pages predicted just off page one.

        Returns our keywords whose predicted position is 4–15 (close but not
        top), ranked by search volume — the highest-ROI SEO targets.
        """
        ranked = self.predict_ranking(keywords)
        ours = ranked[ranked["is_ours"] == 1]
        striking = ours[(ours["predicted_position"] >= 4) &
                        (ours["predicted_position"] <= 15)]
        cols = ["keyword", "search_volume", "keyword_difficulty",
                "predicted_position", "relevance", "backlinks", "domain_authority"]
        return (striking.sort_values("search_volume", ascending=False)
                .head(top)[cols].reset_index(drop=True))

    def feature_importance(self) -> pd.Series:
        self._check_ready()
        return pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)

    # -- persistence -------------------------------------------------------
    def save(self, path=RANKING_MODEL) -> None:
        self._check_ready()
        joblib.dump({"model": self.model, "features": self.features}, path)

    @classmethod
    def load(cls, path=RANKING_MODEL) -> "KeywordRankingModel":
        obj = cls()
        blob = joblib.load(path)
        obj.model = blob["model"]
        obj.features = blob["features"]
        return obj

    def _check_ready(self) -> None:
        if self.model is None:
            raise RuntimeError("KeywordRankingModel is not trained. Call fit() or load().")
