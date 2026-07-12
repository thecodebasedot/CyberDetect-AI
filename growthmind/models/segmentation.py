"""Customer Behavior — K-Means user segmentation.

Groups visitors into behavioural segments from engagement + purchase features,
then labels each cluster with a human-readable persona (New, Returning,
Potential Buyer, Buyer) based on its centroid, and estimates a simple Customer
Lifetime Value per segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import KMEANS_PARAMS, SEGMENTER_MODEL, USER_FEATURES


@dataclass
class SegmentationMetrics:
    silhouette: float
    n_clusters: int

    def pretty(self) -> str:
        return (f"User segmenter — {self.n_clusters} segments  "
                f"silhouette={self.silhouette:.3f}")


class UserSegmenter:
    """K-Means segmentation with automatic persona labelling."""

    def __init__(self, params: dict | None = None):
        self.params = params or dict(KMEANS_PARAMS)
        self.model: KMeans | None = None
        self.scaler: StandardScaler | None = None
        self.features = USER_FEATURES
        self.labels_: dict[int, str] = {}

    def fit(self, users: pd.DataFrame) -> SegmentationMetrics:
        X = users[self.features].to_numpy()
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        self.model = KMeans(**self.params)
        clusters = self.model.fit_predict(Xs)
        self._assign_personas(users, clusters)

        sil = float(silhouette_score(Xs, clusters))
        return SegmentationMetrics(silhouette=sil, n_clusters=self.params["n_clusters"])

    def _assign_personas(self, users: pd.DataFrame, clusters: np.ndarray) -> None:
        """Label clusters by their purchase + engagement profile."""
        tmp = users[self.features].copy()
        tmp["cluster"] = clusters
        profile = tmp.groupby("cluster").mean()

        # Rank clusters: buyers spend most; potential buyers engage but don't buy;
        # returning visitors engage moderately; new visitors barely engage.
        labels: dict[int, str] = {}
        buyer = int(profile["total_spent"].idxmax())
        labels[buyer] = "Buyer"

        remaining = [c for c in profile.index if c != buyer]
        # Among the rest, the most engaged non-buyer is a Potential Buyer.
        pot = max(remaining, key=lambda c: profile.loc[c, "pageviews"])
        labels[pot] = "Potential Buyer"

        rest = [c for c in remaining if c != pot]
        rest_sorted = sorted(rest, key=lambda c: profile.loc[c, "sessions"], reverse=True)
        names = ["Returning Visitor", "New Visitor"]
        for c, name in zip(rest_sorted, names):
            labels[c] = name
        # Any leftover clusters get a generic label.
        for c in profile.index:
            labels.setdefault(int(c), f"Segment {int(c)}")

        self.labels_ = {int(k): v for k, v in labels.items()}

    def predict(self, users: pd.DataFrame) -> pd.DataFrame:
        """Return the input with ``segment`` and ``segment_label`` columns added."""
        self._check_ready()
        Xs = self.scaler.transform(users[self.features].to_numpy())
        clusters = self.model.predict(Xs)
        out = users.copy()
        out["segment"] = clusters
        out["segment_label"] = [self.labels_.get(int(c), f"Segment {c}") for c in clusters]
        return out

    def segment_summary(self, users: pd.DataFrame) -> pd.DataFrame:
        """Per-segment size, avg spend and a simple CLV estimate."""
        labelled = self.predict(users)
        summary = (
            labelled.groupby("segment_label")
            .agg(
                users=("user_id", "count") if "user_id" in labelled else ("segment", "count"),
                avg_orders=("num_orders", "mean"),
                avg_spent=("total_spent", "mean"),
                avg_pageviews=("pageviews", "mean"),
            )
            .reset_index()
        )
        # Simple CLV proxy: average spend scaled by a loyalty factor.
        summary["est_clv"] = (summary["avg_spent"] * (1 + summary["avg_orders"])).round(2)
        return summary.sort_values("est_clv", ascending=False).reset_index(drop=True)

    def save(self, path=SEGMENTER_MODEL) -> None:
        self._check_ready()
        joblib.dump(
            {"model": self.model, "scaler": self.scaler,
             "features": self.features, "labels": self.labels_},
            path,
        )

    @classmethod
    def load(cls, path=SEGMENTER_MODEL) -> "UserSegmenter":
        obj = cls()
        blob = joblib.load(path)
        obj.model = blob["model"]
        obj.scaler = blob["scaler"]
        obj.features = blob["features"]
        obj.labels_ = blob["labels"]
        return obj

    def _check_ready(self) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("UserSegmenter is not trained. Call fit() or load().")
