"""Tests for the LightGBM keyword ranking model and its dataset."""

from __future__ import annotations

import numpy as np
import pytest

from growthmind.config import RANKING_FEATURES
from growthmind.data import generate_keywords
from growthmind.models import KeywordRankingModel


@pytest.fixture(scope="module")
def keywords():
    return generate_keywords(n_keywords=200, pages_per_keyword=10, random_state=17)


def test_keyword_dataset_schema_and_groups(keywords):
    for col in RANKING_FEATURES + ["keyword_id", "is_ours", "relevance_label", "position"]:
        assert col in keywords.columns
    # Each keyword has exactly one "our" page and contiguous groups.
    per_kw = keywords.groupby("keyword_id")
    assert (per_kw["is_ours"].sum() == 1).all()
    assert (per_kw.size() == 10).all()
    # Positions within a group are a permutation of 1..10.
    first = keywords[keywords["keyword_id"] == keywords["keyword_id"].iloc[0]]
    assert sorted(first["position"]) == list(range(1, 11))
    # Graded labels are in the LTR range.
    assert keywords["relevance_label"].between(0, 4).all()


def test_ranker_learns_ordering(keywords):
    ranker = KeywordRankingModel()
    metrics = ranker.fit(keywords)
    # A learnable ranking signal -> high NDCG.
    assert metrics.ndcg_at_10 > 0.85
    assert metrics.n_test_keywords > 0


def test_predict_ranking_positions(keywords):
    ranker = KeywordRankingModel()
    ranker.fit(keywords)
    ranked = ranker.predict_ranking(keywords)
    assert "predicted_position" in ranked.columns
    # Within each keyword, predicted positions are a 1..K permutation.
    grp = ranked[ranked["keyword_id"] == 0]
    assert sorted(grp["predicted_position"]) == list(range(1, len(grp) + 1))


def test_opportunities_are_striking_distance(keywords):
    ranker = KeywordRankingModel()
    ranker.fit(keywords)
    opps = ranker.opportunities(keywords, top=10)
    assert len(opps) <= 10
    if len(opps):
        assert opps["predicted_position"].between(4, 15).all()
        # Sorted by search volume descending.
        sv = opps["search_volume"].to_numpy()
        assert np.all(np.diff(sv) <= 0)


def test_feature_importance_covers_all_features(keywords):
    ranker = KeywordRankingModel()
    ranker.fit(keywords)
    imp = ranker.feature_importance()
    assert set(imp.index) == set(RANKING_FEATURES)


def test_unfit_raises(keywords):
    with pytest.raises(RuntimeError):
        KeywordRankingModel().predict_ranking(keywords)


def test_save_load_roundtrip(keywords, tmp_path):
    ranker = KeywordRankingModel()
    ranker.fit(keywords)
    p = tmp_path / "ranker.joblib"
    ranker.save(p)
    restored = KeywordRankingModel.load(p)
    np.testing.assert_allclose(
        ranker.predict_ranking(keywords)["rank_score"].to_numpy(),
        restored.predict_ranking(keywords)["rank_score"].to_numpy(),
    )
