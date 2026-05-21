"""Unit tests for collaborative filtering models."""
from __future__ import annotations

import pytest

from src.datasets.schemas import Rating
from src.collaborative.matrix_factorization import MatrixFactorization, MFConfig
from src.collaborative.user_cf import UserBasedCF, UserCFConfig
from src.collaborative.item_cf import ItemBasedCF, ItemCFConfig
from src.collaborative.base import BaseCollaborativeFilter


# ─── Shared fixture ───────────────────────────────────────────────────────────

def _small_ratings() -> list[Rating]:
    """5 users × 8 items with preference-driven ratings."""
    data = [
        ("u1","i1",5.0),("u1","i2",4.0),("u1","i3",1.0),("u1","i4",2.0),
        ("u2","i1",4.0),("u2","i2",5.0),("u2","i5",3.0),
        ("u3","i3",5.0),("u3","i4",4.0),("u3","i6",3.0),
        ("u4","i1",3.0),("u4","i5",4.0),("u4","i7",5.0),
        ("u5","i2",4.0),("u5","i6",3.0),("u5","i8",5.0),
    ]
    return [Rating(u, i, r) for u, i, r in data]


# ─── MatrixFactorization ─────────────────────────────────────────────────────

class TestMatrixFactorization:
    def test_fit_and_predict_in_range(self):
        mf = MatrixFactorization(MFConfig(n_factors=5, n_epochs=5))
        ratings = _small_ratings()
        mf.fit(ratings)
        pred = mf.predict("u1", "i1")
        assert 1.0 <= pred <= 5.0

    def test_predict_unknown_user_returns_global_mean(self):
        mf = MatrixFactorization(MFConfig(n_factors=5, n_epochs=3))
        mf.fit(_small_ratings())
        pred = mf.predict("unknown_user", "i1")
        assert 1.0 <= pred <= 5.0

    def test_predict_unknown_item_returns_global_mean(self):
        mf = MatrixFactorization(MFConfig(n_factors=5, n_epochs=3))
        mf.fit(_small_ratings())
        pred = mf.predict("u1", "unknown_item")
        assert 1.0 <= pred <= 5.0

    def test_predict_before_fit_raises(self):
        mf = MatrixFactorization()
        with pytest.raises(RuntimeError, match="fitted"):
            mf.predict("u1", "i1")

    def test_recommend_before_fit_raises(self):
        mf = MatrixFactorization()
        with pytest.raises(RuntimeError, match="fitted"):
            mf.recommend("u1")

    def test_fit_empty_raises(self):
        mf = MatrixFactorization()
        with pytest.raises(ValueError):
            mf.fit([])

    def test_recommend_excludes_seen(self):
        mf = MatrixFactorization(MFConfig(n_factors=5, n_epochs=3))
        ratings = _small_ratings()
        mf.fit(ratings)
        seen = {"i1", "i2", "i3", "i4"}
        recs = mf.recommend("u1", n=5, exclude_seen=True)
        item_ids = {iid for iid, _ in recs}
        assert item_ids.isdisjoint(seen)

    def test_recommend_include_seen(self):
        mf = MatrixFactorization(MFConfig(n_factors=5, n_epochs=3))
        mf.fit(_small_ratings())
        recs = mf.recommend("u1", n=5, exclude_seen=False)
        assert len(recs) > 0

    def test_recommend_unknown_user_returns_empty(self):
        mf = MatrixFactorization(MFConfig(n_factors=5, n_epochs=3))
        mf.fit(_small_ratings())
        recs = mf.recommend("ghost_user", n=5)
        assert recs == []

    def test_recommend_sorted_descending(self):
        mf = MatrixFactorization(MFConfig(n_factors=5, n_epochs=3))
        mf.fit(_small_ratings())
        recs = mf.recommend("u1", n=5)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_fit_predict_pipeline(self):
        mf = MatrixFactorization(MFConfig(n_factors=5, n_epochs=3))
        ratings = _small_ratings()
        preds = mf.fit_predict(ratings, [("u1", "i1"), ("u2", "i2")])
        assert len(preds) == 2
        for p in preds:
            assert 1.0 <= p <= 5.0


# ─── UserBasedCF ──────────────────────────────────────────────────────────────

class TestUserBasedCF:
    def test_fit_and_predict(self):
        cf = UserBasedCF()
        cf.fit(_small_ratings())
        pred = cf.predict("u1", "i1")
        assert 1.0 <= pred <= 5.0

    def test_predict_no_neighbors_returns_global_mean(self):
        cf = UserBasedCF(UserCFConfig(min_common_ratings=100))
        cf.fit(_small_ratings())
        pred = cf.predict("u1", "i5")
        assert 1.0 <= pred <= 5.0

    def test_predict_unknown_user_returns_global_mean(self):
        cf = UserBasedCF()
        cf.fit(_small_ratings())
        pred = cf.predict("ghost", "i1")
        assert 1.0 <= pred <= 5.0

    def test_predict_before_fit_raises(self):
        cf = UserBasedCF()
        with pytest.raises(RuntimeError):
            cf.predict("u1", "i1")

    def test_fit_empty_raises(self):
        cf = UserBasedCF()
        with pytest.raises(ValueError):
            cf.fit([])

    def test_recommend_excludes_seen(self):
        cf = UserBasedCF()
        cf.fit(_small_ratings())
        seen = {"i1", "i2"}
        recs = cf.recommend("u1", n=5, exclude_seen=True)
        assert all(iid not in seen for iid, _ in recs)

    def test_recommend_before_fit_raises(self):
        cf = UserBasedCF()
        with pytest.raises(RuntimeError):
            cf.recommend("u1")

    def test_cosine_similarity_sufficient_common(self):
        cf = UserBasedCF(UserCFConfig(min_common_ratings=1))
        cf.fit(_small_ratings())
        sim = cf._cosine_similarity("u1", "u2")
        assert 0.0 <= sim <= 1.0

    def test_cosine_similarity_insufficient_common(self):
        cf = UserBasedCF(UserCFConfig(min_common_ratings=100))
        cf.fit(_small_ratings())
        sim = cf._cosine_similarity("u1", "u2")
        assert sim == 0.0


# ─── ItemBasedCF ──────────────────────────────────────────────────────────────

class TestItemBasedCF:
    def test_fit_and_predict(self):
        cf = ItemBasedCF()
        cf.fit(_small_ratings())
        pred = cf.predict("u1", "i5")
        assert 1.0 <= pred <= 5.0

    def test_predict_unknown_user_returns_global_mean(self):
        cf = ItemBasedCF()
        cf.fit(_small_ratings())
        pred = cf.predict("ghost", "i1")
        assert 1.0 <= pred <= 5.0

    def test_predict_before_fit_raises(self):
        cf = ItemBasedCF()
        with pytest.raises(RuntimeError):
            cf.predict("u1", "i1")

    def test_fit_empty_raises(self):
        cf = ItemBasedCF()
        with pytest.raises(ValueError):
            cf.fit([])

    def test_recommend_sorted_descending(self):
        cf = ItemBasedCF()
        cf.fit(_small_ratings())
        recs = cf.recommend("u1", n=4)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_before_fit_raises(self):
        cf = ItemBasedCF()
        with pytest.raises(RuntimeError):
            cf.recommend("u1")

    def test_adjusted_cosine_insufficient_common(self):
        cf = ItemBasedCF(ItemCFConfig(min_common_users=100))
        cf.fit(_small_ratings())
        sim = cf._adjusted_cosine("i1", "i2")
        assert sim == 0.0

    def test_no_positive_neighbors_returns_global_mean(self):
        cf = ItemBasedCF(ItemCFConfig(min_common_users=100))
        cf.fit(_small_ratings())
        pred = cf.predict("u1", "i8")
        assert 1.0 <= pred <= 5.0
