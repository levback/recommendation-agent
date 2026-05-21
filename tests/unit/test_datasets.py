"""Unit tests for datasets module."""
from __future__ import annotations

import pytest

from src.datasets.loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor
from src.datasets.schemas import DatasetSplit, ItemProfile, Rating, UserProfile


# ─── Rating validation ───────────────────────────────────────────────────────

def test_rating_valid():
    r = Rating(user_id="u1", item_id="i1", rating=3.5)
    assert r.user_id == "u1"
    assert r.item_id == "i1"
    assert r.rating == 3.5


def test_rating_invalid_user_id():
    with pytest.raises(ValueError, match="user_id"):
        Rating(user_id="", item_id="i1", rating=3.0)


def test_rating_invalid_item_id():
    with pytest.raises(ValueError, match="item_id"):
        Rating(user_id="u1", item_id="", rating=3.0)


def test_rating_out_of_range():
    with pytest.raises(ValueError, match="rating"):
        Rating(user_id="u1", item_id="i1", rating=6.0)


def test_rating_frozen():
    r = Rating(user_id="u1", item_id="i1", rating=3.0)
    with pytest.raises(Exception):
        r.rating = 4.0  # type: ignore[misc]


# ─── DatasetSplit ─────────────────────────────────────────────────────────────

def test_dataset_split_counts():
    train = [Rating("u1", "i1", 4.0), Rating("u2", "i1", 3.0)]
    test = [Rating("u1", "i2", 5.0)]
    ds = DatasetSplit(train=train, test=test, users=[], items=[])
    assert ds.n_users == 2
    assert ds.n_items == 2


# ─── DatasetPreprocessor ─────────────────────────────────────────────────────

def _make_ratings(n: int = 20) -> list[Rating]:
    return [
        Rating(user_id=f"u{i % 5}", item_id=f"i{i % 10}", rating=float(1 + (i % 5)))
        for i in range(n)
    ]


def test_preprocessor_split_sizes():
    pp = DatasetPreprocessor(test_ratio=0.2, seed=0)
    ratings = _make_ratings(20)
    ds = pp.split(ratings)
    assert len(ds.train) + len(ds.test) == 20
    assert len(ds.test) >= 1


def test_preprocessor_split_empty():
    pp = DatasetPreprocessor()
    with pytest.raises(ValueError):
        pp.split([])


def test_preprocessor_invalid_ratio():
    with pytest.raises(ValueError):
        DatasetPreprocessor(test_ratio=0.0)

    with pytest.raises(ValueError):
        DatasetPreprocessor(test_ratio=1.0)


def test_normalize_ratings():
    pp = DatasetPreprocessor()
    ratings = [Rating("u1", "i1", 1.0), Rating("u2", "i2", 5.0)]
    normed = pp.normalize_ratings(ratings)
    assert normed[0].rating == pytest.approx(0.0)
    assert normed[1].rating == pytest.approx(1.0)


def test_normalize_ratings_midpoint():
    pp = DatasetPreprocessor()
    ratings = [Rating("u1", "i1", 3.0)]
    normed = pp.normalize_ratings(ratings)
    assert normed[0].rating == pytest.approx(0.5)


def test_normalize_ratings_equal_min_max():
    pp = DatasetPreprocessor()
    with pytest.raises(ValueError):
        pp.normalize_ratings([Rating("u1", "i1", 3.0)], min_r=3.0, max_r=3.0)


def test_get_relevance_sets():
    pp = DatasetPreprocessor()
    ratings = [
        Rating("u1", "i1", 4.0),
        Rating("u1", "i2", 2.0),
        Rating("u2", "i1", 5.0),
    ]
    rel = pp.get_relevance_sets(ratings, threshold=4.0)
    assert "i1" in rel["u1"]
    assert "i2" not in rel.get("u1", set())
    assert "i1" in rel["u2"]


# ─── DatasetLoader ────────────────────────────────────────────────────────────

def test_generate_synthetic_default():
    loader = DatasetLoader()
    ratings, items, users = loader.generate_synthetic(n_users=20, n_items=30, n_ratings=100)
    assert len(ratings) > 0
    assert len(items) == 30
    assert len(users) == 20
    for r in ratings:
        assert 1.0 <= r.rating <= 5.0


def test_generate_synthetic_rating_validity():
    loader = DatasetLoader()
    ratings, _, _ = loader.generate_synthetic(n_users=10, n_items=20, n_ratings=50)
    user_ids = {r.user_id for r in ratings}
    item_ids = {r.item_id for r in ratings}
    assert len(user_ids) >= 1
    assert len(item_ids) >= 1


def test_generate_synthetic_invalid():
    loader = DatasetLoader()
    with pytest.raises(ValueError):
        loader.generate_synthetic(n_users=0, n_items=10, n_ratings=5)


def test_generate_synthetic_unique_pairs():
    loader = DatasetLoader()
    ratings, _, _ = loader.generate_synthetic(n_users=5, n_items=10, n_ratings=30)
    pairs = {(r.user_id, r.item_id) for r in ratings}
    assert len(pairs) == len(ratings)   # no duplicates


def test_load_huggingface_missing_package(monkeypatch):
    """Should raise ImportError when 'datasets' is not installed."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    loader = DatasetLoader()
    with pytest.raises(ImportError):
        loader.load_huggingface()


def test_load_huggingface_fallback_on_error(monkeypatch):
    """Should fall back to synthetic data when dataset download fails."""
    import builtins
    real_import = builtins.__import__

    class _FakeDatasets:
        def load_dataset(self, *a, **kw):
            raise RuntimeError("network error")

    fake_module = _FakeDatasets()

    def mock_import(name, *args, **kwargs):
        if name == "datasets":
            return fake_module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    loader = DatasetLoader()
    ratings, items, users = loader.load_huggingface()
    assert len(ratings) > 0


def test_parse_hf_dataset_handles_missing_fields(monkeypatch):
    """_parse_hf_dataset skips rows with missing/invalid data."""
    loader = DatasetLoader()

    class _FakeDs:
        def get(self, key):
            return self

        def __iter__(self):
            yield {"userId": "u1", "movieId": "i1", "rating": 4.0, "timestamp": 0}
            yield {"garbage": "row"}   # should be skipped

    fake_ds = {"train": _FakeDs()}
    ratings, items, users = loader._parse_hf_dataset(fake_ds)
    assert any(r.user_id == "u1" for r in ratings)
