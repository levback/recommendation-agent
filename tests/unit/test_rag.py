"""Unit tests for the recommendation_agent RAG layer."""
from __future__ import annotations

import numpy as np
import pytest

from src.datasets.schemas import ItemProfile
from src.rag.embedder import Embedder
from src.rag.item_retriever import ItemRetriever, _item_text
from src.rag.vector_store import SearchResult, VectorStore


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_item(item_id: str, title: str, *genres: str) -> ItemProfile:
    return ItemProfile(
        item_id=item_id,
        title=title,
        genres=tuple(genres),
        features={},
        metadata={},
    )


def _random_unit(dim: int) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ─── _item_text helper ───────────────────────────────────────────────────────

class TestItemText:
    def test_includes_title_and_genres(self):
        item = _make_item("1", "Toy Story", "Animation", "Comedy")
        text = _item_text(item)
        assert "Toy Story" in text
        assert "Animation" in text
        assert "Comedy" in text

    def test_no_genres(self):
        item = _make_item("2", "Unknown Film")
        text = _item_text(item)
        assert "unknown" in text.lower()


# ─── VectorStore ─────────────────────────────────────────────────────────────

class TestVectorStore:
    def test_add_and_search(self):
        pytest.importorskip("faiss")
        ids = ["a", "b", "c"]
        dim = 8
        vecs = np.random.randn(3, dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        store = VectorStore()
        store.add(ids, vecs)
        assert store.size == 3
        results = store.search(vecs[0], k=2)
        assert len(results) == 2
        assert results[0].item_id == "a"  # self-match

    def test_search_returns_empty_for_empty_store(self):
        store = VectorStore()
        assert store.search(np.ones(4, dtype=np.float32)) == []

    def test_mismatched_lengths_raises(self):
        store = VectorStore()
        with pytest.raises(ValueError):
            store.add(["a"], np.zeros((2, 4), dtype=np.float32))

    def test_save_load(self, tmp_path):
        pytest.importorskip("faiss")
        ids = ["x", "y"]
        vecs = np.eye(2, dtype=np.float32)
        store = VectorStore()
        store.add(ids, vecs)
        path = tmp_path / "vs.pkl"
        store.save(path)
        loaded = VectorStore.load(path)
        assert loaded.size == 2
        r = loaded.search(vecs[0], k=1)
        assert r[0].item_id == "x"


# ─── Embedder (mocked) ───────────────────────────────────────────────────────

class TestEmbedder:
    def test_embed_shape(self, mocker):
        mock_model = mocker.MagicMock()
        dim = 384
        mock_model.encode.return_value = np.random.randn(2, dim).astype(np.float32)
        emb = Embedder.__new__(Embedder)
        emb._model_name = "all-MiniLM-L6-v2"
        emb._model = mock_model
        result = emb.embed(["text one", "text two"])
        assert result.shape == (2, dim)
        assert result.dtype == np.float32

    def test_embed_one_is_1d(self, mocker):
        mock_model = mocker.MagicMock()
        mock_model.encode.return_value = np.ones((1, 16), dtype=np.float32)
        emb = Embedder.__new__(Embedder)
        emb._model_name = "all-MiniLM-L6-v2"
        emb._model = mock_model
        result = emb.embed_one("hello")
        assert result.ndim == 1


# ─── ItemRetriever ───────────────────────────────────────────────────────────

@pytest.fixture
def fake_embedder(mocker):
    """Deterministic embedder: assigns a unique axis vector to each item."""
    emb = mocker.MagicMock(spec=Embedder)

    def _embed(texts):
        n = len(texts)
        vecs = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            vecs[i, i] = 1.0  # orthonormal basis
        # pad or truncate to fixed dim=8
        dim = 8
        out = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            out[i, i % dim] = 1.0
        return out

    emb.embed.side_effect = _embed
    return emb


ITEMS = [
    _make_item("m1", "Toy Story", "Animation", "Comedy"),
    _make_item("m2", "Jumanji", "Adventure", "Family"),
    _make_item("m3", "GoodFellas", "Crime", "Drama"),
    _make_item("m4", "Aladdin", "Animation", "Comedy"),
    _make_item("m5", "Heat", "Crime", "Drama"),
]


class TestItemRetriever:
    def test_fit_populates_store(self, mocker, fake_embedder):
        pytest.importorskip("faiss")
        retriever = ItemRetriever.__new__(ItemRetriever)
        retriever.model_name = "all-MiniLM-L6-v2"
        retriever._embedder = fake_embedder
        from src.rag.vector_store import VectorStore as VS
        retriever._store = VS()
        retriever._id_to_vec = {}
        retriever._fitted = False
        retriever.fit(ITEMS)
        assert retriever.is_fitted
        assert retriever.item_count == len(ITEMS)

    def test_find_similar_excludes_self(self, mocker, fake_embedder):
        pytest.importorskip("faiss")
        retriever = ItemRetriever.__new__(ItemRetriever)
        retriever.model_name = "all-MiniLM-L6-v2"
        retriever._embedder = fake_embedder
        from src.rag.vector_store import VectorStore as VS
        retriever._store = VS()
        retriever._id_to_vec = {}
        retriever._fitted = False
        retriever.fit(ITEMS)
        similar = retriever.find_similar("m1", n=3)
        ids = [iid for iid, _ in similar]
        assert "m1" not in ids

    def test_find_similar_returns_n_items(self, mocker, fake_embedder):
        pytest.importorskip("faiss")
        retriever = ItemRetriever.__new__(ItemRetriever)
        retriever.model_name = "all-MiniLM-L6-v2"
        retriever._embedder = fake_embedder
        from src.rag.vector_store import VectorStore as VS
        retriever._store = VS()
        retriever._id_to_vec = {}
        retriever._fitted = False
        retriever.fit(ITEMS)
        similar = retriever.find_similar("m1", n=3)
        assert len(similar) <= 3

    def test_find_similar_unknown_id_returns_empty(self, mocker, fake_embedder):
        pytest.importorskip("faiss")
        retriever = ItemRetriever.__new__(ItemRetriever)
        retriever.model_name = "all-MiniLM-L6-v2"
        retriever._embedder = fake_embedder
        from src.rag.vector_store import VectorStore as VS
        retriever._store = VS()
        retriever._id_to_vec = {}
        retriever._fitted = False
        retriever.fit(ITEMS)
        assert retriever.find_similar("nonexistent") == []

    def test_user_taste_score_returns_dict(self, mocker, fake_embedder):
        pytest.importorskip("faiss")
        retriever = ItemRetriever.__new__(ItemRetriever)
        retriever.model_name = "all-MiniLM-L6-v2"
        retriever._embedder = fake_embedder
        from src.rag.vector_store import VectorStore as VS
        retriever._store = VS()
        retriever._id_to_vec = {}
        retriever._fitted = False
        retriever.fit(ITEMS)
        candidates = ["m3", "m4", "m5"]
        scores = retriever.user_taste_score(liked_item_ids=["m1", "m2"], candidate_ids=candidates)
        assert set(scores.keys()) == set(candidates)
        assert all(isinstance(v, float) for v in scores.values())

    def test_user_taste_empty_liked_returns_zeros(self, mocker, fake_embedder):
        pytest.importorskip("faiss")
        retriever = ItemRetriever.__new__(ItemRetriever)
        retriever.model_name = "all-MiniLM-L6-v2"
        retriever._embedder = fake_embedder
        from src.rag.vector_store import VectorStore as VS
        retriever._store = VS()
        retriever._id_to_vec = {}
        retriever._fitted = False
        retriever.fit(ITEMS)
        scores = retriever.user_taste_score(liked_item_ids=[], candidate_ids=["m1"])
        assert scores == {"m1": 0.0}

    def test_unfitted_returns_empty(self):
        retriever = ItemRetriever.__new__(ItemRetriever)
        retriever.model_name = "all-MiniLM-L6-v2"
        retriever._embedder = None
        retriever._store = VectorStore()
        retriever._id_to_vec = {}
        retriever._fitted = False
        scores = retriever.user_taste_score(["m1"], ["m2"])
        assert scores == {"m2": 0.0}
