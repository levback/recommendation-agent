"""FAISS-backed in-memory vector store for item embeddings."""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    item_id: str
    score: float   # cosine similarity (0–1)


class VectorStore:
    """Store item vectors and search by cosine similarity.

    Uses ``faiss.IndexFlatIP`` on L2-normalised vectors which is equivalent
    to cosine similarity.
    """

    def __init__(self) -> None:
        self._index = None
        self._ids: list[str] = []

    def add(self, item_ids: list[str], embeddings: np.ndarray) -> None:
        if len(item_ids) != len(embeddings):
            raise ValueError("item_ids and embeddings must have the same length")
        if not item_ids:
            return
        import faiss  # type: ignore[import]
        dim = embeddings.shape[1]
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._ids.extend(item_ids)
        logger.debug("VectorStore: added %d items (total %d)", len(item_ids), len(self._ids))

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[SearchResult]:
        if self._index is None or not self._ids:
            return []
        k = min(k, len(self._ids))
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, k)
        return [
            SearchResult(item_id=self._ids[idx], score=float(scores[0][rank]))
            for rank, idx in enumerate(indices[0])
            if idx >= 0
        ]

    @property
    def size(self) -> int:
        return len(self._ids)

    def save(self, path: str | Path) -> None:
        import faiss  # type: ignore[import]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        index_bytes = faiss.serialize_index(self._index) if self._index else None
        with open(path, "wb") as f:
            pickle.dump({"index_bytes": index_bytes, "ids": self._ids}, f)

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        import faiss  # type: ignore[import]
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls()
        if data["index_bytes"] is not None:
            store._index = faiss.deserialize_index(data["index_bytes"])
        store._ids = data["ids"]
        return store
