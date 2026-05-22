"""ItemRetriever: embed ItemProfile metadata and find semantically similar items."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..datasets.schemas import ItemProfile
from .embedder import Embedder
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class ItemRetriever:
    """Build a semantic FAISS index over :class:`ItemProfile` objects.

    Usage::

        retriever = ItemRetriever()
        retriever.fit(items)
        similar = retriever.find_similar("movie:1", n=10)
        taste = retriever.user_taste_score(
            liked_item_ids=["movie:1", "movie:4"],
            candidate_ids=["movie:7", "movie:9", "movie:12"],
        )
    """

    model_name: str = "all-MiniLM-L6-v2"

    _embedder: Embedder = field(init=False)
    _store: VectorStore = field(init=False)
    _id_to_vec: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    _fitted: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._embedder = Embedder(self.model_name)
        self._store = VectorStore()

    # ── public ────────────────────────────────────────────────────────────────

    def fit(self, items: list[ItemProfile]) -> "ItemRetriever":
        """Embed all *items* and build FAISS index.  Returns *self* for chaining."""
        if not items:
            logger.warning("ItemRetriever.fit: received empty item list")
            return self
        texts = [_item_text(item) for item in items]
        embeddings = self._embedder.embed(texts)
        ids = [item.item_id for item in items]
        self._store.add(ids, embeddings)
        self._id_to_vec = {iid: embeddings[i] for i, iid in enumerate(ids)}
        self._fitted = True
        logger.info("ItemRetriever fitted on %d items", len(items))
        return self

    def find_similar(self, item_id: str, n: int = 10) -> list[tuple[str, float]]:
        """Return top-*n* items most similar to *item_id* (excluding itself)."""
        vec = self._id_to_vec.get(item_id)
        if vec is None:
            logger.debug("ItemRetriever.find_similar: unknown item_id %r", item_id)
            return []
        results = self._store.search(vec, k=n + 1)  # +1 to skip self
        return [
            (r.item_id, r.score)
            for r in results
            if r.item_id != item_id
        ][:n]

    def user_taste_score(
        self, liked_item_ids: list[str], candidate_ids: list[str]
    ) -> dict[str, float]:
        """Compute a semantic affinity score for each candidate item.

        Averages cosine similarities from each *liked* item to each *candidate*.
        Returns a dict mapping candidate_id → score in [0, 1].
        """
        if not liked_item_ids or not candidate_ids or not self._fitted:
            return {c: 0.0 for c in candidate_ids}

        liked_vecs = [
            self._id_to_vec[iid]
            for iid in liked_item_ids
            if iid in self._id_to_vec
        ]
        if not liked_vecs:
            return {c: 0.0 for c in candidate_ids}

        # Mean embedding of liked items = the user's latent taste vector
        taste_vec = np.mean(liked_vecs, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(taste_vec))
        if norm > 0:
            taste_vec /= norm

        # Score candidates
        scores: dict[str, float] = {}
        for cid in candidate_ids:
            cvec = self._id_to_vec.get(cid)
            if cvec is None:
                scores[cid] = 0.0
            else:
                scores[cid] = float(np.dot(taste_vec, cvec))
        return scores

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def item_count(self) -> int:
        return self._store.size


# ── helpers ───────────────────────────────────────────────────────────────────

def _item_text(item: ItemProfile) -> str:
    genres = ", ".join(item.genres) if item.genres else "unknown"
    return f"Title: {item.title}. Genres: {genres}."
