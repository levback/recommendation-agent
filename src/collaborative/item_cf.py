from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .base import BaseCollaborativeFilter
from ..datasets.schemas import Rating

logger = logging.getLogger(__name__)


@dataclass
class ItemCFConfig:
    n_neighbors: int = 20
    min_common_users: int = 3
    rating_min: float = 1.0
    rating_max: float = 5.0


class ItemBasedCF(BaseCollaborativeFilter):
    """Item-based CF using adjusted cosine similarity."""

    def __init__(self, config: ItemCFConfig | None = None) -> None:
        self.config = config or ItemCFConfig()
        self._user_ratings: dict[str, dict[str, float]] = {}
        self._item_ratings: dict[str, dict[str, float]] = {}  # item → {user: r}
        self._user_means: dict[str, float] = {}
        self._global_mean: float = 0.0

    def fit(self, ratings: list[Rating]) -> "ItemBasedCF":
        if not ratings:
            raise ValueError("Cannot fit on empty ratings list")
        self._user_ratings = {}
        self._item_ratings = {}
        for r in ratings:
            self._user_ratings.setdefault(r.user_id, {})[r.item_id] = r.rating
            self._item_ratings.setdefault(r.item_id, {})[r.user_id] = r.rating
        self._user_means = {
            u: float(np.mean(list(rs.values())))
            for u, rs in self._user_ratings.items()
        }
        self._global_mean = float(np.mean([r.rating for r in ratings]))
        return self

    def _adjusted_cosine(self, i1: str, i2: str) -> float:
        r1 = self._item_ratings.get(i1, {})
        r2 = self._item_ratings.get(i2, {})
        common = set(r1) & set(r2)
        if len(common) < self.config.min_common_users:
            return 0.0
        v1 = np.array(
            [r1[u] - self._user_means.get(u, self._global_mean) for u in common],
            dtype=float,
        )
        v2 = np.array(
            [r2[u] - self._user_means.get(u, self._global_mean) for u in common],
            dtype=float,
        )
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0

    def predict(self, user_id: str, item_id: str) -> float:
        if not self._item_ratings:
            raise RuntimeError("Model must be fitted before calling predict()")
        user_rs = self._user_ratings.get(user_id, {})
        if not user_rs:
            return self._global_mean
        sims = [
            (it, self._adjusted_cosine(item_id, it))
            for it in user_rs
            if it != item_id
        ]
        positives = sorted(
            [(it, s) for it, s in sims if s > 0],
            key=lambda x: x[1],
            reverse=True,
        )[: self.config.n_neighbors]
        if not positives:
            return self._global_mean
        num = sum(s * user_rs[it] for it, s in positives)
        den = sum(s for _, s in positives)
        if den == 0:
            return self._global_mean
        return float(np.clip(num / den, self.config.rating_min, self.config.rating_max))

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        if not self._item_ratings:
            raise RuntimeError("Model must be fitted before calling recommend()")
        seen = set(self._user_ratings.get(user_id, {})) if exclude_seen else set()
        candidates = {it for it in self._item_ratings if it not in seen}
        scores = [(it, self.predict(user_id, it)) for it in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]
