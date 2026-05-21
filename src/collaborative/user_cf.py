from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .base import BaseCollaborativeFilter
from ..datasets.schemas import Rating

logger = logging.getLogger(__name__)


@dataclass
class UserCFConfig:
    n_neighbors: int = 20
    min_common_ratings: int = 3
    rating_min: float = 1.0
    rating_max: float = 5.0


class UserBasedCF(BaseCollaborativeFilter):
    """User-based collaborative filtering with cosine similarity."""

    def __init__(self, config: UserCFConfig | None = None) -> None:
        self.config = config or UserCFConfig()
        self._user_ratings: dict[str, dict[str, float]] = {}
        self._item_users: dict[str, set[str]] = {}
        self._global_mean: float = 0.0

    def fit(self, ratings: list[Rating]) -> "UserBasedCF":
        if not ratings:
            raise ValueError("Cannot fit on empty ratings list")
        self._user_ratings = {}
        self._item_users = {}
        for r in ratings:
            self._user_ratings.setdefault(r.user_id, {})[r.item_id] = r.rating
            self._item_users.setdefault(r.item_id, set()).add(r.user_id)
        self._global_mean = float(np.mean([r.rating for r in ratings]))
        return self

    def _cosine_similarity(self, u1: str, u2: str) -> float:
        r1 = self._user_ratings.get(u1, {})
        r2 = self._user_ratings.get(u2, {})
        common = set(r1) & set(r2)
        if len(common) < self.config.min_common_ratings:
            return 0.0
        v1 = np.array([r1[i] for i in common], dtype=float)
        v2 = np.array([r2[i] for i in common], dtype=float)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0

    def _get_neighbors(self, user_id: str) -> list[tuple[str, float]]:
        sims = [
            (other, self._cosine_similarity(user_id, other))
            for other in self._user_ratings
            if other != user_id
        ]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[: self.config.n_neighbors]

    def predict(self, user_id: str, item_id: str) -> float:
        if not self._user_ratings:
            raise RuntimeError("Model must be fitted before calling predict()")
        neighbors = self._get_neighbors(user_id)
        positives = [
            (u, s)
            for u, s in neighbors
            if s > 0 and item_id in self._user_ratings.get(u, {})
        ]
        if not positives:
            return self._global_mean
        num = sum(s * self._user_ratings[u][item_id] for u, s in positives)
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
        if not self._user_ratings:
            raise RuntimeError("Model must be fitted before calling recommend()")
        seen = set(self._user_ratings.get(user_id, {})) if exclude_seen else set()
        candidates = {it for it in self._item_users if it not in seen}
        scores = [(it, self.predict(user_id, it)) for it in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]
