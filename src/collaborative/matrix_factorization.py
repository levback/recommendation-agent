from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .base import BaseCollaborativeFilter
from ..datasets.schemas import Rating

logger = logging.getLogger(__name__)


@dataclass
class MFConfig:
    n_factors: int = 20
    n_epochs: int = 20
    lr: float = 0.005
    reg: float = 0.02
    rating_min: float = 1.0
    rating_max: float = 5.0
    seed: int = 42


class MatrixFactorization(BaseCollaborativeFilter):
    """SGD-based matrix factorization with user/item biases.

    Implements a standard biased MF:
        r̂_ui = μ + b_u + b_i + p_u · q_i
    where μ is the global mean, b_u / b_i are user/item biases,
    and p_u / q_i are latent factor vectors.
    """

    def __init__(self, config: MFConfig | None = None) -> None:
        self.config = config or MFConfig()
        self._user_idx: dict[str, int] = {}
        self._item_idx: dict[str, int] = {}
        self._P: np.ndarray | None = None   # (n_users, k)
        self._Q: np.ndarray | None = None   # (n_items, k)
        self._bu: np.ndarray | None = None  # (n_users,)
        self._bi: np.ndarray | None = None  # (n_items,)
        self._global_mean: float = 0.0
        self._seen: dict[str, set[str]] = {}

    def fit(self, ratings: list[Rating]) -> "MatrixFactorization":
        if not ratings:
            raise ValueError("Cannot fit on empty ratings list")

        rng = np.random.default_rng(self.config.seed)
        users = sorted({r.user_id for r in ratings})
        items = sorted({r.item_id for r in ratings})
        self._user_idx = {u: i for i, u in enumerate(users)}
        self._item_idx = {it: i for i, it in enumerate(items)}

        n_u, n_i, k = len(users), len(items), self.config.n_factors
        self._P = rng.normal(0, 0.1, (n_u, k))
        self._Q = rng.normal(0, 0.1, (n_i, k))
        self._bu = np.zeros(n_u)
        self._bi = np.zeros(n_i)
        self._global_mean = float(np.mean([r.rating for r in ratings]))
        self._seen = {u: set() for u in users}
        for r in ratings:
            self._seen[r.user_id].add(r.item_id)

        lr, reg = self.config.lr, self.config.reg
        for epoch in range(self.config.n_epochs):
            ep_rng = np.random.default_rng(self.config.seed + epoch)
            order = ep_rng.permutation(len(ratings))
            total_loss = 0.0
            for idx in order:
                r = ratings[int(idx)]
                u = self._user_idx[r.user_id]
                i = self._item_idx[r.item_id]
                err = r.rating - self._predict_idx(u, i)
                total_loss += err ** 2
                self._bu[u] += lr * (err - reg * self._bu[u])
                self._bi[i] += lr * (err - reg * self._bi[i])
                p_u = self._P[u].copy()
                q_i = self._Q[i].copy()
                self._P[u] += lr * (err * q_i - reg * p_u)
                self._Q[i] += lr * (err * p_u - reg * q_i)
            logger.debug(
                "MF epoch %d/%d  RMSE=%.4f",
                epoch + 1,
                self.config.n_epochs,
                np.sqrt(total_loss / len(ratings)),
            )
        return self

    def _predict_idx(self, u: int, i: int) -> float:
        assert self._P is not None and self._Q is not None
        raw = (
            self._global_mean
            + self._bu[u]
            + self._bi[i]
            + float(self._P[u] @ self._Q[i])
        )
        return float(np.clip(raw, self.config.rating_min, self.config.rating_max))

    def predict(self, user_id: str, item_id: str) -> float:
        if self._P is None:
            raise RuntimeError("Model must be fitted before calling predict()")
        u = self._user_idx.get(user_id)
        i = self._item_idx.get(item_id)
        if u is None or i is None:
            return self._global_mean
        return self._predict_idx(u, i)

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        if self._P is None:
            raise RuntimeError("Model must be fitted before calling recommend()")
        if user_id not in self._user_idx:
            return []
        seen = self._seen.get(user_id, set()) if exclude_seen else set()
        candidates = [it for it in self._item_idx if it not in seen]
        scores = [(it, self.predict(user_id, it)) for it in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]
