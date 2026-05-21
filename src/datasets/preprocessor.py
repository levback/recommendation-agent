from __future__ import annotations

import random

import numpy as np

from .schemas import DatasetSplit, ItemProfile, Rating, UserProfile


class DatasetPreprocessor:
    """Splits and normalizes rating datasets."""

    def __init__(self, test_ratio: float = 0.2, seed: int = 42) -> None:
        if not (0.0 < test_ratio < 1.0):
            raise ValueError("test_ratio must be in (0, 1)")
        self.test_ratio = test_ratio
        self.seed = seed

    def split(
        self,
        ratings: list[Rating],
        users: list[UserProfile] | None = None,
        items: list[ItemProfile] | None = None,
    ) -> DatasetSplit:
        """Randomly split ratings into train and test sets."""
        if not ratings:
            raise ValueError("Cannot split empty ratings list")
        rng = random.Random(self.seed)
        shuffled = list(ratings)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * self.test_ratio))
        return DatasetSplit(
            train=shuffled[n_test:],
            test=shuffled[:n_test],
            users=users or [],
            items=items or [],
        )

    def normalize_ratings(
        self,
        ratings: list[Rating],
        min_r: float = 1.0,
        max_r: float = 5.0,
    ) -> list[Rating]:
        """Min-max normalize ratings to [0, 1]."""
        span = max_r - min_r
        if span == 0:
            raise ValueError("min_r and max_r must differ")
        return [
            Rating(
                user_id=r.user_id,
                item_id=r.item_id,
                rating=float(np.clip((r.rating - min_r) / span, 0.0, 1.0)),
                timestamp=r.timestamp,
            )
            for r in ratings
        ]

    def get_relevance_sets(
        self,
        ratings: list[Rating],
        threshold: float = 4.0,
    ) -> dict[str, set[str]]:
        """Return a mapping of user_id → set of relevant item_ids."""
        result: dict[str, set[str]] = {}
        for r in ratings:
            if r.rating >= threshold:
                result.setdefault(r.user_id, set()).add(r.item_id)
        return result
