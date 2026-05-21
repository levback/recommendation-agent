from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Rating:
    """An observed user-item rating (e.g. 1–5 stars)."""

    user_id: str
    item_id: str
    rating: float
    timestamp: int | None = None

    def __post_init__(self) -> None:
        if not self.user_id or not isinstance(self.user_id, str):
            raise ValueError("user_id must be a non-empty string")
        if not self.item_id or not isinstance(self.item_id, str):
            raise ValueError("item_id must be a non-empty string")
        if not (0.0 <= self.rating <= 5.0):
            raise ValueError(f"rating must be in [0, 5], got {self.rating}")


@dataclass(frozen=True)
class UserProfile:
    """A user with optional genre-preference feature vector."""

    user_id: str
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ItemProfile:
    """A recommendable item (movie, product, etc.)."""

    item_id: str
    title: str
    genres: tuple[str, ...] = field(default_factory=tuple)
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSplit:
    """Train / test split of a rating dataset."""

    train: list[Rating]
    test: list[Rating]
    users: list[UserProfile]
    items: list[ItemProfile]
    n_users: int = 0
    n_items: int = 0

    def __post_init__(self) -> None:
        all_ratings = self.train + self.test
        self.n_users = len({r.user_id for r in all_ratings})
        self.n_items = len({r.item_id for r in all_ratings})
