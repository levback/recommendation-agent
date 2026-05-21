from __future__ import annotations

from abc import ABC, abstractmethod

from ..datasets.schemas import Rating


class BaseCollaborativeFilter(ABC):
    """Abstract base for all collaborative filtering models."""

    @abstractmethod
    def fit(self, ratings: list[Rating]) -> "BaseCollaborativeFilter": ...

    @abstractmethod
    def predict(self, user_id: str, item_id: str) -> float: ...

    @abstractmethod
    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        """Return top-n (item_id, predicted_score) pairs for a user."""
        ...

    def fit_predict(
        self,
        train: list[Rating],
        test_pairs: list[tuple[str, str]],
    ) -> list[float]:
        """Convenience: fit on train, return predictions for test pairs."""
        self.fit(train)
        return [self.predict(u, i) for u, i in test_pairs]
