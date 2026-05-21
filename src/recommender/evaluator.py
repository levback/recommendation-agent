from __future__ import annotations

import numpy as np


class RecommendationEvaluator:
    """Standard recommendation evaluation metrics."""

    @staticmethod
    def rmse(actual: list[float], predicted: list[float]) -> float:
        if not actual or not predicted:
            raise ValueError("actual and predicted must be non-empty")
        if len(actual) != len(predicted):
            raise ValueError("actual and predicted must have the same length")
        return float(np.sqrt(np.mean([(a - p) ** 2 for a, p in zip(actual, predicted)])))

    @staticmethod
    def mae(actual: list[float], predicted: list[float]) -> float:
        if not actual or not predicted:
            raise ValueError("actual and predicted must be non-empty")
        if len(actual) != len(predicted):
            raise ValueError("actual and predicted must have the same length")
        return float(np.mean([abs(a - p) for a, p in zip(actual, predicted)]))

    @staticmethod
    def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
        if k <= 0:
            raise ValueError("k must be > 0")
        top_k = recommended[:k]
        return sum(1 for it in top_k if it in relevant) / k if top_k else 0.0

    @staticmethod
    def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
        if k <= 0:
            raise ValueError("k must be > 0")
        if not relevant:
            return 0.0
        top_k = recommended[:k]
        return sum(1 for it in top_k if it in relevant) / len(relevant)

    @staticmethod
    def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
        if k <= 0:
            raise ValueError("k must be > 0")
        top_k = recommended[:k]
        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, it in enumerate(top_k)
            if it in relevant
        )
        ideal_hits = min(k, len(relevant))
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
        return float(dcg / idcg) if idcg > 0 else 0.0

    def evaluate_all(
        self,
        recommended: list[str],
        relevant: set[str],
        predicted: list[float] | None = None,
        actual: list[float] | None = None,
        k: int = 10,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {
            f"precision@{k}": self.precision_at_k(recommended, relevant, k),
            f"recall@{k}": self.recall_at_k(recommended, relevant, k),
            f"ndcg@{k}": self.ndcg_at_k(recommended, relevant, k),
        }
        if predicted is not None and actual is not None:
            metrics["rmse"] = self.rmse(actual, predicted)
            metrics["mae"] = self.mae(actual, predicted)
        return metrics
