from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .evaluator import RecommendationEvaluator
from .hybrid import HybridConfig, HybridRecommender, RecommendationResult
from ..datasets.schemas import ItemProfile, Rating, UserProfile
from ..narration.narrator import NarrationResult, RecommendationNarrator

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    n_recommendations: int = 10
    use_narration: bool = False   # requires AWS Bedrock credentials
    test_ratio: float = 0.2
    seed: int = 42


@dataclass
class PipelineResult:
    user_id: str
    recommendations: list[tuple[str, float]]
    narrative: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    method: str = "pipeline"


class RecommendationPipeline:
    """End-to-end pipeline: fit → recommend → (optionally) narrate."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        narrator: RecommendationNarrator | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._recommender = HybridRecommender(self.config.hybrid)
        self._evaluator = RecommendationEvaluator()
        self._narrator = narrator
        self._items_by_id: dict[str, ItemProfile] = {}
        self._users_by_id: dict[str, UserProfile] = {}
        self._fitted = False

    def fit(
        self,
        ratings: list[Rating],
        items: list[ItemProfile] | None = None,
        users: list[UserProfile] | None = None,
    ) -> "RecommendationPipeline":
        self._recommender.fit(ratings)
        self._items_by_id = {it.item_id: it for it in (items or [])}
        self._users_by_id = {u.user_id: u for u in (users or [])}
        self._fitted = True
        return self

    def run(
        self,
        user_id: str,
        n: int | None = None,
        context: dict[str, float] | None = None,
    ) -> PipelineResult:
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before calling run()")
        start = time.monotonic()
        n = n or self.config.n_recommendations

        # Build a fixed-length context vector from user/context features
        np_context: np.ndarray | None = None
        user_prof = self._users_by_id.get(user_id)
        combined = {**(user_prof.features if user_prof else {}), **(context or {})}
        if combined:
            dim = self.config.hybrid.context_dim
            keys = sorted(combined)[:dim]
            vec = np.zeros(dim)
            for i, k in enumerate(keys):
                vec[i] = combined[k]
            np_context = vec

        rec: RecommendationResult = self._recommender.recommend(
            user_id, n=n, context=np_context
        )

        narrative: str | None = None
        if self.config.use_narration and self._narrator is not None:
            try:
                nr: NarrationResult = self._narrator.narrate(
                    user_id=user_id,
                    recommendations=rec.recommendations,
                    items_by_id=self._items_by_id,
                    user_profile=user_prof,
                )
                narrative = nr.narrative
            except Exception as exc:
                logger.warning("Narration failed: %s", exc)

        return PipelineResult(
            user_id=user_id,
            recommendations=rec.recommendations,
            narrative=narrative,
            elapsed_seconds=time.monotonic() - start,
            method=rec.method,
        )

    def evaluate(self, test_ratings: list[Rating], k: int = 10) -> dict[str, float]:
        """Compute RMSE/MAE on held-out ratings."""
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before evaluate()")
        actuals, preds = [], []
        cf = self._recommender._cf
        for r in test_ratings:
            actuals.append(r.rating)
            preds.append(cf.predict(r.user_id, r.item_id) if cf else 3.0)
        metrics: dict[str, float] = {}
        if actuals:
            metrics["rmse"] = self._evaluator.rmse(actuals, preds)
            metrics["mae"] = self._evaluator.mae(actuals, preds)
        return metrics

    def update_feedback(
        self,
        user_id: str,
        item_id: str,
        reward: float,
    ) -> None:
        self._recommender.update_feedback(user_id, item_id, reward)
