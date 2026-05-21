from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..collaborative.base import BaseCollaborativeFilter
from ..collaborative.item_cf import ItemBasedCF
from ..collaborative.matrix_factorization import MFConfig, MatrixFactorization
from ..collaborative.user_cf import UserBasedCF
from ..datasets.schemas import Rating
from ..rl.agent import BanditStrategy, RLAgentConfig, RLRecommendationAgent

logger = logging.getLogger(__name__)


class CFMethod(str, Enum):
    MATRIX_FACTORIZATION = "matrix_factorization"
    USER_CF = "user_cf"
    ITEM_CF = "item_cf"


@dataclass
class HybridConfig:
    cf_method: CFMethod = CFMethod.MATRIX_FACTORIZATION
    rl_strategy: BanditStrategy = BanditStrategy.THOMPSON_SAMPLING
    cf_weight: float = 0.6      # RL weight = 1 - cf_weight
    n_recommendations: int = 10
    context_dim: int = 8
    use_rl: bool = True


@dataclass
class RecommendationResult:
    user_id: str
    recommendations: list[tuple[str, float]]
    cf_scores: dict[str, float] = field(default_factory=dict)
    rl_scores: dict[str, float] = field(default_factory=dict)
    method: str = "hybrid"


class HybridRecommender:
    """Blends collaborative filtering scores with RL-based exploration scores."""

    def __init__(self, config: HybridConfig | None = None) -> None:
        self.config = config or HybridConfig()
        self._cf: BaseCollaborativeFilter | None = None
        self._rl: RLRecommendationAgent | None = None
        self._item_ids: list[str] = []
        self._fitted = False

    def _build_cf(self) -> BaseCollaborativeFilter:
        m = self.config.cf_method
        if m == CFMethod.MATRIX_FACTORIZATION:
            return MatrixFactorization(MFConfig(n_factors=20, n_epochs=15))
        if m == CFMethod.USER_CF:
            return UserBasedCF()
        if m == CFMethod.ITEM_CF:
            return ItemBasedCF()
        raise ValueError(f"Unknown CF method: {m}")  # pragma: no cover

    def fit(self, ratings: list[Rating]) -> "HybridRecommender":
        if not ratings:
            raise ValueError("Cannot fit on empty ratings list")
        self._item_ids = sorted({r.item_id for r in ratings})
        self._cf = self._build_cf()
        self._cf.fit(ratings)
        if self.config.use_rl:
            rl_cfg = RLAgentConfig(
                strategy=self.config.rl_strategy,
                context_dim=self.config.context_dim,
            )
            self._rl = RLRecommendationAgent(self._item_ids, rl_cfg)
        self._fitted = True
        logger.info(
            "HybridRecommender fitted | CF=%s  RL=%s  items=%d",
            self.config.cf_method.value,
            self.config.rl_strategy.value if self.config.use_rl else "off",
            len(self._item_ids),
        )
        return self

    def recommend(
        self,
        user_id: str,
        n: int | None = None,
        context: np.ndarray | None = None,
        exclude_seen: bool = True,
    ) -> RecommendationResult:
        if not self._fitted or self._cf is None:
            raise RuntimeError("Must call fit() before recommend()")
        n = n or self.config.n_recommendations

        cf_recs = self._cf.recommend(
            user_id, n=len(self._item_ids), exclude_seen=exclude_seen
        )
        cf_raw = dict(cf_recs)

        # Normalize CF scores to [0, 1]
        if cf_raw:
            vals = np.array(list(cf_raw.values()))
            vmin, vmax = vals.min(), vals.max()
            cf_norm = (
                {iid: (s - vmin) / (vmax - vmin) for iid, s in cf_raw.items()}
                if vmax > vmin
                else {iid: 0.5 for iid in cf_raw}
            )
        else:
            cf_norm = {}

        # Get and normalize RL estimates
        rl_norm: dict[str, float] = {}
        if self.config.use_rl and self._rl is not None:
            rl_vals = self._rl.estimated_item_values
            vals = np.array(list(rl_vals.values()))
            vmin, vmax = vals.min(), vals.max()
            rl_norm = (
                {iid: (v - vmin) / (vmax - vmin) for iid, v in rl_vals.items()}
                if vmax > vmin
                else {iid: 0.5 for iid in rl_vals}
            )

        w_cf = self.config.cf_weight
        w_rl = 1.0 - w_cf
        all_items = set(cf_norm) | set(rl_norm)
        blended = {
            iid: w_cf * cf_norm.get(iid, 0.5) + w_rl * rl_norm.get(iid, 0.5)
            for iid in all_items
        }
        ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:n]

        return RecommendationResult(
            user_id=user_id,
            recommendations=ranked,
            cf_scores=cf_raw,
            rl_scores={iid: float(np.array(list(rl_norm.values())).mean()) for iid in rl_norm} if rl_norm else {},
            method=f"hybrid({self.config.cf_method.value}+{self.config.rl_strategy.value})",
        )

    def update_feedback(
        self,
        user_id: str,
        item_id: str,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        """Feed observed reward back into the RL agent."""
        if not self.config.use_rl or self._rl is None:
            return
        # Normalize from rating scale [1,5] to [0,1]
        norm = (reward - 1.0) / 4.0 if reward > 1.0 else max(0.0, reward)
        self._rl.update(item_id, norm, context)
