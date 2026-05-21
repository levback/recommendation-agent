from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .bandit import EpsilonGreedyBandit, UCBBandit, ThompsonSamplingBandit
from .contextual_bandit import LinUCBBandit, LinUCBConfig

logger = logging.getLogger(__name__)


class BanditStrategy(str, Enum):
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"
    THOMPSON_SAMPLING = "thompson_sampling"
    LINUCB = "linucb"


@dataclass
class RLAgentConfig:
    strategy: BanditStrategy = BanditStrategy.THOMPSON_SAMPLING
    epsilon: float = 0.1
    ucb_c: float = 2.0
    linucb_alpha: float = 1.0
    context_dim: int = 10
    seed: int = 42


@dataclass
class RLRecommendationResult:
    user_id: str
    item_ids: list[str]
    arm_indices: list[int]
    scores: list[float]
    strategy: str


class RLRecommendationAgent:
    """Bandit-based recommendation agent.

    Each candidate item is an arm.  The agent learns which items maximize
    reward (e.g. normalized rating) and balances exploration vs. exploitation
    via the chosen bandit strategy.
    """

    def __init__(
        self,
        item_ids: list[str],
        config: RLAgentConfig | None = None,
    ) -> None:
        if not item_ids:
            raise ValueError("item_ids must not be empty")
        self.item_ids = list(item_ids)
        self._item_idx: dict[str, int] = {iid: i for i, iid in enumerate(item_ids)}
        self.config = config or RLAgentConfig()
        self._bandit = self._build_bandit()

    def _build_bandit(self):
        n = len(self.item_ids)
        s = self.config.strategy
        if s == BanditStrategy.EPSILON_GREEDY:
            return EpsilonGreedyBandit(n, epsilon=self.config.epsilon, seed=self.config.seed)
        if s == BanditStrategy.UCB:
            return UCBBandit(n, c=self.config.ucb_c)
        if s == BanditStrategy.THOMPSON_SAMPLING:
            return ThompsonSamplingBandit(n, seed=self.config.seed)
        if s == BanditStrategy.LINUCB:
            cfg = LinUCBConfig(
                alpha=self.config.linucb_alpha,
                context_dim=self.config.context_dim,
            )
            return LinUCBBandit(n, cfg)
        raise ValueError(f"Unknown strategy: {s}")  # pragma: no cover

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        context: np.ndarray | None = None,
        exclude_items: set[str] | None = None,
    ) -> RLRecommendationResult:
        """Select top-n items via the bandit policy."""
        exclude = exclude_items or set()
        scores = self._bandit.estimated_values

        # Rank by estimated value, skipping excluded items
        ranked = sorted(
            [
                (i, float(scores[i]), iid)
                for i, iid in enumerate(self.item_ids)
                if iid not in exclude
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        top = ranked[: min(n, len(ranked))]

        return RLRecommendationResult(
            user_id=user_id,
            item_ids=[iid for _, _, iid in top],
            arm_indices=[i for i, _, _ in top],
            scores=[s for _, s, _ in top],
            strategy=self.config.strategy.value,
        )

    def update(
        self,
        item_id: str,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        """Record observed reward for an item."""
        arm = self._item_idx.get(item_id)
        if arm is None:
            logger.warning("Unknown item_id '%s' — skipping update", item_id)
            return
        if isinstance(self._bandit, LinUCBBandit):
            ctx = context if context is not None else np.zeros(self.config.context_dim)
            self._bandit.update(arm, ctx, reward)
        else:
            self._bandit.update(arm, reward)

    @property
    def estimated_item_values(self) -> dict[str, float]:
        vals = self._bandit.estimated_values
        return {iid: float(vals[i]) for i, iid in enumerate(self.item_ids)}
