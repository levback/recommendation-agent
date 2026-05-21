from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LinUCBConfig:
    alpha: float = 1.0     # exploration parameter
    context_dim: int = 10  # dimensionality of context vectors


class LinUCBBandit:
    """LinUCB contextual bandit for personalized recommendations.

    Each arm maintains a ridge regression model; UCB is derived from the
    posterior variance of the regression coefficients.

    Reference: Li et al., "A Contextual-Bandit Approach to Personalized
    News Article Recommendation" (WWW 2010).
    """

    def __init__(self, n_arms: int, config: LinUCBConfig | None = None) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        self.n_arms = n_arms
        self.config = config or LinUCBConfig()
        d = self.config.context_dim
        # A_a: d×d matrix (initialized to identity), b_a: d-vector (initialized to 0)
        self._A = [np.eye(d) for _ in range(n_arms)]
        self._b = [np.zeros(d) for _ in range(n_arms)]
        self._step = 0

    def _theta(self, arm: int) -> np.ndarray:
        return np.linalg.solve(self._A[arm], self._b[arm])

    def select_arm(self, context: np.ndarray) -> int:
        """Select the arm with the highest UCB given context."""
        x = self._validate_context(context)
        ucb_values = np.empty(self.n_arms)
        for a in range(self.n_arms):
            theta = self._theta(a)
            A_inv = np.linalg.inv(self._A[a])
            ucb_values[a] = theta @ x + self.config.alpha * np.sqrt(x @ A_inv @ x)
        return int(np.argmax(ucb_values))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"arm must be in [0, {self.n_arms}), got {arm}")
        x = self._validate_context(context)
        self._A[arm] += np.outer(x, x)
        self._b[arm] += reward * x
        self._step += 1

    def _validate_context(self, context: np.ndarray) -> np.ndarray:
        x = np.asarray(context, dtype=float).flatten()
        d = self.config.context_dim
        if len(x) != d:
            raise ValueError(f"context must have {d} dimensions, got {len(x)}")
        return x

    @property
    def estimated_values(self) -> np.ndarray:
        """Estimated reward per arm at zero context (prior mean)."""
        zero = np.zeros(self.config.context_dim)
        return np.array([float(self._theta(a) @ zero) for a in range(self.n_arms)])
