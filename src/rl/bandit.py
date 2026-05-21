from __future__ import annotations

import numpy as np


class EpsilonGreedyBandit:
    """Epsilon-greedy multi-armed bandit."""

    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int = 42) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        self.n_arms = n_arms
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self._counts = np.zeros(n_arms)
        self._values = np.zeros(n_arms)

    def select_arm(self) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_arms))
        return int(np.argmax(self._values))

    def update(self, arm: int, reward: float) -> None:
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"arm must be in [0, {self.n_arms}), got {arm}")
        self._counts[arm] += 1
        self._values[arm] += (reward - self._values[arm]) / self._counts[arm]

    @property
    def estimated_values(self) -> np.ndarray:
        return self._values.copy()


class UCBBandit:
    """Upper Confidence Bound (UCB1) multi-armed bandit."""

    def __init__(self, n_arms: int, c: float = 2.0) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        self.n_arms = n_arms
        self.c = c
        self._counts = np.zeros(n_arms)
        self._values = np.zeros(n_arms)
        self._step = 0

    def select_arm(self) -> int:
        # Pull each arm once before applying UCB
        for arm in range(self.n_arms):
            if self._counts[arm] == 0:
                return arm
        ucb = self._values + self.c * np.sqrt(np.log(self._step) / self._counts)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"arm must be in [0, {self.n_arms}), got {arm}")
        self._step += 1
        self._counts[arm] += 1
        self._values[arm] += (reward - self._values[arm]) / self._counts[arm]

    @property
    def estimated_values(self) -> np.ndarray:
        return self._values.copy()


class ThompsonSamplingBandit:
    """Thompson Sampling using Beta distribution priors."""

    def __init__(self, n_arms: int, seed: int = 42) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        self.n_arms = n_arms
        self._rng = np.random.default_rng(seed)
        self._alpha = np.ones(n_arms)
        self._beta = np.ones(n_arms)

    def select_arm(self) -> int:
        samples = self._rng.beta(self._alpha, self._beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        """reward ≥ 0.5 → success; < 0.5 → failure."""
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"arm must be in [0, {self.n_arms}), got {arm}")
        if reward >= 0.5:
            self._alpha[arm] += 1.0
        else:
            self._beta[arm] += 1.0

    @property
    def estimated_values(self) -> np.ndarray:
        return self._alpha / (self._alpha + self._beta)
