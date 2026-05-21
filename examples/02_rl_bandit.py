"""Example 02: Reinforcement Learning bandits for exploration."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.datasets.loader import DatasetLoader
from src.rl.agent import BanditStrategy, RLAgentConfig, RLRecommendationAgent
from src.rl.bandit import EpsilonGreedyBandit, UCBBandit, ThompsonSamplingBandit
from src.rl.contextual_bandit import LinUCBBandit, LinUCBConfig


def simulate_bandit(bandit, n_rounds: int = 500, true_means: list[float] | None = None) -> dict:
    """Simulate a bandit with a Bernoulli environment."""
    rng = np.random.default_rng(42)
    n_arms = len(true_means) if true_means else 5
    true_means = true_means or [rng.uniform(0, 1) for _ in range(n_arms)]
    rewards = []
    for _ in range(n_rounds):
        if hasattr(bandit, "select_arm") and "context" in bandit.select_arm.__code__.co_varnames:
            arm = bandit.select_arm(np.ones(4))
        else:
            arm = bandit.select_arm()
        reward = float(rng.random() < true_means[arm])
        bandit.update(arm, reward) if not hasattr(bandit, "_A") else bandit.update(arm, np.ones(4), reward)
        rewards.append(reward)
    return {
        "avg_reward": round(float(np.mean(rewards)), 4),
        "cumulative_reward": round(float(np.sum(rewards)), 2),
        "estimated_values": [round(v, 4) for v in bandit.estimated_values],
    }


def main() -> None:
    true_means = [0.2, 0.5, 0.8, 0.4, 0.6]
    n_arms = len(true_means)

    results = {
        "epsilon_greedy": simulate_bandit(EpsilonGreedyBandit(n_arms=n_arms, epsilon=0.1), true_means=true_means),
        "ucb": simulate_bandit(UCBBandit(n_arms=n_arms), true_means=true_means),
        "thompson": simulate_bandit(ThompsonSamplingBandit(n_arms=n_arms), true_means=true_means),
    }

    # RL agent on synthetic items
    loader = DatasetLoader()
    ratings, items, users = loader.generate_synthetic(n_users=20, n_items=30, n_ratings=200, seed=0)
    item_ids = [it.item_id for it in items]

    rng = np.random.default_rng(42)
    agent_results = {}
    for strategy in BanditStrategy:
        cfg = RLAgentConfig(strategy=strategy, context_dim=4)
        agent = RLRecommendationAgent(item_ids, cfg)
        for _ in range(50):
            res = agent.recommend("u1", n=3)
            for iid in res.item_ids:
                agent.update(iid, reward=float(rng.uniform(0, 1)))
        rec = agent.recommend("u1", n=5)
        agent_results[strategy.value] = {
            "top5_items": rec.item_ids,
            "top5_scores": [round(s, 4) for s in rec.scores],
        }

    output = {"pure_bandits": results, "rl_agent_by_strategy": agent_results}
    os.makedirs(os.path.join(os.path.dirname(__file__), "output"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "output", "02_rl_bandit.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
