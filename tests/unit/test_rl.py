"""Unit tests for reinforcement learning bandits and RL agent."""
from __future__ import annotations

import numpy as np
import pytest

from src.rl.bandit import EpsilonGreedyBandit, UCBBandit, ThompsonSamplingBandit
from src.rl.contextual_bandit import LinUCBBandit, LinUCBConfig
from src.rl.agent import BanditStrategy, RLAgentConfig, RLRecommendationAgent


# ─── EpsilonGreedyBandit ─────────────────────────────────────────────────────

class TestEpsilonGreedyBandit:
    def test_selects_arm_in_range(self):
        b = EpsilonGreedyBandit(n_arms=5)
        arm = b.select_arm()
        assert 0 <= arm < 5

    def test_update_changes_value(self):
        b = EpsilonGreedyBandit(n_arms=3)
        b.update(arm=0, reward=1.0)
        assert b.estimated_values[0] == pytest.approx(1.0)

    def test_exploits_best_arm(self):
        b = EpsilonGreedyBandit(n_arms=3, epsilon=0.0)
        b.update(0, 0.2)
        b.update(1, 0.8)
        b.update(2, 0.5)
        assert b.select_arm() == 1

    def test_invalid_n_arms(self):
        with pytest.raises(ValueError):
            EpsilonGreedyBandit(n_arms=0)

    def test_invalid_arm_update(self):
        b = EpsilonGreedyBandit(n_arms=3)
        with pytest.raises(ValueError):
            b.update(arm=10, reward=1.0)

    def test_estimated_values_length(self):
        b = EpsilonGreedyBandit(n_arms=4)
        assert len(b.estimated_values) == 4


# ─── UCBBandit ───────────────────────────────────────────────────────────────

class TestUCBBandit:
    def test_pulls_each_arm_first(self):
        b = UCBBandit(n_arms=3)
        arms_pulled = set()
        for _ in range(3):
            arm = b.select_arm()
            arms_pulled.add(arm)
            b.update(arm, reward=0.5)
        assert arms_pulled == {0, 1, 2}

    def test_update_and_estimate(self):
        b = UCBBandit(n_arms=2)
        b.update(0, 1.0)
        b.update(1, 0.0)
        b.update(0, 1.0)
        assert b.estimated_values[0] > b.estimated_values[1]

    def test_invalid_n_arms(self):
        with pytest.raises(ValueError):
            UCBBandit(n_arms=0)

    def test_invalid_arm_update(self):
        b = UCBBandit(n_arms=3)
        with pytest.raises(ValueError):
            b.update(arm=5, reward=0.5)


# ─── ThompsonSamplingBandit ──────────────────────────────────────────────────

class TestThompsonSamplingBandit:
    def test_selects_arm_in_range(self):
        b = ThompsonSamplingBandit(n_arms=4)
        arm = b.select_arm()
        assert 0 <= arm < 4

    def test_update_success(self):
        b = ThompsonSamplingBandit(n_arms=2)
        b.update(0, reward=1.0)
        # alpha should increase for arm 0
        assert b._alpha[0] > 1.0

    def test_update_failure(self):
        b = ThompsonSamplingBandit(n_arms=2)
        b.update(0, reward=0.0)
        assert b._beta[0] > 1.0

    def test_estimated_values_between_0_and_1(self):
        b = ThompsonSamplingBandit(n_arms=3)
        for v in b.estimated_values:
            assert 0.0 <= v <= 1.0

    def test_invalid_n_arms(self):
        with pytest.raises(ValueError):
            ThompsonSamplingBandit(n_arms=-1)

    def test_invalid_arm_update(self):
        b = ThompsonSamplingBandit(n_arms=3)
        with pytest.raises(ValueError):
            b.update(arm=99, reward=0.5)


# ─── LinUCBBandit ─────────────────────────────────────────────────────────────

class TestLinUCBBandit:
    def _ctx(self, dim=4):
        return np.ones(dim)

    def test_select_arm_with_context(self):
        b = LinUCBBandit(n_arms=4, config=LinUCBConfig(context_dim=4))
        arm = b.select_arm(self._ctx(4))
        assert 0 <= arm < 4

    def test_update_and_estimate(self):
        b = LinUCBBandit(n_arms=3, config=LinUCBConfig(context_dim=4))
        b.update(0, self._ctx(4), 1.0)
        b.update(0, self._ctx(4), 1.0)
        vals = b.estimated_values
        assert len(vals) == 3

    def test_wrong_context_dim_raises(self):
        b = LinUCBBandit(n_arms=3, config=LinUCBConfig(context_dim=4))
        with pytest.raises(ValueError):
            b.select_arm(np.ones(3))    # wrong dim

    def test_update_wrong_arm_raises(self):
        b = LinUCBBandit(n_arms=3, config=LinUCBConfig(context_dim=4))
        with pytest.raises(ValueError):
            b.update(5, self._ctx(4), 0.5)

    def test_update_wrong_context_dim_raises(self):
        b = LinUCBBandit(n_arms=3, config=LinUCBConfig(context_dim=4))
        with pytest.raises(ValueError):
            b.update(0, np.ones(2), 0.5)


# ─── RLRecommendationAgent ───────────────────────────────────────────────────

class TestRLRecommendationAgent:
    _ITEMS = [f"item_{i}" for i in range(10)]

    def test_recommend_returns_items(self):
        agent = RLRecommendationAgent(self._ITEMS, RLAgentConfig())
        result = agent.recommend("u1", n=5)
        assert len(result.item_ids) == 5
        assert all(iid in self._ITEMS for iid in result.item_ids)

    def test_recommend_excludes_items(self):
        agent = RLRecommendationAgent(self._ITEMS, RLAgentConfig())
        exclude = {"item_0", "item_1"}
        result = agent.recommend("u1", n=5, exclude_items=exclude)
        assert not exclude.intersection(result.item_ids)

    def test_recommend_all_strategies(self):
        for strategy in BanditStrategy:
            cfg = RLAgentConfig(strategy=strategy, context_dim=4)
            agent = RLRecommendationAgent(self._ITEMS, cfg)
            result = agent.recommend("u1", n=3)
            assert len(result.item_ids) <= 3

    def test_update_arm(self):
        agent = RLRecommendationAgent(self._ITEMS, RLAgentConfig())
        agent.update("item_0", reward=0.8)
        vals = agent.estimated_item_values
        assert "item_0" in vals

    def test_update_linucb_with_context(self):
        cfg = RLAgentConfig(strategy=BanditStrategy.LINUCB, context_dim=4)
        agent = RLRecommendationAgent(self._ITEMS, cfg)
        ctx = np.ones(4)
        agent.update("item_0", reward=0.8, context=ctx)

    def test_update_unknown_item(self):
        agent = RLRecommendationAgent(self._ITEMS, RLAgentConfig())
        # Should not raise; just skips unknown item
        agent.update("nonexistent_item", reward=0.5)

    def test_estimated_item_values_all_items(self):
        agent = RLRecommendationAgent(self._ITEMS, RLAgentConfig())
        vals = agent.estimated_item_values
        assert set(vals.keys()) == set(self._ITEMS)

    def test_recommend_n_larger_than_items(self):
        agent = RLRecommendationAgent(self._ITEMS, RLAgentConfig())
        result = agent.recommend("u1", n=1000)
        assert len(result.item_ids) <= len(self._ITEMS)

    def test_empty_item_list_raises(self):
        with pytest.raises(ValueError):
            RLRecommendationAgent([], RLAgentConfig())

    def test_recommend_result_scores_length(self):
        agent = RLRecommendationAgent(self._ITEMS, RLAgentConfig())
        result = agent.recommend("u1", n=4)
        assert len(result.scores) == len(result.item_ids)
