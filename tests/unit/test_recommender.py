"""Unit tests for recommender module (evaluator, hybrid, pipeline)."""
from __future__ import annotations

import pytest
import numpy as np

from src.core.base import BaseLLM, LLMConfig, LLMResponse, Message
from src.datasets.schemas import ItemProfile, Rating, UserProfile
from src.recommender.evaluator import RecommendationEvaluator
from src.recommender.hybrid import CFMethod, HybridConfig, HybridRecommender
from src.recommender.pipeline import PipelineConfig, PipelineResult, RecommendationPipeline
from src.rl.agent import BanditStrategy


# ─── Helper fixtures ─────────────────────────────────────────────────────────

def _ratings() -> list[Rating]:
    data = [
        ("u1","i1",5.0),("u1","i2",4.0),("u1","i3",2.0),
        ("u2","i1",4.0),("u2","i4",5.0),("u2","i5",3.0),
        ("u3","i2",5.0),("u3","i3",4.0),("u3","i6",3.0),
        ("u4","i1",3.0),("u4","i5",4.0),("u4","i7",5.0),
        ("u5","i2",4.0),("u5","i6",3.0),("u5","i8",5.0),
    ]
    return [Rating(u, i, r) for u, i, r in data]


def _items() -> list[ItemProfile]:
    return [
        ItemProfile(item_id=f"i{j}", title=f"Movie {j}", genres=("Action",))
        for j in range(1, 9)
    ]


def _users() -> list[UserProfile]:
    return [UserProfile(user_id=f"u{j}", features={"age": float(j * 5)}) for j in range(1, 6)]


class _MockLLM(BaseLLM):
    def __init__(self):
        super().__init__(LLMConfig(model="mock"))

    def complete(self, messages):
        return LLMResponse(content="Enjoy these picks!", model=self.config.model)

    def stream(self, messages):
        yield "Enjoy these picks!"


# ─── RecommendationEvaluator ─────────────────────────────────────────────────

class TestEvaluator:
    def test_rmse_perfect(self):
        ev = RecommendationEvaluator()
        assert ev.rmse([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_rmse_nonzero(self):
        ev = RecommendationEvaluator()
        assert ev.rmse([1.0, 3.0], [2.0, 2.0]) == pytest.approx(1.0)

    def test_mae_basic(self):
        ev = RecommendationEvaluator()
        assert ev.mae([1.0, 3.0], [2.0, 2.0]) == pytest.approx(1.0)

    def test_rmse_length_mismatch(self):
        ev = RecommendationEvaluator()
        with pytest.raises(ValueError):
            ev.rmse([1.0, 2.0], [1.0])

    def test_rmse_empty(self):
        ev = RecommendationEvaluator()
        with pytest.raises(ValueError):
            ev.rmse([], [])

    def test_precision_at_k_all_relevant(self):
        ev = RecommendationEvaluator()
        assert ev.precision_at_k(["a","b","c"], {"a","b","c"}, k=3) == pytest.approx(1.0)

    def test_precision_at_k_none_relevant(self):
        ev = RecommendationEvaluator()
        assert ev.precision_at_k(["a","b","c"], {"x","y"}, k=3) == pytest.approx(0.0)

    def test_recall_at_k_all_found(self):
        ev = RecommendationEvaluator()
        assert ev.recall_at_k(["a","b"], {"a","b"}, k=2) == pytest.approx(1.0)

    def test_recall_at_k_empty_relevant(self):
        ev = RecommendationEvaluator()
        assert ev.recall_at_k(["a","b"], set(), k=2) == pytest.approx(0.0)

    def test_ndcg_at_k_perfect(self):
        ev = RecommendationEvaluator()
        assert ev.ndcg_at_k(["a","b"], {"a","b"}, k=2) == pytest.approx(1.0)

    def test_ndcg_at_k_no_hit(self):
        ev = RecommendationEvaluator()
        assert ev.ndcg_at_k(["x","y"], {"a","b"}, k=2) == pytest.approx(0.0)

    def test_invalid_k(self):
        ev = RecommendationEvaluator()
        with pytest.raises(ValueError):
            ev.precision_at_k(["a"], {"a"}, k=0)
        with pytest.raises(ValueError):
            ev.recall_at_k(["a"], {"a"}, k=-1)
        with pytest.raises(ValueError):
            ev.ndcg_at_k(["a"], {"a"}, k=0)

    def test_evaluate_all(self):
        ev = RecommendationEvaluator()
        metrics = ev.evaluate_all(
            recommended=["a","b","c"],
            relevant={"a","b"},
            predicted=[3.5, 4.0],
            actual=[4.0, 4.5],
            k=3,
        )
        assert "precision@3" in metrics
        assert "recall@3" in metrics
        assert "ndcg@3" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics


# ─── HybridRecommender ───────────────────────────────────────────────────────

class TestHybridRecommender:
    @pytest.fixture
    def fitted_hybrid(self):
        hr = HybridRecommender(HybridConfig(cf_method=CFMethod.MATRIX_FACTORIZATION, n_recommendations=5))
        hr.fit(_ratings())
        return hr

    def test_fit_returns_self(self):
        hr = HybridRecommender()
        assert hr.fit(_ratings()) is hr

    def test_recommend_before_fit_raises(self):
        hr = HybridRecommender()
        with pytest.raises(RuntimeError):
            hr.recommend("u1")

    def test_fit_empty_raises(self):
        hr = HybridRecommender()
        with pytest.raises(ValueError):
            hr.fit([])

    def test_recommend_returns_n_items(self, fitted_hybrid):
        result = fitted_hybrid.recommend("u1", n=4)
        assert len(result.recommendations) == 4

    def test_recommend_sorted_by_score(self, fitted_hybrid):
        result = fitted_hybrid.recommend("u1", n=4)
        scores = [s for _, s in result.recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_all_cf_methods(self):
        for method in CFMethod:
            hr = HybridRecommender(HybridConfig(cf_method=method, n_recommendations=3))
            hr.fit(_ratings())
            result = hr.recommend("u1", n=3)
            assert len(result.recommendations) <= 3

    def test_recommend_no_rl(self):
        hr = HybridRecommender(HybridConfig(use_rl=False, n_recommendations=4))
        hr.fit(_ratings())
        result = hr.recommend("u1", n=4)
        assert result.rl_scores == {}

    def test_update_feedback(self, fitted_hybrid):
        # Should not raise
        fitted_hybrid.update_feedback("u1", "i1", reward=4.0)

    def test_update_feedback_no_rl_noop(self):
        hr = HybridRecommender(HybridConfig(use_rl=False))
        hr.fit(_ratings())
        hr.update_feedback("u1", "i1", reward=5.0)  # no-op

    def test_recommend_with_all_rl_strategies(self):
        for strat in BanditStrategy:
            hr = HybridRecommender(
                HybridConfig(rl_strategy=strat, n_recommendations=3, context_dim=4)
            )
            hr.fit(_ratings())
            result = hr.recommend("u1", n=3)
            assert len(result.recommendations) <= 3


# ─── RecommendationPipeline ──────────────────────────────────────────────────

class TestPipeline:
    @pytest.fixture
    def fitted_pipeline(self):
        cfg = PipelineConfig(
            n_recommendations=5,
            use_narration=False,
        )
        p = RecommendationPipeline(cfg)
        p.fit(_ratings(), items=_items(), users=_users())
        return p

    def test_fit_and_run(self, fitted_pipeline):
        result = fitted_pipeline.run("u1", n=4)
        assert isinstance(result, PipelineResult)
        assert len(result.recommendations) == 4
        assert result.narrative is None

    def test_run_before_fit_raises(self):
        p = RecommendationPipeline()
        with pytest.raises(RuntimeError):
            p.run("u1")

    def test_evaluate_returns_metrics(self, fitted_pipeline):
        test = [Rating("u1","i5",4.0), Rating("u2","i3",3.0)]
        m = fitted_pipeline.evaluate(test)
        assert "rmse" in m
        assert "mae" in m

    def test_evaluate_before_fit_raises(self):
        p = RecommendationPipeline()
        with pytest.raises(RuntimeError):
            p.evaluate([Rating("u1","i1",3.0)])

    def test_run_with_context(self, fitted_pipeline):
        result = fitted_pipeline.run("u1", n=3, context={"genre_pref": 1.0})
        assert len(result.recommendations) <= 3

    def test_update_feedback(self, fitted_pipeline):
        fitted_pipeline.update_feedback("u1", "i1", 5.0)

    def test_run_with_narration(self):
        from src.narration.narrator import RecommendationNarrator

        narrator = RecommendationNarrator(_MockLLM())
        cfg = PipelineConfig(use_narration=True)
        p = RecommendationPipeline(cfg, narrator=narrator)
        p.fit(_ratings(), items=_items(), users=_users())
        result = p.run("u1", n=3)
        assert result.narrative == "Enjoy these picks!"

    def test_run_narration_failure_graceful(self):
        from src.core.base import BaseLLM

        class _FailLLM(BaseLLM):
            def complete(self, _):
                raise RuntimeError("fail")

            def stream(self, _):
                raise RuntimeError("fail")
                yield

        from src.narration.narrator import RecommendationNarrator
        narrator = RecommendationNarrator(_FailLLM(LLMConfig(model="fail")))
        cfg = PipelineConfig(use_narration=True)
        p = RecommendationPipeline(cfg, narrator=narrator)
        p.fit(_ratings(), items=_items(), users=_users())
        result = p.run("u1", n=3)
        # Narration failure should be caught; result should still be valid
        assert isinstance(result, PipelineResult)
