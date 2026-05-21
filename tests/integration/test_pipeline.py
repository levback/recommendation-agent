"""Integration tests for the full recommendation pipeline."""
from __future__ import annotations

import pytest

from src.core.base import BaseLLM, LLMConfig, LLMResponse, Message
from src.datasets.loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor
from src.datasets.schemas import ItemProfile, Rating, UserProfile
from src.narration.narrator import RecommendationNarrator
from src.recommender.evaluator import RecommendationEvaluator
from src.recommender.hybrid import CFMethod, HybridConfig, HybridRecommender
from src.recommender.pipeline import PipelineConfig, RecommendationPipeline
from src.rl.agent import BanditStrategy, RLAgentConfig


# ─── Mock LLM (never calls AWS) ──────────────────────────────────────────────

class _MockLLM(BaseLLM):
    def __init__(self):
        super().__init__(LLMConfig(model="mock"))

    def complete(self, messages):
        return LLMResponse(content="These are great movies for you!", model=self.config.model)

    def stream(self, messages):
        yield "These are great movies for you!"


# ─── Shared data ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_dataset():
    loader = DatasetLoader()
    ratings, items, users = loader.generate_synthetic(
        n_users=30, n_items=50, n_ratings=500, seed=42
    )
    return ratings, items, users


@pytest.fixture(scope="module")
def split_dataset(synthetic_dataset):
    ratings, items, users = synthetic_dataset
    pp = DatasetPreprocessor(test_ratio=0.2, seed=42)
    ds = pp.split(ratings)
    return ds, items, users


# ─── Test: hybrid recommender with real data ─────────────────────────────────

class TestHybridRecommenderIntegration:
    def test_mf_recommend_all_users(self, split_dataset):
        ds, items, users = split_dataset
        hr = HybridRecommender(HybridConfig(cf_method=CFMethod.MATRIX_FACTORIZATION))
        hr.fit(ds.train)
        for uid in sorted({r.user_id for r in ds.train})[:5]:
            result = hr.recommend(uid, n=10)
            assert len(result.recommendations) <= 10

    def test_user_cf_recommend(self, split_dataset):
        ds, items, users = split_dataset
        hr = HybridRecommender(HybridConfig(cf_method=CFMethod.USER_CF, use_rl=False))
        hr.fit(ds.train)
        uid = sorted({r.user_id for r in ds.train})[0]
        result = hr.recommend(uid, n=5)
        assert len(result.recommendations) <= 5

    def test_item_cf_recommend(self, split_dataset):
        ds, items, users = split_dataset
        hr = HybridRecommender(HybridConfig(cf_method=CFMethod.ITEM_CF, use_rl=False))
        hr.fit(ds.train)
        uid = sorted({r.user_id for r in ds.train})[0]
        result = hr.recommend(uid, n=5)
        assert len(result.recommendations) <= 5

    def test_rl_feedback_loop(self, split_dataset):
        ds, items, users = split_dataset
        hr = HybridRecommender(
            HybridConfig(
                cf_method=CFMethod.MATRIX_FACTORIZATION,
                rl_strategy=BanditStrategy.THOMPSON_SAMPLING,
            )
        )
        hr.fit(ds.train)
        uid = sorted({r.user_id for r in ds.train})[0]
        # Simulate 5 rounds of feedback
        for _ in range(5):
            result = hr.recommend(uid, n=3)
            for item_id, _ in result.recommendations[:1]:
                hr.update_feedback(uid, item_id, reward=4.5)

    def test_evaluate_rmse_reasonable(self, split_dataset):
        ds, items, users = split_dataset
        ev = RecommendationEvaluator()
        hr = HybridRecommender(HybridConfig(cf_method=CFMethod.MATRIX_FACTORIZATION))
        hr.fit(ds.train)
        preds = [hr._cf.predict(r.user_id, r.item_id) for r in ds.test]
        actuals = [r.rating for r in ds.test]
        rmse = ev.rmse(actuals, preds)
        assert rmse < 4.0  # should do better than worst-case (4-point error)

    def test_all_strategies_combined(self, split_dataset):
        ds, items, users = split_dataset
        uid = sorted({r.user_id for r in ds.train})[0]
        for strategy in BanditStrategy:
            hr = HybridRecommender(
                HybridConfig(rl_strategy=strategy, context_dim=4)
            )
            hr.fit(ds.train)
            result = hr.recommend(uid, n=5)
            assert len(result.recommendations) <= 5


# ─── Test: full pipeline ─────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_end_to_end_no_narration(self, split_dataset):
        ds, items, users = split_dataset
        cfg = PipelineConfig(n_recommendations=10, use_narration=False)
        p = RecommendationPipeline(cfg)
        p.fit(ds.train, items=items, users=users)
        for uid in sorted({r.user_id for r in ds.train})[:3]:
            result = p.run(uid, n=5)
            assert len(result.recommendations) <= 5
            assert result.narrative is None

    def test_end_to_end_with_narration(self, split_dataset):
        ds, items, users = split_dataset
        narrator = RecommendationNarrator(_MockLLM())
        cfg = PipelineConfig(n_recommendations=5, use_narration=True)
        p = RecommendationPipeline(cfg, narrator=narrator)
        p.fit(ds.train, items=items, users=users)
        uid = sorted({r.user_id for r in ds.train})[0]
        result = p.run(uid, n=3)
        assert result.narrative is not None
        assert len(result.narrative) > 0

    def test_pipeline_evaluate(self, split_dataset):
        ds, items, users = split_dataset
        p = RecommendationPipeline()
        p.fit(ds.train)
        metrics = p.evaluate(ds.test)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    def test_pipeline_context_features(self, split_dataset):
        ds, items, users = split_dataset
        p = RecommendationPipeline()
        p.fit(ds.train, users=users)
        uid = sorted({r.user_id for r in ds.train})[0]
        result = p.run(uid, n=5, context={"action_pref": 1.0, "drama_pref": 0.5})
        assert isinstance(result.elapsed_seconds, float)
        assert result.elapsed_seconds >= 0

    def test_multiple_feedback_updates(self, split_dataset):
        ds, items, users = split_dataset
        p = RecommendationPipeline()
        p.fit(ds.train)
        uid = sorted({r.user_id for r in ds.train})[0]
        result = p.run(uid, n=5)
        for item_id, _ in result.recommendations:
            p.update_feedback(uid, item_id, reward=3.5)


# ─── Test: dataset loader → pipeline ─────────────────────────────────────────

class TestDatasetToPipeline:
    def test_synthetic_data_pipeline(self):
        loader = DatasetLoader()
        ratings, items, users = loader.generate_synthetic(
            n_users=20, n_items=30, n_ratings=200, seed=7
        )
        pp = DatasetPreprocessor(test_ratio=0.2, seed=7)
        ds = pp.split(ratings)
        p = RecommendationPipeline()
        p.fit(ds.train, items=items, users=users)
        m = p.evaluate(ds.test)
        assert m["rmse"] < 5.0

    def test_normalized_ratings_pipeline(self):
        loader = DatasetLoader()
        ratings, items, users = loader.generate_synthetic(
            n_users=20, n_items=30, n_ratings=150, seed=99
        )
        pp = DatasetPreprocessor()
        normed = pp.normalize_ratings(ratings)
        # All normalized ratings in [0, 1]
        for r in normed:
            assert 0.0 <= r.rating <= 1.0
