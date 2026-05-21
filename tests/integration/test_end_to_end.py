"""End-to-end tests: HuggingFace dataset → full recommendation scenario."""
from __future__ import annotations

import pytest

from src.core.base import BaseLLM, LLMConfig, LLMResponse, Message
from src.datasets.loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor
from src.narration.narrator import RecommendationNarrator
from src.recommender.evaluator import RecommendationEvaluator
from src.recommender.hybrid import CFMethod, HybridConfig, HybridRecommender
from src.recommender.pipeline import PipelineConfig, RecommendationPipeline
from src.rl.agent import BanditStrategy


class _MockLLM(BaseLLM):
    def __init__(self):
        super().__init__(LLMConfig(model="mock"))

    def complete(self, messages):
        return LLMResponse(content="AI narration placeholder", model=self.config.model)

    def stream(self, messages):
        yield "AI narration placeholder"


@pytest.fixture(scope="module")
def movielens_data():
    """Loads HuggingFace MovieLens or falls back to synthetic."""
    loader = DatasetLoader()
    try:
        ratings, items, users = loader.load_huggingface(dataset_name="nateraw/movie-lens-latest-small")
    except Exception:
        ratings, items, users = loader.generate_synthetic(n_users=50, n_items=100, n_ratings=1000, seed=42)
    return ratings, items, users


class TestEndToEndMovieLens:
    def test_data_loaded(self, movielens_data):
        ratings, items, users = movielens_data
        assert len(ratings) > 0
        assert len(items) > 0

    def test_split_and_train(self, movielens_data):
        ratings, items, users = movielens_data
        pp = DatasetPreprocessor(test_ratio=0.2, seed=42)
        ds = pp.split(ratings)
        assert len(ds.train) > len(ds.test)

    def test_mf_pipeline_rmse(self, movielens_data):
        ratings, items, users = movielens_data
        pp = DatasetPreprocessor(test_ratio=0.2, seed=42)
        ds = pp.split(ratings)
        p = RecommendationPipeline()
        p.fit(ds.train, items=items, users=users)
        metrics = p.evaluate(ds.test)
        assert metrics["rmse"] < 5.0
        assert metrics["mae"] < 5.0

    def test_top10_recommendations(self, movielens_data):
        ratings, items, users = movielens_data
        pp = DatasetPreprocessor(test_ratio=0.2, seed=42)
        ds = pp.split(ratings)
        cfg = PipelineConfig(n_recommendations=10, use_narration=False)
        p = RecommendationPipeline(cfg)
        p.fit(ds.train, items=items, users=users)
        uid = sorted({r.user_id for r in ds.train})[0]
        result = p.run(uid, n=10)
        assert len(result.recommendations) <= 10
        scores = [s for _, s in result.recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_narrated_recommendations(self, movielens_data):
        ratings, items, users = movielens_data
        pp = DatasetPreprocessor(test_ratio=0.2, seed=42)
        ds = pp.split(ratings)
        narrator = RecommendationNarrator(_MockLLM())
        cfg = PipelineConfig(n_recommendations=5, use_narration=True)
        p = RecommendationPipeline(cfg, narrator=narrator)
        p.fit(ds.train, items=items, users=users)
        uid = sorted({r.user_id for r in ds.train})[0]
        result = p.run(uid, n=5)
        assert result.narrative == "AI narration placeholder"

    def test_rl_online_learning(self, movielens_data):
        ratings, items, users = movielens_data
        pp = DatasetPreprocessor(test_ratio=0.2, seed=42)
        ds = pp.split(ratings)
        hr = HybridRecommender(
            HybridConfig(
                rl_strategy=BanditStrategy.EPSILON_GREEDY,
                cf_method=CFMethod.MATRIX_FACTORIZATION,
            )
        )
        hr.fit(ds.train)
        uid = sorted({r.user_id for r in ds.train})[0]
        for _round in range(10):
            result = hr.recommend(uid, n=5)
            for iid, _ in result.recommendations[:2]:
                hr.update_feedback(uid, iid, reward=4.0)

    def test_precision_recall_at_k(self, movielens_data):
        ratings, items, users = movielens_data
        pp = DatasetPreprocessor(test_ratio=0.2, seed=42)
        ds = pp.split(ratings)
        ev = RecommendationEvaluator()
        relev = pp.get_relevance_sets(ds.test, threshold=4.0)
        hr = HybridRecommender(HybridConfig(cf_method=CFMethod.MATRIX_FACTORIZATION))
        hr.fit(ds.train)
        precisions, recalls = [], []
        train_users = sorted({r.user_id for r in ds.train})[:10]
        for uid in train_users:
            if uid not in relev:
                continue
            result = hr.recommend(uid, n=10)
            rec_ids = [iid for iid, _ in result.recommendations]
            precisions.append(ev.precision_at_k(rec_ids, relev[uid], k=10))
            recalls.append(ev.recall_at_k(rec_ids, relev[uid], k=10))
        if precisions:
            avg_p = sum(precisions) / len(precisions)
            avg_r = sum(recalls) / len(recalls)
            assert avg_p >= 0.0
            assert avg_r >= 0.0

    def test_all_cf_methods_e2e(self, movielens_data):
        ratings, items, users = movielens_data
        pp = DatasetPreprocessor(test_ratio=0.2, seed=42)
        ds = pp.split(ratings)
        for method in CFMethod:
            hr = HybridRecommender(HybridConfig(cf_method=method, n_recommendations=5, use_rl=False))
            hr.fit(ds.train)
            uid = sorted({r.user_id for r in ds.train})[0]
            result = hr.recommend(uid, n=5)
            assert len(result.recommendations) <= 5
