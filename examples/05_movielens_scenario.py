"""Example 05: Full MovieLens scenario using real HuggingFace data.

Downloads ashraq/movielens_ratings (~30 MB, 891k ratings) from HuggingFace Hub.
The first run downloads and caches to data/datasets/; subsequent runs are instant.

Set MAX_RATINGS to control experiment size (default 50_000 for a fast demo).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.base import BaseLLM, LLMConfig, LLMResponse, Message
from src.datasets.loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor
from src.narration.narrator import RecommendationNarrator
from src.recommender.evaluator import RecommendationEvaluator
from src.recommender.hybrid import CFMethod, HybridConfig, HybridRecommender
from src.recommender.pipeline import PipelineConfig, RecommendationPipeline
from src.rl.agent import BanditStrategy

# Cap for quick demo — set to None to use all 891k ratings
MAX_RATINGS = int(os.environ.get("MAX_RATINGS", "50000"))


class _MockNarrationLLM(BaseLLM):
    """Placeholder LLM used when Bedrock is not configured."""

    def complete(self, messages: list[Message]) -> LLMResponse:
        return LLMResponse(
            content=(
                "Here are your personalized movie picks! "
                "Based on your history with action-packed thrillers and thought-provoking dramas, "
                "these selections should resonate with your taste."
            ),
            model=self.config.model,
        )

    def stream(self, messages: list[Message]):
        yield self.complete(messages).content


def main() -> None:
    print("=== MovieLens Recommendation Scenario (real HuggingFace data) ===")

    # 1. Load dataset (cached after first download)
    loader = DatasetLoader()
    print(f"Loading dataset (max_ratings={MAX_RATINGS}, min_ratings_per_user=10)…")
    ratings, items, users = loader.load_huggingface(
        max_ratings=MAX_RATINGS,
        min_ratings_per_user=10,
    )
    print(f"  {len(ratings):,} ratings | {len(items):,} movies | {len(users):,} users")

    items_by_id = {it.item_id: it for it in items}

    # 2. Preprocess
    pp = DatasetPreprocessor(test_ratio=0.2, seed=42)
    ds = pp.split(ratings)
    print(f"  Train: {len(ds.train):,}  Test: {len(ds.test):,}")

    # 3. Evaluate all CF methods
    ev = RecommendationEvaluator()
    cf_metrics: dict[str, dict] = {}
    for method in CFMethod:
        hr = HybridRecommender(HybridConfig(cf_method=method, use_rl=False))
        hr.fit(ds.train)
        preds = [hr._cf.predict(r.user_id, r.item_id) for r in ds.test]
        actuals = [r.rating for r in ds.test]
        rmse = ev.rmse(actuals, preds)
        mae = ev.mae(actuals, preds)
        cf_metrics[method.value] = {"rmse": round(rmse, 4), "mae": round(mae, 4)}
        print(f"  CF={method.value}: RMSE={rmse:.4f}  MAE={mae:.4f}")

    # 4. RL online learning simulation
    print("\nSimulating RL online learning…")
    best_method = min(cf_metrics, key=lambda k: cf_metrics[k]["rmse"])
    hr = HybridRecommender(
        HybridConfig(
            cf_method=CFMethod(best_method),
            rl_strategy=BanditStrategy.THOMPSON_SAMPLING,
        )
    )
    hr.fit(ds.train)
    uid = sorted({r.user_id for r in ds.train})[0]
    for round_n in range(10):
        result = hr.recommend(uid, n=5)
        for iid, _ in result.recommendations[:2]:
            hr.update_feedback(uid, iid, reward=4.0)
    print(f"  Completed 10 RL feedback rounds for user {uid}.")

    # 5. Full pipeline with narration
    print("\nRunning full pipeline…")
    try:
        from src.core.factory import create_bedrock_llm
        llm = create_bedrock_llm()
        print("  Using AWS Bedrock (Claude Haiku 4.5).")
    except Exception:
        llm = _MockNarrationLLM(LLMConfig(model="mock-narration"))
        print("  Using mock narration LLM.")

    narrator = RecommendationNarrator(llm)
    cfg = PipelineConfig(n_recommendations=10, use_narration=True)
    pipeline = RecommendationPipeline(cfg, narrator=narrator)
    pipeline.fit(ds.train, items=items, users=users)

    result = pipeline.run(uid, n=10)
    pipeline_metrics = pipeline.evaluate(ds.test)

    # 6. Ranking metrics (precision/recall @10)
    relev = pp.get_relevance_sets(ds.test, threshold=4.0)
    ranking_metrics: dict[str, float] = {}
    train_users = sorted({r.user_id for r in ds.train})[:20]
    for u in train_users:
        if u not in relev:
            continue
        r = pipeline.run(u, n=10)
        rec_ids = [iid for iid, _ in r.recommendations]
        p = ev.precision_at_k(rec_ids, relev[u], k=10)
        r_val = ev.recall_at_k(rec_ids, relev[u], k=10)
        n_val = ev.ndcg_at_k(rec_ids, relev[u], k=10)
        for k, v in [("precision@10", p), ("recall@10", r_val), ("ndcg@10", n_val)]:
            ranking_metrics[k] = ranking_metrics.get(k, 0.0) + v / 20

    # 7. Output
    output = {
        "dataset_stats": {"ratings": len(ratings), "movies": len(items), "users": len(users)},
        "cf_metrics": cf_metrics,
        "pipeline_metrics": pipeline_metrics,
        "ranking_metrics": {k: round(v, 4) for k, v in ranking_metrics.items()},
        "sample_recommendations": {
            "user_id": result.user_id,
            "recommendations": [{"item_id": iid, "score": round(s, 4), "title": items_by_id.get(iid, type("", (), {"title": "?"})()).title} for iid, s in result.recommendations],
            "narrative": result.narrative,
        },
    }
    print(json.dumps(output, indent=2))

    os.makedirs(os.path.join(os.path.dirname(__file__), "output"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "output", "05_movielens_scenario.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
