"""CLI entry point for the recommendation agent."""
from __future__ import annotations

import argparse
import json
import logging
import logging.config
import os
import sys

import yaml


def _setup_logging() -> None:
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "logging_config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(level=logging.INFO)


def _run_demo(args: argparse.Namespace) -> None:
    from src.datasets.loader import DatasetLoader
    from src.datasets.preprocessor import DatasetPreprocessor
    from src.recommender.hybrid import CFMethod, HybridConfig
    from src.recommender.pipeline import PipelineConfig, RecommendationPipeline

    logger = logging.getLogger(__name__)
    loader = DatasetLoader()

    if args.dataset == "movielens":
        logger.info("Loading HuggingFace MovieLens dataset (may fall back to synthetic)…")
        ratings, items, users = loader.load_huggingface()
    else:
        logger.info("Generating synthetic dataset…")
        ratings, items, users = loader.generate_synthetic(
            n_users=args.n_users, n_items=args.n_items, n_ratings=args.n_ratings
        )

    pp = DatasetPreprocessor(test_ratio=0.2, seed=42)
    ds = pp.split(ratings)
    logger.info("Train=%d  Test=%d  Users=%d  Items=%d", len(ds.train), len(ds.test), ds.n_users, ds.n_items)

    method_map = {
        "mf": CFMethod.MATRIX_FACTORIZATION,
        "user_cf": CFMethod.USER_CF,
        "item_cf": CFMethod.ITEM_CF,
    }
    cf_method = method_map.get(args.cf_method, CFMethod.MATRIX_FACTORIZATION)

    use_narration = args.narrate
    narrator = None
    if use_narration:
        try:
            from src.core.factory import create_bedrock_llm
            from src.narration.narrator import RecommendationNarrator
            narrator = RecommendationNarrator(create_bedrock_llm())
        except Exception as exc:
            logger.warning("Narration disabled (Bedrock unavailable): %s", exc)
            use_narration = False

    cfg = PipelineConfig(
        hybrid=HybridConfig(cf_method=cf_method, n_recommendations=args.n_recs),
        n_recommendations=args.n_recs,
        use_narration=use_narration,
    )
    pipeline = RecommendationPipeline(cfg, narrator=narrator)
    pipeline.fit(ds.train, items=items, users=users)

    metrics = pipeline.evaluate(ds.test)
    logger.info("Evaluation: %s", json.dumps(metrics, indent=2))

    uid = ds.users[0]
    result = pipeline.run(uid, n=args.n_recs)
    output = {
        "user_id": result.user_id,
        "recommendations": [{"item_id": iid, "score": round(s, 4)} for iid, s in result.recommendations],
        "narrative": result.narrative,
        "metrics": metrics,
        "elapsed_seconds": round(result.elapsed_seconds, 4),
        "method": result.method,
    }
    print(json.dumps(output, indent=2))


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(description="Recommendation Agent CLI")
    parser.add_argument("--dataset", choices=["movielens", "synthetic"], default="synthetic")
    parser.add_argument("--cf-method", choices=["mf", "user_cf", "item_cf"], default="mf")
    parser.add_argument("--n-recs", type=int, default=10)
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--n-ratings", type=int, default=2000)
    parser.add_argument("--narrate", action="store_true", help="Enable Bedrock LLM narration")
    args = parser.parse_args()
    _run_demo(args)


if __name__ == "__main__":
    main()
