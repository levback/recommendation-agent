"""Example 03: Hybrid Recommender (CF + RL blended)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor
from src.recommender.evaluator import RecommendationEvaluator
from src.recommender.hybrid import CFMethod, HybridConfig, HybridRecommender
from src.rl.agent import BanditStrategy


def main() -> None:
    loader = DatasetLoader()
    ratings, items, users = loader.generate_synthetic(
        n_users=50, n_items=100, n_ratings=1000, seed=7
    )
    pp = DatasetPreprocessor(test_ratio=0.2, seed=7)
    ds = pp.split(ratings)
    ev = RecommendationEvaluator()

    results = {}
    configs = [
        ("mf_thompson_60_40", HybridConfig(cf_method=CFMethod.MATRIX_FACTORIZATION, rl_strategy=BanditStrategy.THOMPSON_SAMPLING, cf_weight=0.6)),
        ("user_cf_ucb_70_30", HybridConfig(cf_method=CFMethod.USER_CF, rl_strategy=BanditStrategy.UCB, cf_weight=0.7, use_rl=True)),
        ("item_cf_no_rl", HybridConfig(cf_method=CFMethod.ITEM_CF, use_rl=False)),
    ]

    for label, cfg in configs:
        hr = HybridRecommender(cfg)
        hr.fit(ds.train)
        uid = ds.users[0]
        result = hr.recommend(uid, n=10)
        preds = [hr._cf.predict(r.user_id, r.item_id) for r in ds.test]
        actuals = [r.rating for r in ds.test]
        rmse = ev.rmse(actuals, preds)

        # Simulate 10 feedback rounds
        for _ in range(10):
            rec = hr.recommend(uid, n=3)
            for iid, _ in rec.recommendations[:1]:
                hr.update_feedback(uid, iid, reward=4.0)

        results[label] = {
            "rmse": round(rmse, 4),
            "top10": [{"item_id": iid, "score": round(s, 4)} for iid, s in result.recommendations],
            "method": result.method,
        }
        print(f"{label}: RMSE={rmse:.4f}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "output"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "output", "03_hybrid_recommender.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
