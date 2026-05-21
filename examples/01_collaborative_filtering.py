"""Example 01: Collaborative Filtering on synthetic data."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.collaborative.matrix_factorization import MatrixFactorization, MFConfig
from src.collaborative.user_cf import UserBasedCF
from src.collaborative.item_cf import ItemBasedCF
from src.datasets.loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor
from src.recommender.evaluator import RecommendationEvaluator


def main() -> None:
    loader = DatasetLoader()
    ratings, items, users = loader.generate_synthetic(
        n_users=50, n_items=100, n_ratings=1000, seed=42
    )
    pp = DatasetPreprocessor(test_ratio=0.2, seed=42)
    ds = pp.split(ratings)
    ev = RecommendationEvaluator()

    results = {}
    for name, model in [
        ("matrix_factorization", MatrixFactorization(MFConfig(n_factors=20, n_epochs=20))),
        ("user_cf", UserBasedCF()),
        ("item_cf", ItemBasedCF()),
    ]:
        model.fit(ds.train)
        preds = [model.predict(r.user_id, r.item_id) for r in ds.test]
        actuals = [r.rating for r in ds.test]
        rmse = ev.rmse(actuals, preds)
        mae = ev.mae(actuals, preds)
        uid = ds.users[0]
        recs = model.recommend(uid, n=10)
        results[name] = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "top10_for_user0": [{"item_id": iid, "score": round(s, 4)} for iid, s in recs],
        }
        print(f"{name}: RMSE={rmse:.4f}  MAE={mae:.4f}  recs={len(recs)}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "output"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "output", "01_collaborative_filtering.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
