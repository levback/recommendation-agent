# Examples

Cross-references: [Get Started](user-guide/get-started.md) · [Configuration](configuration.md)

All examples are in `examples/`. Run them individually or all at once:

```bash
bash scripts/run_examples.sh
# outputs written to examples/output/
```

---

## 01 — Collaborative Filtering

**File:** `examples/01_collaborative_filtering.py`

Compares three CF algorithms on 50 users × 100 items × 1000 synthetic ratings.

```bash
python examples/01_collaborative_filtering.py
```

**Sample output:**

```
matrix_factorization: RMSE=0.9136  MAE=0.7594  recs=10
user_cf:              RMSE=1.0363  MAE=0.8338  recs=10
item_cf:              RMSE=1.1007  MAE=0.8978  recs=10
Results saved → examples/output/01_collaborative_filtering.json
```

Matrix factorization consistently achieves the lowest RMSE due to its global
latent factor representation.

---

## 02 — RL Bandits

**File:** `examples/02_rl_bandit.py`

Simulates 500 rounds of exploration across 5 arms for ε-Greedy, UCB1, and
Thompson Sampling. Also runs the `RLRecommendationAgent` with all four
strategies.

```bash
python examples/02_rl_bandit.py
```

**Sample output (truncated):**

```json
{
  "pure_bandits": {
    "epsilon_greedy": { "avg_reward": 0.782, "cumulative_reward": 391.0 },
    "ucb":            { "avg_reward": 0.636, "cumulative_reward": 318.0 },
    "thompson":       { "avg_reward": 0.778, "cumulative_reward": 389.0 }
  }
}
```

Thompson Sampling and ε-Greedy converge to the best arm fastest in this scenario.

---

## 03 — Hybrid Recommender

**File:** `examples/03_hybrid_recommender.py`

Blends CF and RL with three configurations, then simulates 10 online feedback
rounds per user.

```bash
python examples/03_hybrid_recommender.py
```

**Sample output:**

```
mf_thompson_60_40: RMSE=0.8970
user_cf_ucb_70_30: RMSE=1.0518
item_cf_no_rl:     RMSE=1.0557
```

---

## 04 — Bedrock Narration

**File:** `examples/04_bedrock_narration.py`

Generates a natural-language narrative for top-5 recommendations using Claude
Haiku 4.5 on Amazon Bedrock. Falls back to a mock LLM if credentials are
unavailable.

```bash
# With real Bedrock credentials:
AWS_PROFILE=your_profile python examples/04_bedrock_narration.py

# Without credentials (mock fallback):
python examples/04_bedrock_narration.py
```

**Sample output:**

```json
{
  "user_id": "1",
  "recommendations": [
    { "item_id": "2",  "score": 0.8 },
    { "item_id": "26", "score": 0.7376 }
  ],
  "narrative": "Based on your love of animation and comedy, Movie 2 and Movie 26 were chosen because they blend your top genres with strong thriller elements you've consistently rated highly.",
  "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0"
}
```

---

## 05 — Full MovieLens Scenario

**File:** `examples/05_movielens_scenario.py`

End-to-end scenario: load MovieLens (falls back to 2000 synthetic ratings if
the dataset is unavailable), compare all CF algorithms, run RL simulation,
evaluate the full pipeline with ranking metrics.

```bash
python examples/05_movielens_scenario.py
```

**Sample output (truncated):**

```json
{
  "dataset_stats": { "ratings": 2000, "movies": 200, "users": 100 },
  "cf_metrics": {
    "matrix_factorization": { "rmse": 0.8941, "mae": 0.7258 },
    "user_cf":              { "rmse": 1.0371, "mae": 0.8374 },
    "item_cf":              { "rmse": 0.9804, "mae": 0.7828 }
  },
  "ranking_metrics": {
    "precision@10": 0.0612,
    "recall@10":    0.0683,
    "ndcg@10":      0.0711
  }
}
```
