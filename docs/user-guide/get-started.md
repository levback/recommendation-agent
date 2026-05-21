# Get Started

Cross-references: [Installation](../installation.md) · [Learn the Basics](learn-basics.md) · [Examples](../examples.md)

---

## Prerequisites

- Python 3.10+ installed
- Git
- (Optional) AWS account with Bedrock Claude Haiku 4.5 access for narration

---

## 1. Install

```bash
git clone git@github.com:levback/recommendation-agent.git
cd recommendation_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Run your first recommendation (no credentials needed)

```bash
python examples/01_collaborative_filtering.py
```

Expected output:

```
matrix_factorization: RMSE=0.9136  MAE=0.7594  recs=10
user_cf:              RMSE=1.0363  MAE=0.8338  recs=10
item_cf:              RMSE=1.1007  MAE=0.8978  recs=10
Results saved → examples/output/01_collaborative_filtering.json
```

---

## 3. Run the hybrid recommender

```bash
python examples/03_hybrid_recommender.py
```

This blends CF scores with RL bandit scores and simulates 10 online feedback
rounds per user.

---

## 4. Run with Bedrock narration (AWS credentials required)

```bash
cp .env.example .env
# Edit .env and set AWS_PROFILE or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY

python examples/04_bedrock_narration.py
```

If credentials are not set, the example falls back to a mock narrative automatically.

---

## 5. Run all examples at once

```bash
bash scripts/run_examples.sh
```

All JSON outputs land in `examples/output/`.

---

## 6. Use the pipeline in your own code

```python
from src.datasets.loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor
from src.recommender.pipeline import RecommendationPipeline, PipelineConfig

# Load data
loader = DatasetLoader()
ratings, items, users = loader.generate_synthetic(n_users=100, n_items=200, n_ratings=2000, seed=42)

# Split
pp = DatasetPreprocessor(test_ratio=0.2, seed=42)
ds = pp.split(ratings)

# Build and fit pipeline
pipeline = RecommendationPipeline(PipelineConfig(n_recommendations=10))
pipeline.fit(ds.train, items=items, users=users)

# Get recommendations
user_id = sorted({r.user_id for r in ds.train})[0]
result = pipeline.run(user_id, n=10)
for item_id, score in result.recommendations:
    print(f"  {item_id}: {score:.4f}")
```

---

## Next steps

- [Learn the Basics](learn-basics.md) — understand CF, bandits, and hybrid blending
- [Main Components](main-components.md) — detailed module guide
- [Configuration](../configuration.md) — tune hyperparameters
