# Main Components

Cross-references: [Learn the Basics](learn-basics.md) · [API Reference](../reference/index.md)

---

## `src/core` — LLM Abstractions

### `BaseLLM`

Abstract base class for all LLM backends. Implement `complete()` and `stream()`:

```python
class BaseLLM(ABC):
    def __init__(self, config: LLMConfig) -> None: ...
    def complete(self, messages: list[Message]) -> LLMResponse: ...
    def stream(self, messages: list[Message]) -> Iterator[str]: ...
    def chat(self, prompt: str, system: str = "") -> str: ...
```

### `BedrockClient`

Calls Claude Haiku 4.5 via the Bedrock Converse API.
Model: `us.anthropic.claude-haiku-4-5-20251001-v1:0`.
Credentials resolved via boto3 chain (never hardcoded).

```python
from src.core.bedrock_client import BedrockClient
client = BedrockClient(region="us-east-1")
response = client.chat("Recommend me a movie.")
```

### `create_bedrock_llm()`

Factory that reads region / profile / role_arn from environment variables:

```python
from src.core.factory import create_bedrock_llm
llm = create_bedrock_llm()
```

---

## `src/datasets` — Data Layer

### `DatasetLoader`

```python
loader = DatasetLoader()

# Try HuggingFace first, fall back to synthetic on any error
ratings, items, users = loader.load_huggingface("nateraw/movie-lens-latest-small")

# Generate synthetic data directly
ratings, items, users = loader.generate_synthetic(
    n_users=100, n_items=200, n_ratings=2000, seed=42
)
```

### `DatasetPreprocessor`

```python
pp = DatasetPreprocessor(test_ratio=0.2, seed=42)
ds = pp.split(ratings)               # → DatasetSplit
normed = pp.normalize_ratings(ds)    # min-max to [0, 1]
relev = pp.get_relevance_sets(ds.test, threshold=4.0)  # user → set of liked items
```

---

## `src/collaborative` — CF Algorithms

All algorithms implement `BaseCollaborativeFilter`:

```python
model.fit(train_ratings)
score = model.predict(user_id, item_id)    # → float
recs  = model.recommend(user_id, n=10)    # → list[(item_id, score)]
```

```python
from src.collaborative.matrix_factorization import MatrixFactorization, MFConfig
from src.collaborative.user_cf import UserBasedCF
from src.collaborative.item_cf import ItemBasedCF

mf = MatrixFactorization(MFConfig(n_factors=50, n_epochs=100))
```

---

## `src/rl` — Bandit Algorithms

### Pure bandits

```python
from src.rl.bandit import EpsilonGreedyBandit, UCBBandit, ThompsonSamplingBandit

bandit = ThompsonSamplingBandit(n_arms=100)
arm = bandit.select_arm()
bandit.update(arm, reward=1.0)     # reward ∈ [0, 1]
```

### `RLRecommendationAgent`

Higher-level agent that maps item IDs to bandit arms:

```python
from src.rl.agent import RLRecommendationAgent, RLAgentConfig, BanditStrategy

agent = RLRecommendationAgent(
    RLAgentConfig(strategy=BanditStrategy.THOMPSON_SAMPLING),
    item_ids=["movie_1", "movie_2", ...]
)
result = agent.recommend(user_id="u1", n=10)
agent.update(user_id="u1", item_id="movie_1", reward=0.8)
```

---

## `src/narration` — LLM Narration

### `RecommendationNarrator`

```python
from src.narration.narrator import RecommendationNarrator

narrator = RecommendationNarrator(llm, max_items_in_prompt=5)

result = narrator.narrate(
    user_id="u1",
    recommendations=[("movie_1", 0.9), ("movie_2", 0.8)],
    items_by_id={"movie_1": item_profile, ...},
)
print(result.narrative)   # LLM-generated explanation

# Explain a single item
explanation = narrator.explain_item(item_profile, user_profile=user_profile)
```

---

## `src/recommender` — Top-Level Orchestration

### `HybridRecommender`

```python
from src.recommender.hybrid import HybridRecommender, HybridConfig, CFMethod
from src.rl.agent import BanditStrategy

hr = HybridRecommender(HybridConfig(
    cf_method=CFMethod.MATRIX_FACTORIZATION,
    rl_strategy=BanditStrategy.THOMPSON_SAMPLING,
    cf_weight=0.6,
))
hr.fit(train_ratings)
result = hr.recommend("user_1", n=10)
hr.update_feedback("user_1", "movie_5", reward=4.0)
```

### `RecommendationPipeline`

End-to-end: fit → recommend → evaluate → update feedback.

```python
from src.recommender.pipeline import RecommendationPipeline, PipelineConfig

pipeline = RecommendationPipeline(PipelineConfig(n_recommendations=10), narrator=narrator)
pipeline.fit(train_ratings, items=items, users=users)
result   = pipeline.run("user_1", n=10)
metrics  = pipeline.evaluate(test_ratings)
pipeline.update_feedback("user_1", "movie_5", reward=4.0)
```

### `RecommendationEvaluator`

```python
from src.recommender.evaluator import RecommendationEvaluator

ev = RecommendationEvaluator()
print(ev.rmse(actuals, preds))
print(ev.precision_at_k(rec_ids, relevant_ids, k=10))
all_metrics = ev.evaluate_all(actuals, preds, rec_ids, relevant_ids)
```
