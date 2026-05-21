# Reference: `src/recommender`

---

## `RecommendationEvaluator`

```python
class RecommendationEvaluator:
    def rmse(self, actuals: list[float], preds: list[float]) -> float
    def mae(self, actuals: list[float], preds: list[float]) -> float

    def precision_at_k(
        self,
        rec_ids:      list[str],
        relevant_ids: set[str],
        k:            int = 10,
    ) -> float

    def recall_at_k(
        self,
        rec_ids:      list[str],
        relevant_ids: set[str],
        k:            int = 10,
    ) -> float

    def ndcg_at_k(
        self,
        rec_ids:      list[str],
        relevant_ids: set[str],
        k:            int = 10,
    ) -> float

    def evaluate_all(
        self,
        actuals:      list[float],
        preds:        list[float],
        rec_ids:      list[str]        | None = None,
        relevant_ids: set[str]         | None = None,
        k:            int              = 10,
    ) -> dict[str, float]
```

`evaluate_all()` returns a dict with keys `rmse`, `mae`, and (if ranking args
provided) `precision@k`, `recall@k`, `ndcg@k`.

---

## `CFMethod`

```python
class CFMethod(str, Enum):
    MATRIX_FACTORIZATION = "matrix_factorization"
    USER_CF              = "user_cf"
    ITEM_CF              = "item_cf"
```

---

## `HybridConfig`

```python
@dataclass
class HybridConfig:
    cf_method:         CFMethod       = CFMethod.MATRIX_FACTORIZATION
    rl_strategy:       BanditStrategy = BanditStrategy.THOMPSON_SAMPLING
    cf_weight:         float          = 0.6     # ∈ [0, 1]
    use_rl:            bool           = True
    n_recommendations: int            = 10
```

---

## `RecommendationResult`

```python
@dataclass(frozen=True)
class RecommendationResult:
    user_id:         str
    recommendations: list[tuple[str, float]]   # (item_id, blended_score)
    method:          str
```

---

## `HybridRecommender`

```python
class HybridRecommender:
    def __init__(self, config: HybridConfig | None = None) -> None

    def fit(self, ratings: list[Rating]) -> None

    def recommend(self, user_id: str, n: int = 10) -> RecommendationResult
    # blended_score = cf_score_norm × cf_weight + rl_score_norm × (1 - cf_weight)

    def update_feedback(
        self,
        user_id: str,
        item_id: str,
        reward:  float,    # ∈ [1, 5]; normalised to [0, 1] internally
    ) -> None
```

---

## `PipelineConfig`

```python
@dataclass
class PipelineConfig:
    n_recommendations: int          = 10
    use_narration:     bool         = False
    hybrid_config:     HybridConfig = field(default_factory=HybridConfig)
```

---

## `PipelineResult`

```python
@dataclass(frozen=True)
class PipelineResult:
    user_id:         str
    recommendations: list[tuple[str, float]]
    narrative:       str | None
    method:          str
```

---

## `RecommendationPipeline`

```python
class RecommendationPipeline:
    def __init__(
        self,
        config:   PipelineConfig | None   = None,
        narrator: RecommendationNarrator | None = None,
    ) -> None

    def fit(
        self,
        ratings: list[Rating],
        items:   list[ItemProfile] | None = None,
        users:   list[UserProfile] | None = None,
    ) -> None

    def run(self, user_id: str, n: int = 10) -> PipelineResult

    def evaluate(self, test_ratings: list[Rating]) -> dict[str, float]
    # Returns {"rmse": ..., "mae": ...}

    def update_feedback(
        self,
        user_id: str,
        item_id: str,
        reward:  float,
    ) -> None
```
