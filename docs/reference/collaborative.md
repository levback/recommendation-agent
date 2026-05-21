# Reference: `src/collaborative`

---

## `BaseCollaborativeFilter`

Abstract base class. All CF algorithms implement these methods:

```python
class BaseCollaborativeFilter(ABC):
    def fit(self, ratings: list[Rating]) -> None: ...
    def predict(self, user_id: str, item_id: str) -> float: ...
    def recommend(self, user_id: str, n: int = 10) -> list[tuple[str, float]]: ...
    def fit_predict(self, train: list[Rating], test: list[Rating]) -> list[float]: ...
```

- `predict()` returns a raw rating score (clipped to `[rating_min, rating_max]`)
- `recommend()` excludes items already seen by the user; returns `[]` for unknown users
- Unknown users / items fall back to the global mean rating

---

## `MatrixFactorization`

```python
class MFConfig:
    n_factors:      int   = 50
    n_epochs:       int   = 100
    learning_rate:  float = 0.01
    regularization: float = 0.01
    rating_min:     float = 1.0
    rating_max:     float = 5.0

class MatrixFactorization(BaseCollaborativeFilter):
    def __init__(self, config: MFConfig | None = None) -> None
```

Biased SGD matrix factorization. Learns:
- User latent factors `P[u]` (shape: `n_factors`)
- Item latent factors `Q[i]` (shape: `n_factors`)
- User biases `bu[u]`
- Item biases `bi[i]`
- Global mean `mu`

Prediction: `clip(mu + bu[u] + bi[i] + P[u] · Q[i], rating_min, rating_max)`

---

## `UserBasedCF`

```python
class UserCFConfig:
    n_neighbors: int   = 20
    min_common:  int   = 3      # minimum co-rated items to be a neighbour
    rating_min:  float = 1.0
    rating_max:  float = 5.0

class UserBasedCF(BaseCollaborativeFilter):
    def __init__(self, config: UserCFConfig | None = None) -> None
```

Uses cosine similarity on rating vectors. The `_cosine_similarity()` method
handles sparse vectors efficiently by only computing over co-rated items.

---

## `ItemBasedCF`

```python
class ItemCFConfig:
    n_neighbors: int   = 20
    min_common:  int   = 3
    rating_min:  float = 1.0
    rating_max:  float = 5.0

class ItemBasedCF(BaseCollaborativeFilter):
    def __init__(self, config: ItemCFConfig | None = None) -> None
```

Uses **adjusted cosine similarity** — subtracts the user's mean rating before
computing cosine, which corrects for users who systematically rate high or low.
