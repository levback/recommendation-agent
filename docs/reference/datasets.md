# Reference: `src/datasets`

---

## `DatasetLoader`

```python
class DatasetLoader:
    def load_huggingface(
        self,
        dataset_name: str = "nateraw/movie-lens-latest-small",
    ) -> tuple[list[Rating], list[ItemProfile], list[UserProfile]]
```

Attempts to load via HuggingFace `datasets`. Falls back to `generate_synthetic()`
on any error (missing dataset, network failure, `trust_remote_code` rejection).

```python
    def generate_synthetic(
        self,
        n_users:   int   = 100,
        n_items:   int   = 200,
        n_ratings: int   = 2000,
        seed:      int   = 42,
    ) -> tuple[list[Rating], list[ItemProfile], list[UserProfile]]
```

Generates preference-aware synthetic ratings across 15 genres.
Each (user, item) pair is unique. Returns consistent results for the same seed.

---

## `DatasetPreprocessor`

```python
class DatasetPreprocessor:
    def __init__(self, test_ratio: float = 0.2, seed: int = 42) -> None
```

### `split()`

```python
def split(
    self,
    ratings: list[Rating],
    users:   list[UserProfile] | None = None,
    items:   list[ItemProfile] | None = None,
) -> DatasetSplit
```

Shuffles ratings with `seed` and splits into train / test sets by `test_ratio`.
If `users` / `items` are provided they are stored on the returned `DatasetSplit`.

### `normalize_ratings()`

```python
def normalize_ratings(self, ds: DatasetSplit) -> DatasetSplit
```

Applies min-max normalisation to scale all ratings to `[0, 1]`.
Returns a new `DatasetSplit` — does not mutate the original.

### `get_relevance_sets()`

```python
def get_relevance_sets(
    self,
    ratings:   list[Rating],
    threshold: float = 4.0,
) -> dict[str, set[str]]
```

Returns a mapping of `user_id → set of item_ids` where rating ≥ threshold.
Used for computing Precision@K, Recall@K, NDCG@K.
