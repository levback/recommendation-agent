# Reference: Schemas (`src/datasets/schemas.py`)

---

## `Rating`

```python
@dataclass(frozen=True)
class Rating:
    user_id:   str
    item_id:   str
    rating:    float    # validated: must be in [0, 5], no NaN/inf
    timestamp: float = 0.0
```

**Validation (in `__post_init__`):**
- `user_id` and `item_id` must be non-empty strings
- `rating` must be finite and in `[0, 5]`

---

## `UserProfile`

```python
@dataclass(frozen=True)
class UserProfile:
    user_id:          str
    preferred_genres: list[str] = field(default_factory=list)
    avg_rating:       float     = 0.0
    n_ratings:        int       = 0
```

---

## `ItemProfile`

```python
@dataclass(frozen=True)
class ItemProfile:
    item_id:    str
    title:      str
    genres:     list[str] = field(default_factory=list)
    avg_rating: float     = 0.0
    n_ratings:  int       = 0
```

---

## `DatasetSplit`

```python
@dataclass
class DatasetSplit:
    train: list[Rating]
    test:  list[Rating]
    users: list[UserProfile] = field(default_factory=list)
    items: list[ItemProfile] = field(default_factory=list)
    n_users: int = field(init=False)   # computed from train
    n_items: int = field(init=False)   # computed from train
```

> **Note:** `n_users` / `n_items` count distinct IDs in the **train** split.
> `users` and `items` fields are only populated if profiles are passed to the
> preprocessor. To enumerate user IDs safely, use
> `sorted({r.user_id for r in ds.train})`.
