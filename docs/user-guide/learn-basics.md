# Learn the Basics

Cross-references: [Overview](overview.md) · [Main Components](main-components.md)

---

## Ratings and the Data Model

Everything starts with a `Rating`:

```python
@dataclass(frozen=True)
class Rating:
    user_id: str      # opaque string identifier
    item_id: str      # opaque string identifier
    rating: float     # in [0, 5]; validated at construction
    timestamp: float  # Unix epoch (optional, defaults to 0.0)
```

Ratings are **immutable value objects** — you never modify them in place.

A `DatasetSplit` holds train and test sets:

```python
@dataclass
class DatasetSplit:
    train: list[Rating]
    test:  list[Rating]
    n_users: int       # computed from train
    n_items: int       # computed from train
```

> **Tip:** To get a list of user IDs from a split, use
> `sorted({r.user_id for r in ds.train})` — not `ds.users` which is a
> `list[UserProfile]` and may be empty if profiles were not provided.

---

## Collaborative Filtering

CF predicts ratings based on the patterns of many users.

### Matrix Factorization

Decomposes the rating matrix into user and item latent factors, plus bias terms:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i$$

where $\mu$ is the global mean, $b_u$ / $b_i$ are user/item biases, and
$\mathbf{p}_u$, $\mathbf{q}_i$ are learned embedding vectors.

Training uses stochastic gradient descent (SGD) with L2 regularisation.

### User-CF

Finds the K most similar users to the target user (cosine similarity on rating
vectors), then predicts by weighted average of their ratings:

$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N(u)} \text{sim}(u,v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N(u)} |\text{sim}(u,v)|}$$

### Item-CF

Uses adjusted cosine similarity (subtracts per-user mean before computing
cosine) to find similar items, then predicts for the target (user, item) pair.

---

## Reinforcement Learning Bandits

Bandits model the exploration-exploitation trade-off: should we recommend items
we know the user likes (exploit), or try new ones to learn their preferences
(explore)?

### ε-Greedy

With probability ε, pick a random item (explore). Otherwise pick the item with
the highest estimated reward (exploit).

### UCB1 (Upper Confidence Bound)

Pick the item $i$ that maximises:

$$\hat{\mu}_i + \sqrt{\frac{2 \ln t}{n_i}}$$

where $\hat{\mu}_i$ is the estimated reward, $t$ is total rounds, $n_i$ is arm
pull count. Items that haven't been tried much get a large bonus.

### Thompson Sampling

Maintains a Beta distribution $\text{Beta}(\alpha_i, \beta_i)$ per item.
Samples from each distribution and picks the highest sample. Naturally balances
exploration and exploitation through Bayesian uncertainty.

### LinUCB

Extends UCB to contextual settings. Each arm has a ridge regression model
$\theta_i$ fit on context vectors $x$, predicting reward as $x^\top \theta_i$.
The uncertainty bonus is $\alpha \sqrt{x^\top A_i^{-1} x}$.

---

## Hybrid Blending

The `HybridRecommender` scores every candidate item as:

$$\text{score}(u, i) = w_\text{CF} \cdot \hat{r}_{ui}^\text{norm} + (1 - w_\text{CF}) \cdot \hat{r}_{ui}^\text{RL}$$

Both scores are normalised to $[0, 1]$ before blending. Default $w_\text{CF} = 0.6$.

Online feedback updates only the RL layer:

```python
recommender.update_feedback(user_id, item_id, reward=4.0)
# reward is in [1, 5], normalised to [0, 1] internally
```

---

## Evaluation Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| RMSE | $\sqrt{\frac{1}{n}\sum(r - \hat{r})^2}$ | Rating prediction error |
| MAE  | $\frac{1}{n}\sum|r - \hat{r}|$ | Average absolute error |
| Precision@K | $\frac{\|relevant \cap top\text{-}K\|}{K}$ | Fraction of top-K that are relevant |
| Recall@K | $\frac{\|relevant \cap top\text{-}K\|}{\|relevant\|}$ | Fraction of relevant items found |
| NDCG@K | $\frac{DCG@K}{IDCG@K}$ | Ranking quality (position-aware) |
