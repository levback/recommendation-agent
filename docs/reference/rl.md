# Reference: `src/rl`

---

## Bandit Base Interface

All bandits expose:

```python
def select_arm(self) -> int: ...         # returns arm index
def update(self, arm: int, reward: float) -> None: ...  # reward ∈ [0, 1]
```

> **Important:** Counts are updated in `update()`, not in `select_arm()`.
> Always call `update()` after each selection.

---

## `EpsilonGreedyBandit`

```python
class EpsilonGreedyBandit:
    def __init__(self, n_arms: int, epsilon: float = 0.1) -> None
```

With probability `epsilon`: picks a random arm.
Otherwise: picks the arm with the highest estimated mean reward.

---

## `UCBBandit`

```python
class UCBBandit:
    def __init__(self, n_arms: int) -> None
```

Picks arm maximising $\hat{\mu}_i + \sqrt{2 \ln t / n_i}$.
Untried arms are always selected first.

---

## `ThompsonSamplingBandit`

```python
class ThompsonSamplingBandit:
    def __init__(self, n_arms: int) -> None
    # Internal state: _alpha[i], _beta[i]  (Beta distribution parameters)
```

Samples from `Beta(alpha_i, beta_i)` per arm and picks the highest sample.
Update rule: `alpha[arm] += reward`, `beta[arm] += (1 - reward)`.

---

## `LinUCBBandit`

```python
class LinUCBConfig:
    n_arms:      int
    context_dim: int
    alpha:       float = 1.0   # exploration coefficient

class LinUCBBandit:
    def __init__(self, config: LinUCBConfig) -> None

    def select_arm(self, context: np.ndarray) -> int
    def update(self, arm: int, reward: float, context: np.ndarray) -> None
```

Per-arm ridge regression model. Raises `ValueError` if context dimension
doesn't match `config.context_dim`.

---

## `BanditStrategy`

```python
class BanditStrategy(str, Enum):
    EPSILON_GREEDY    = "epsilon_greedy"
    UCB               = "ucb"
    THOMPSON_SAMPLING = "thompson_sampling"
    LINUCB            = "linucb"
```

---

## `RLRecommendationAgent`

```python
class RLAgentConfig:
    strategy:    BanditStrategy = BanditStrategy.THOMPSON_SAMPLING
    epsilon:     float          = 0.1
    context_dim: int            = 10
    alpha:       float          = 1.0

@dataclass
class RLRecommendationResult:
    user_id:         str
    recommendations: list[tuple[str, float]]   # (item_id, score)
    strategy:        str

class RLRecommendationAgent:
    def __init__(self, config: RLAgentConfig, item_ids: list[str]) -> None

    def recommend(self, user_id: str, n: int = 10) -> RLRecommendationResult
    def update(self, user_id: str, item_id: str, reward: float) -> None
    @property
    def estimated_item_values(self) -> dict[str, float]
```
