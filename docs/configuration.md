# Configuration Reference

Cross-references: [Installation](installation.md) · [Architecture](architecture.md)

---

## `config/model_config.yaml`

```yaml
bedrock:
  default_model: us.anthropic.claude-haiku-4-5-20251001-v1:0
  region: us-east-1
  profile: null          # AWS named profile (null = use env vars / IAM role)

recommender:
  n_recommendations: 10
  cf_weight: 0.6         # CF contribution (RL = 1 - cf_weight)
  cf_method: matrix_factorization

matrix_factorization:
  n_factors: 50
  n_epochs: 100
  learning_rate: 0.01
  regularization: 0.01

bandit:
  strategy: thompson_sampling
  epsilon: 0.1           # ε-Greedy only
  alpha: 1.0             # LinUCB exploration factor
```

---

## `config/logging_config.yaml`

Configures `logging.config.dictConfig`. Two handlers:

- **console** — `INFO` level, plain text format
- **file** — `DEBUG` level, JSON lines to `data/logs/recommendation_agent.log`

Root logger level: `INFO`.

---

## Environment Variables

All variables are optional unless noted. Set them in `.env` (copied from `.env.example`).

### AWS / Bedrock

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | — | AWS access key (or use named profile) |
| `AWS_SECRET_ACCESS_KEY` | — | AWS secret key |
| `AWS_DEFAULT_REGION` | `eu-central-1` | Bedrock region |
| `AWS_PROFILE` | — | Named profile from `~/.aws/credentials` |
| `AWS_ROLE_ARN` | — | IAM role ARN to assume (EC2 / ECS) |

### Application

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Root log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `DATA_DIR` | `data/` | Root data directory |

---

## Python Config Dataclasses

### `MFConfig`

```python
@dataclass
class MFConfig:
    n_factors: int = 50
    n_epochs: int = 100
    learning_rate: float = 0.01
    regularization: float = 0.01
    rating_min: float = 1.0
    rating_max: float = 5.0
```

### `HybridConfig`

```python
@dataclass
class HybridConfig:
    cf_method: CFMethod = CFMethod.MATRIX_FACTORIZATION
    rl_strategy: BanditStrategy = BanditStrategy.THOMPSON_SAMPLING
    cf_weight: float = 0.6          # must be in [0, 1]
    use_rl: bool = True
    n_recommendations: int = 10
```

### `PipelineConfig`

```python
@dataclass
class PipelineConfig:
    n_recommendations: int = 10
    use_narration: bool = False
    hybrid_config: HybridConfig = field(default_factory=HybridConfig)
```

### `LLMConfig`

```python
@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
```
