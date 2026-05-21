# Installation Guide

Cross-references: [Get Started](user-guide/get-started.md) · [Configuration](configuration.md) · [Examples](examples.md)

---

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.12 |
| RAM | 4 GB | 8 GB |
| Disk | 500 MB | 2 GB (dataset cache) |
| OS | macOS 12+, Ubuntu 20.04+ | macOS 14+ arm64 / Ubuntu 22.04 |
| AWS account | — | Required for Bedrock narration |

---

## 1. Clone the repository

```bash
git clone git@github.com:levback/recommendation-agent.git
cd recommendation_agent
```

---

## 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

Or use the bootstrap script (also installs all dependencies):

```bash
bash scripts/setup_env.sh
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:

| Package | Purpose |
|---------|---------|
| `numpy`, `scipy` | Matrix factorization, bandit math |
| `pandas`, `scikit-learn` | Data processing, similarity metrics |
| `boto3` | Amazon Bedrock Converse API |
| `datasets` | HuggingFace MovieLens dataset |
| `pydantic` | Data validation and schemas |
| `pytest`, `pytest-cov` | Test runner with coverage |
| `bandit`, `safety` | Security scanning |

---

## 4. Configure AWS credentials (for Bedrock narration)

Narration via Claude Haiku 4.5 requires an AWS account with Bedrock access.
Credentials are resolved via the standard boto3 chain — **never hardcode them**.

### Option A — Environment variables

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=eu-central-1
```

### Option B — Named profile (`~/.aws/credentials`)

```ini
[recommendation-agent]
aws_access_key_id     = ...
aws_secret_access_key = ...
region                = us-east-1
```

Then set in `.env`:

```
AWS_PROFILE=recommendation-agent
```

### Option C — IAM Role (EC2 / ECS / Lambda)

Set `AWS_ROLE_ARN` in `.env`. The client will call `sts:AssumeRole` automatically.

### Enable Bedrock model access

In the AWS console → **Amazon Bedrock → Model access** → request access for
**Claude Haiku 4.5** (`us.anthropic.claude-haiku-4-5-20251001-v1:0`).

---

## 5. Copy and edit the environment file

```bash
cp .env.example .env
# Edit .env — only set values you need
```

---

## 6. Verify installation

```bash
# Run unit tests (no AWS credentials needed)
bash scripts/run_tests.sh

# Run all examples (Bedrock steps fall back gracefully without credentials)
bash scripts/run_examples.sh
```

Expected output from `run_tests.sh`:

```
200 passed, 97%+ coverage
```

---

## Docker (optional)

```bash
docker-compose up --build
```

AWS credentials must be passed via environment variables in `docker-compose.yml`
or a `.env` file — never baked into the image.
