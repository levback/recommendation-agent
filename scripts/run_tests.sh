#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate 2>/dev/null || true

echo "=== Running tests with coverage ==="
python -m pytest tests/ \
  --cov=src \
  --cov-report=term-missing \
  --cov-report=html:htmlcov \
  --cov-fail-under=95 \
  -v "$@"

echo ""
echo "✓ Coverage report: htmlcov/index.html"
