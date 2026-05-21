#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate 2>/dev/null || true

echo "=== OWASP / NIST Security Checks ==="

echo ""
echo "--- Bandit static analysis ---"
bandit -r src/ -ll -ii --exit-zero

echo ""
echo "--- Safety dependency audit ---"
safety check --full-report || true

echo ""
echo "--- Security test suite ---"
python -m pytest tests/security/ -v

echo ""
echo "✓ Security checks complete."
