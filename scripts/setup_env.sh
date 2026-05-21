#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Setting up recommendation_agent environment ==="

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Environment ready. Activate with: source .venv/bin/activate"
echo "  Copy .env.example → .env and fill in AWS credentials."
