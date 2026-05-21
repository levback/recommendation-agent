#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate 2>/dev/null || true

echo "=== Running all examples ==="
for script in examples/0*.py; do
  echo ""
  echo "--- $script ---"
  python "$script"
done
echo ""
echo "✓ All examples complete. Outputs in examples/output/"
