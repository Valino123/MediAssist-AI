#!/usr/bin/env bash
set -euo pipefail

# Run backend test suite with verbose logging and JUnit XML output.
# Usage: ./backend/scripts/run_tests.sh

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p test-results

PYTHONPATH="$ROOT_DIR" \
pytest tests \
  -v \
  -s \
  --tb=short \
  --maxfail=1 \
  --junitxml="test-results/junit.xml" \
  --log-cli-level=INFO \
  --color=yes

echo "JUnit report: $ROOT_DIR/test-results/junit.xml"

