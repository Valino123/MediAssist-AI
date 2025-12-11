#!/usr/bin/env bash
set -euo pipefail

# Run backend only (use Vite dev server separately on port 5173)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}/backend"
uvicorn app:app --reload --host 0.0.0.0 --port 8000

