#!/usr/bin/env bash
set -euo pipefail

# Build frontend and start backend serving built assets
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Building frontend..."
cd "${PROJECT_ROOT}/frontend"
npm install
npm run build

echo "Starting backend on 0.0.0.0:8000..."
cd "${PROJECT_ROOT}/backend"
uvicorn app:app --host 0.0.0.0 --port 8000

