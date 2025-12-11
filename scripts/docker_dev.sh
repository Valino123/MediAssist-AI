#!/usr/bin/env bash
set -euo pipefail

# Build and run the stack (backend + qdrant) with Docker/Compose.
# Uses BuildKit for faster builds; run from any working directory.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"

export DOCKER_BUILDKIT=1

echo "Building images with docker compose..."
docker compose -f "${COMPOSE_FILE}" build

echo "Starting services..."
docker compose -f "${COMPOSE_FILE}" up -d

echo "Done. Backend: http://localhost:8000  Qdrant: http://localhost:6333 (if exposed)"

