Param(
    [string]$ComposeFile
)

# Resolve compose path relative to this script if not provided
if (-not $ComposeFile) {
    $ComposeFile = Join-Path $PSScriptRoot "..\docker-compose.yml"
}

# Set Qdrant API key if not provided by the user
if (-not $env:QDRANT__SERVICE__API_KEY) {
    $env:QDRANT__SERVICE__API_KEY = "dev-qdrant-key"
}

# Enables BuildKit for faster builds; requires Docker Desktop
$env:DOCKER_BUILDKIT = "1"

Write-Host "Building images with docker compose using $ComposeFile ..."
docker compose -f $ComposeFile build

Write-Host "Starting services..."
docker compose -f $ComposeFile up -d

Write-Host "Done. Backend: http://localhost:8000  Qdrant: http://localhost:6333 (if exposed)"

