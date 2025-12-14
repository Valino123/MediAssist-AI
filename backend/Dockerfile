# ===== Frontend build stage =====
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend

# Install deps with cache efficiency
COPY frontend/package*.json ./
RUN npm ci --prefer-offline --no-audit

# Build frontend
COPY frontend/ .
RUN npm run build

# ===== Backend runtime stage =====
FROM python:3.11-slim AS backend

# Project root inside container will mirror repo root layout:
# /app/backend -> FastAPI app
# /app/frontend/dist -> built SPA
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install backend dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r requirements.txt

# Copy backend source into /app/backend so path layout matches local dev
COPY backend/ /app/backend

# Copy built frontend assets into /app/frontend/dist for FastAPI static serving
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

# Run from backend directory so 'app.py' is the module root as in local dev
WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

