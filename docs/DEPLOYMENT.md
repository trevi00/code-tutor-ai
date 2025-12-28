# Code Tutor AI - 배포 및 운영

## Deployment, CI/CD, Monitoring

**버전**: 1.0
**최종 수정**: 2025-12-26

---

## 1. 인프라 아키텍처

### 1.1 배포 환경

```
┌─────────────────────────────────────────────────────────────────┐
│                     Production Environment                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                     Docker Host                          │   │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│   │   │  Nginx  │  │ Backend │  │ Backend │  │Frontend │   │   │
│   │   │  :80    │  │  :8000  │  │  :8001  │  │  :3000  │   │   │
│   │   │  :443   │  │  (GPU)  │  │  (CPU)  │  │         │   │   │
│   │   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │   │
│   │        │            │            │            │         │   │
│   │   ┌────┴────────────┴────────────┴────────────┴────┐   │   │
│   │   │              Docker Network                     │   │   │
│   │   └────┬────────────┬────────────┬─────────────────┘   │   │
│   │        │            │            │                      │   │
│   │   ┌────┴────┐  ┌────┴────┐  ┌────┴────┐               │   │
│   │   │PostgreSQL│  │  Redis  │  │ Sandbox │               │   │
│   │   │  :5432  │  │  :6379  │  │  Pool   │               │   │
│   │   └─────────┘  └─────────┘  └─────────┘               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Storage: /data (PostgreSQL, Redis, Models, Logs)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 하드웨어 요구사항

| 환경 | CPU | RAM | GPU | Storage |
|------|-----|-----|-----|---------|
| **Development** | 4 cores | 8GB | - | 50GB |
| **Staging** | 8 cores | 16GB | RTX 4050 (4GB) | 100GB |
| **Production** | 14 cores | 16GB | RTX 4050 (4GB) | 200GB |

---

## 2. Docker 구성

### 2.1 docker-compose.yml

```yaml
version: '3.9'

services:
  # ===================
  # Reverse Proxy
  # ===================
  nginx:
    image: nginx:alpine
    container_name: codetutor-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./frontend/dist:/usr/share/nginx/html:ro
    depends_on:
      - backend-gpu
      - backend-cpu
    restart: unless-stopped
    networks:
      - codetutor-network

  # ===================
  # Backend (GPU - AI)
  # ===================
  backend-gpu:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: codetutor-backend-gpu
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:${DB_PASSWORD}@db:5432/codetutor
      - REDIS_URL=redis://redis:6379/0
      - LLM_DEVICE=cuda
      - WORKER_TYPE=gpu
    volumes:
      - ./models:/app/models:ro
      - /var/run/docker.sock:/var/run/docker.sock
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - codetutor-network

  # ===================
  # Backend (CPU - General)
  # ===================
  backend-cpu:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: codetutor-backend-cpu
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:${DB_PASSWORD}@db:5432/codetutor
      - REDIS_URL=redis://redis:6379/0
      - LLM_DEVICE=cpu
      - WORKER_TYPE=cpu
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - codetutor-network

  # ===================
  # Database
  # ===================
  db:
    image: postgres:16-alpine
    container_name: codetutor-db
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=codetutor
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - codetutor-network

  # ===================
  # Cache
  # ===================
  redis:
    image: redis:7-alpine
    container_name: codetutor-redis
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - codetutor-network

  # ===================
  # Monitoring
  # ===================
  prometheus:
    image: prom/prometheus:latest
    container_name: codetutor-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=15d'
    restart: unless-stopped
    networks:
      - codetutor-network

  grafana:
    image: grafana/grafana:latest
    container_name: codetutor-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - codetutor-network

  # ===================
  # Log Aggregation
  # ===================
  loki:
    image: grafana/loki:latest
    container_name: codetutor-loki
    volumes:
      - ./monitoring/loki-config.yml:/etc/loki/local-config.yaml:ro
      - loki_data:/loki
    command: -config.file=/etc/loki/local-config.yaml
    restart: unless-stopped
    networks:
      - codetutor-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
  loki_data:

networks:
  codetutor-network:
    driver: bridge
```

### 2.2 Nginx 설정

```nginx
# nginx/nginx.conf

upstream backend_gpu {
    server backend-gpu:8000;
}

upstream backend_cpu {
    server backend-cpu:8000;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/m;

server {
    listen 80;
    server_name codetutor.ai;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name codetutor.ai;

    # SSL
    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # Security Headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000" always;

    # Frontend (Static)
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;

        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff2)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # API (General)
    location /api/ {
        limit_req zone=api burst=20 nodelay;

        proxy_pass http://backend_cpu;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # AI Endpoints (GPU)
    location ~ ^/api/v1/(ai|code/review|problems/recommended) {
        limit_req zone=api burst=5 nodelay;

        proxy_pass http://backend_gpu;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Longer timeout for AI
        proxy_read_timeout 30s;
    }

    # Auth (Stricter rate limit)
    location /api/v1/auth/ {
        limit_req zone=auth burst=3 nodelay;

        proxy_pass http://backend_cpu;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket (Chat)
    location /ws/ {
        proxy_pass http://backend_gpu;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;

        proxy_read_timeout 86400;
    }

    # Health Check
    location /health {
        access_log off;
        return 200 "OK";
    }
}
```

### 2.3 Backend Dockerfile

```dockerfile
# backend/Dockerfile

# Stage 1: Build
FROM python:3.11-slim as builder

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser alembic/ alembic/
COPY --chown=appuser:appuser alembic.ini ./

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')"

# Run
CMD ["sh", "-c", "alembic upgrade head && uvicorn src.code_tutor.main:app --host 0.0.0.0 --port 8000"]
```

---

## 3. CI/CD 파이프라인

### 3.1 GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml

name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ===================
  # Backend Tests
  # ===================
  backend-test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v1

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: |
          cd backend
          uv sync --frozen

      - name: Run linting
        run: |
          cd backend
          uv run ruff check src/
          uv run mypy src/

      - name: Run unit tests
        run: |
          cd backend
          uv run pytest tests/unit -v --cov=src --cov-report=xml

      - name: Run integration tests
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:test@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379/0
        run: |
          cd backend
          uv run pytest tests/integration -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: backend/coverage.xml

  # ===================
  # Frontend Tests
  # ===================
  frontend-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        run: |
          cd frontend
          npm ci

      - name: Run linting
        run: |
          cd frontend
          npm run lint

      - name: Run tests
        run: |
          cd frontend
          npm test -- --coverage

      - name: Build
        run: |
          cd frontend
          npm run build

  # ===================
  # Security Scan
  # ===================
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Python security audit
        run: |
          cd backend
          pip install pip-audit
          pip-audit -r requirements.txt || true

      - name: npm audit
        run: |
          cd frontend
          npm audit --audit-level=high || true

      - name: Trivy vulnerability scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'CRITICAL,HIGH'

  # ===================
  # Build & Push Docker
  # ===================
  build:
    needs: [backend-test, frontend-test, security]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Backend
        uses: docker/build-push-action@v5
        with:
          context: ./backend
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/backend:${{ github.sha }}

      - name: Build and push Frontend
        uses: docker/build-push-action@v5
        with:
          context: ./frontend
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/frontend:${{ github.sha }}

  # ===================
  # Deploy to Staging
  # ===================
  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging

    steps:
      - name: Deploy to Staging
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.STAGING_HOST }}
          username: ${{ secrets.STAGING_USER }}
          key: ${{ secrets.STAGING_SSH_KEY }}
          script: |
            cd /opt/codetutor
            docker compose pull
            docker compose up -d --force-recreate
            docker system prune -f

  # ===================
  # E2E Tests
  # ===================
  e2e-test:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install Playwright
        run: |
          cd frontend
          npm ci
          npx playwright install --with-deps

      - name: Run E2E tests
        env:
          BASE_URL: ${{ secrets.STAGING_URL }}
        run: |
          cd frontend
          npx playwright test

  # ===================
  # Deploy to Production
  # ===================
  deploy-prod:
    needs: e2e-test
    runs-on: ubuntu-latest
    environment: production

    steps:
      - name: Deploy to Production
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.PROD_HOST }}
          username: ${{ secrets.PROD_USER }}
          key: ${{ secrets.PROD_SSH_KEY }}
          script: |
            cd /opt/codetutor
            docker compose pull
            docker compose up -d --force-recreate --no-deps backend-gpu backend-cpu
            sleep 10
            docker compose exec backend-gpu curl -f http://localhost:8000/health
            docker system prune -f
```

### 3.2 Rollback 전략

```bash
#!/bin/bash
# scripts/rollback.sh

set -e

PREVIOUS_VERSION=$(docker images --format "{{.Tag}}" | grep -v latest | head -2 | tail -1)

echo "Rolling back to version: $PREVIOUS_VERSION"

docker compose down
docker tag ghcr.io/user/codetutor/backend:$PREVIOUS_VERSION ghcr.io/user/codetutor/backend:latest
docker compose up -d

echo "Rollback complete. Verifying health..."
sleep 10
curl -f http://localhost:8000/health
```

---

## 4. 모니터링

### 4.1 Prometheus 설정

```yaml
# monitoring/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: []

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: 'backend'
    static_configs:
      - targets: ['backend-gpu:8000', 'backend-cpu:8000']
    metrics_path: '/metrics'

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

### 4.2 Alert Rules

```yaml
# monitoring/alerts.yml

groups:
  - name: codetutor
    rules:
      # High Error Rate
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
          / sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # Slow Response Time
      - alert: SlowResponseTime
        expr: |
          histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow response time"
          description: "95th percentile latency is {{ $value }}s"

      # High Memory Usage
      - alert: HighMemoryUsage
        expr: |
          (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes)
          / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"

      # GPU Memory Usage
      - alert: HighGPUMemory
        expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory nearly exhausted"

      # Database Connection Pool
      - alert: DatabaseConnectionPoolExhausted
        expr: pg_stat_activity_count > 80
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool nearly exhausted"
```

### 4.3 Application Metrics

```python
# src/code_tutor/shared/infrastructure/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# AI metrics
LLM_INFERENCE_TIME = Histogram(
    'llm_inference_duration_seconds',
    'LLM inference time',
    buckets=[1.0, 2.0, 5.0, 10.0, 30.0]
)

LLM_TOKENS_GENERATED = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated by LLM'
)

# Code execution metrics
CODE_EXECUTION_TIME = Histogram(
    'code_execution_duration_seconds',
    'Code execution time in sandbox',
    ['verdict'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

ACTIVE_SANDBOXES = Gauge(
    'active_sandboxes',
    'Number of active sandbox containers'
)

# Business metrics
SUBMISSIONS_TOTAL = Counter(
    'submissions_total',
    'Total code submissions',
    ['verdict']
)

ACTIVE_USERS = Gauge(
    'active_users',
    'Number of active users in last 5 minutes'
)
```

### 4.4 Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Code Tutor AI",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (endpoint)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))"
          }
        ]
      },
      {
        "title": "LLM Inference Time (p95)",
        "type": "gauge",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(llm_inference_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Submissions by Verdict",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum(submissions_total) by (verdict)"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes * 100"
          }
        ]
      }
    ]
  }
}
```

---

## 5. 로깅

### 5.1 Structured Logging

```python
# src/code_tutor/shared/infrastructure/logging.py

import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
logger.info(
    "submission_created",
    user_id=str(user_id),
    problem_id=str(problem_id),
    code_length=len(code)
)

logger.error(
    "llm_inference_failed",
    error=str(e),
    model="EEVE-Korean-2.8B",
    retry_count=3
)
```

### 5.2 Log Aggregation (Loki)

```yaml
# monitoring/loki-config.yml

auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
  chunk_idle_period: 5m
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2024-01-01
      store: boltdb
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 168h

storage_config:
  boltdb:
    directory: /loki/index
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s
```

---

## 6. 백업 및 복구

### 6.1 자동 백업

```bash
#!/bin/bash
# scripts/backup.sh

set -e

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# PostgreSQL
docker exec codetutor-db pg_dump -U postgres codetutor | gzip > $BACKUP_DIR/db.sql.gz

# Redis
docker exec codetutor-redis redis-cli BGSAVE
docker cp codetutor-redis:/data/dump.rdb $BACKUP_DIR/redis.rdb

# Models (if changed)
tar -czf $BACKUP_DIR/models.tar.gz /opt/codetutor/models

# Upload to S3 (optional)
aws s3 sync $BACKUP_DIR s3://codetutor-backups/$(date +%Y%m%d)/

# Cleanup old backups (keep 7 days)
find /backups -type d -mtime +7 -exec rm -rf {} +

echo "Backup completed: $BACKUP_DIR"
```

### 6.2 복구 절차

```bash
#!/bin/bash
# scripts/restore.sh

set -e

BACKUP_DATE=$1

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: ./restore.sh YYYYMMDD"
    exit 1
fi

BACKUP_DIR="/backups/$BACKUP_DATE"

# Stop services
docker compose stop backend-gpu backend-cpu

# Restore PostgreSQL
gunzip -c $BACKUP_DIR/db.sql.gz | docker exec -i codetutor-db psql -U postgres codetutor

# Restore Redis
docker cp $BACKUP_DIR/redis.rdb codetutor-redis:/data/dump.rdb
docker restart codetutor-redis

# Restart services
docker compose up -d backend-gpu backend-cpu

echo "Restore completed from: $BACKUP_DIR"
```

---

## 7. 운영 체크리스트

### 7.1 배포 전

- [ ] 모든 테스트 통과
- [ ] 보안 스캔 완료
- [ ] DB 마이그레이션 검토
- [ ] 환경 변수 확인
- [ ] Rollback 계획 수립

### 7.2 배포 후

- [ ] Health check 확인
- [ ] 로그 모니터링
- [ ] 주요 기능 수동 테스트
- [ ] 메트릭 대시보드 확인
- [ ] 알림 채널 확인

### 7.3 정기 운영

- [ ] 주간: 로그 검토, 디스크 정리
- [ ] 월간: 의존성 업데이트, 보안 패치
- [ ] 분기: 성능 리뷰, 백업 복구 테스트
