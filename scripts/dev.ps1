# Development startup script for Windows PowerShell
# Usage: .\scripts\dev.ps1

Write-Host "Code Tutor AI - Development Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if Docker is running
$dockerStatus = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Docker is not running. Starting services manually..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1: Start Docker Desktop and run:" -ForegroundColor Green
    Write-Host "  docker-compose up -d"
    Write-Host ""
    Write-Host "Option 2: Run services manually:" -ForegroundColor Green
    Write-Host ""
    Write-Host "Terminal 1 - Backend:" -ForegroundColor Cyan
    Write-Host "  cd backend"
    Write-Host "  uv run uvicorn code_tutor.main:app --reload"
    Write-Host ""
    Write-Host "Terminal 2 - Frontend:" -ForegroundColor Cyan
    Write-Host "  cd frontend"
    Write-Host "  npm run dev"
    Write-Host ""
    Write-Host "Note: Without PostgreSQL/Redis, some features won't work." -ForegroundColor Yellow
    Write-Host "Backend will use SQLite for development." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Starting Docker services..." -ForegroundColor Green

# Start database and redis
docker-compose up -d db redis

Write-Host ""
Write-Host "Waiting for services to be healthy..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if services are healthy
$dbHealth = docker inspect --format='{{.State.Health.Status}}' codetutor-db 2>&1
$redisHealth = docker inspect --format='{{.State.Health.Status}}' codetutor-redis 2>&1

Write-Host ""
Write-Host "Service Status:" -ForegroundColor Cyan
Write-Host "  PostgreSQL: $dbHealth"
Write-Host "  Redis: $redisHealth"

Write-Host ""
Write-Host "Run the following commands in separate terminals:" -ForegroundColor Green
Write-Host ""
Write-Host "Backend:" -ForegroundColor Cyan
Write-Host "  cd backend && uv run uvicorn code_tutor.main:app --reload"
Write-Host ""
Write-Host "Frontend:" -ForegroundColor Cyan
Write-Host "  cd frontend && npm run dev"
Write-Host ""
Write-Host "Access:" -ForegroundColor Green
Write-Host "  Frontend: http://localhost:5173"
Write-Host "  Backend API: http://localhost:8000"
Write-Host "  API Docs: http://localhost:8000/docs"
