# Code Tutor AI - Backend

AI-based Python Algorithm Learning Platform Backend

## Tech Stack

- FastAPI (Python 3.11+)
- SQLAlchemy (Async PostgreSQL)
- Redis (Caching/Session)
- JWT Authentication
- Docker Sandbox (Code Execution)

## Setup

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Start server
uv run uvicorn code_tutor.main:app --reload
```

## Project Structure

```
src/code_tutor/
├── shared/          # Shared kernel (DB, Redis, Logging, DDD base classes)
├── identity/        # User & Authentication (Bounded Context)
├── learning/        # Problems & Submissions (Bounded Context)
├── tutor/           # AI Tutor Conversations (Bounded Context)
├── execution/       # Code Sandbox Execution (Bounded Context)
└── main.py          # FastAPI application entry point
```
