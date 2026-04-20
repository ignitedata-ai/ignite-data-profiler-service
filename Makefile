# AIPAL Backend Services Makefile

.PHONY: help install dev test lint format clean build up down logs shell init-db

# Default target
help:
	@echo "Available commands:"
	@echo "  dev-setup   - Initial setup for new developers (install deps, hooks, DB)"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Run development server"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean up generated files"
	@echo "  build       - Build Docker images"
	@echo "  up          - Start services with Docker Compose"
	@echo "  down        - Stop services"
	@echo "  logs        - View service logs"
	@echo "  shell       - Open shell in backend container"
	@echo "  migrate     - Run database migrations"
	@echo "  init-db     - Initialize database with seed data"
	@echo "  check       - Run all quality checks"

dev-setup:
	@echo "🚀 Setting up development environment..."
	@echo ""
	@echo "📦 Step 1/4: Installing dependencies..."
	uv sync --frozen --no-cache
	@echo ""
	@echo "🪝 Step 2/4: Installing pre-commit hooks..."
	uv run pre-commit install
	@echo ""
	@echo "🗄️  Step 3/4: Running database migrations..."
	uv run alembic upgrade head
	@echo ""
	@echo "✅ Development environment ready!"
	@echo "You can now run 'make dev' to start the development server."

install:
	uv sync --frozen --no-cache

dev:
	uv run python main.py

test:
	uv run pytest -v

test-cov:
	uv run pytest --cov=core --cov=api --cov-report=html --cov-report=term

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .mypy_cache
	rm -rf .ruff_cache


migrate:
	uv run alembic upgrade head

check: lint test
	@echo "All quality checks passed!"

