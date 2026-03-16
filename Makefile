.DEFAULT_GOAL := help
.PHONY: help install sync lint format typecheck test test-verbose test-pkg clean clean-venv

# ── Variables ──────────────────────────────────────────────────────────────
PROJECT_NAME := lumis-eval
PYTHON       := uv run python

# ── Help ───────────────────────────────────────────────────────────────────
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────
install: ## Create venv and install all workspace packages
	@bash setup.sh

sync: ## Sync dependencies from lockfile (no install)
	uv sync

# ── Code quality ───────────────────────────────────────────────────────────
lint: ## Run ruff linter
	uv run ruff check .

format: ## Auto-format with ruff
	uv run ruff format .

typecheck: ## Run mypy across packages
	uv run mypy packages/

# ── Tests ──────────────────────────────────────────────────────────────────
test: ## Run all tests
	uv run pytest

test-verbose: ## Run all tests with verbose output
	uv run pytest -v

test-pkg: ## Run tests for a specific package. Usage: make test-pkg PKG=packages/lumiseval-core
	uv run pytest $(PKG)

# ── Dev servers ────────────────────────────────────────────────────────────
api: ## Start the FastAPI dev server
	uv run uvicorn lumiseval_api.main:app --reload --port 8080

# ── Clean ──────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist"          -exec rm -rf {} + 2>/dev/null || true

clean-venv: ## Remove virtual environment (force full reinstall)
	rm -rf .venv
