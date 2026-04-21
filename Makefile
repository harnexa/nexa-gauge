.DEFAULT_GOAL := help
.PHONY: help install sync lint lint-fix format typecheck test test-verbose test-pkg test_graph ci clean clean-venv api

# ── Variables ──────────────────────────────────────────────────────────────
PROJECT_NAME := nexa-gauge
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
lint: ## Run ruff linter (use FIX=1 to apply fixes)
	uv run ruff check $(if $(FIX),--fix,) .

lint-fix: ## Run ruff linter and apply safe fixes
	uv run ruff check --fix .

format: ## Auto-format with ruff
	uv run ruff format .

typecheck: ## Run mypy across packages
	uv run mypy packages/

# ── Tests ──────────────────────────────────────────────────────────────────
test: ## Run all tests
	VIRTUAL_ENV= uv run --all-packages --group dev -m pytest

test-verbose: ## Run all tests with verbose output
	VIRTUAL_ENV= uv run --all-packages --group dev -m pytest -v

test-pkg: ## Run tests for a specific package. Usage: make test-pkg PKG=packages/nexagauge-core
	VIRTUAL_ENV= uv run --all-packages --group dev -m pytest $(PKG)

test_graph: ## Run nexagauge-graph tests with output
	VIRTUAL_ENV= uv run --all-packages --group dev -m pytest -s packages/nexagauge-graph/test_ng_graph

# ── CI ─────────────────────────────────────────────────────────────────────
ci: ## Run full CI pipeline locally (format check → lint → test)
	@echo "==> format check"
	uv run ruff format --check .
	@echo "==> lint"
	uv run ruff check .
	@echo "==> test"
	uv run pytest
	@echo "==> CI passed"

# ── Deprecated targets ─────────────────────────────────────────────────────
api: ## Deprecated (API package removed)
	@echo "The API package was removed. Use CLI commands instead (e.g. 'nexagauge --help')."

# ── Clean ──────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist"          -exec rm -rf {} + 2>/dev/null || true

clean-venv: ## Remove virtual environment (force full reinstall)
	rm -rf .venv
