.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check ingest eval-retrieval eval serve docker-build docker-run

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install: ## Install dependencies and git hooks
	uv sync --all-extras
	uv run pre-commit install

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format and fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Static type checking (mypy --strict)
	uv run mypy src tests

test: ## Run the test suite with coverage
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test ## Everything CI runs

ingest: ## Build the vector index and the analytics database
	uv run meetrag ingest

eval-retrieval: ## Offline retrieval benchmark (no API key needed)
	uv run meetrag eval retrieval --min-mrr 0.90

eval: ## End-to-end evaluation with an LLM judge (needs an API key)
	uv run meetrag eval e2e

serve: ## Run the API and demo UI on http://localhost:8000
	uv run meetrag serve --reload

docker-build: ## Build the production image
	docker build -t meetrag:local .

docker-run: ## Run the production image on http://localhost:8080
	docker run --rm -p 8080:8080 --env-file .env meetrag:local
