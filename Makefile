.PHONY: setup up down run-once research eval serve test lint

BUDGET ?= 15

setup:
	uv sync --all-groups --extra faiss
	@test -f .env || cp .env.example .env

up:
	docker compose up -d

down:
	docker compose down

run-once:
	uv run python -m autorag.loop --once

research:
	VECTOR_BACKEND=faiss uv run python -m autorag.loop --budget $(BUDGET) --fresh

eval:
	VECTOR_BACKEND=faiss uv run python -m autorag.eval --split held-out

serve:
	uv run uvicorn autorag.api:app --reload --host 0.0.0.0 --port 8000

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src

test:
	OPENROUTER_API_KEY= VECTOR_BACKEND=faiss uv run pytest --cov=autorag --cov-report=term-missing
