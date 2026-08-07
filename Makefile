.PHONY: setup up down run-once research eval serve test

BUDGET ?= 15

setup:
	uv sync --all-groups
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
	VECTOR_BACKEND=faiss uv run python -m autorag.eval

serve:
	uv run uvicorn autorag.api:app --reload --host 0.0.0.0 --port 8000

test:
	OPENROUTER_API_KEY= uv run pytest
