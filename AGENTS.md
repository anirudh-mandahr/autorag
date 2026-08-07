# AutoRAG — Agent Guide

Autonomous RAG pipeline optimizer: an LLM proposes `PipelineConfig` mutations, each is evaluated on a golden corpus, and the best config is kept under a fixed experiment/spend budget.

## Layout

| Path | Role |
|------|------|
| `src/autorag/config.py` | `PipelineConfig` (tunable knobs), `Settings` (env), search space, objective |
| `src/autorag/loop.py` | Research loop: propose → eval → keep/discard |
| `src/autorag/eval.py` | Golden-eval harness and metrics |
| `src/autorag/pipeline.py` | LCEL RAG chain (chunk → embed → retrieve → rerank → answer) |
| `src/autorag/store.py` | SQLite experiment history + config-hash dedup |
| `src/autorag/llm.py` | OpenRouter client (all LLM calls) |
| `src/autorag/embeddings.py` | Sentence-transformers embeddings |
| `src/autorag/vector_store.py` | `pgvector` (default) or `faiss` backend |
| `src/autorag/api.py` | FastAPI dashboard + query API |
| `program.md` | System prompt for the researcher agent |
| `data/` | Sample corpus, questions, golden eval JSONL |

## Commands

```bash
make setup      # uv sync + copy .env.example
make up         # docker compose (Postgres/pgvector)
make run-once   # single demo query
make eval       # golden eval (VECTOR_BACKEND=faiss)
make research   # full autoresearch loop
make serve      # FastAPI on :8000
make test       # pytest — must pass with no API key
```

## Invariants (do not break silently)

1. **`PipelineConfig` validation** — Pydantic bounds on numeric knobs (`chunk_size ≥ 50`, `retrieval_k ≥ 1`, `mmr_lambda` / `refusal_threshold` in `[0, 1]`, etc.). Invalid proposals must be rejected, not coerced.
2. **Budget stops** — `run_research` must halt when either `MAX_EXPERIMENTS` (session budget) or `MAX_USD_SPEND` is reached.
3. **Config dedup** — `propose_next_config` skips configs whose hash already exists in the experiment store (`store.hash_config` / `config_hash`).
4. **Eval stability** — `EvalHarness` on a frozen mini-corpus with stubbed LLM/embeddings must return deterministic metrics (no network in tests).

## Testing

- Run `make test` (or `uv run pytest`). Tests must pass **without** `OPENROUTER_API_KEY`.
- Mock all model calls in tests (`StubLLM`, `StubEmbeddings` in `tests/stubs.py`).
- Use `VECTOR_BACKEND=faiss` in tests to avoid Postgres.
- Critical coverage lives in:
  - `tests/test_config.py` — validation bounds
  - `tests/test_loop.py` — budget caps + dedup
  - `tests/test_eval.py` — harness integration on `tests/fixtures/`

## Conventions

- Tunable pipeline fields belong on `PipelineConfig`; runtime/env on `Settings`.
- All LLM traffic goes through `LLMClient`; embeddings through `EmbeddingClient`.
- Experiment objective: `compute_objective(metrics)` = weighted quality − 50 × `avg_cost_per_query`.
- Keep/discard: objective must **beat** current best to be kept.
- Prefer targeted 1–3 knob mutations per research proposal (`program.md`).

## Environment

Copy `.env.example` → `.env`. Key vars: `OPENROUTER_API_KEY`, `DATABASE_URL`, `VECTOR_BACKEND` (`pgvector` | `faiss`), `MAX_EXPERIMENTS`, `MAX_USD_SPEND`.
