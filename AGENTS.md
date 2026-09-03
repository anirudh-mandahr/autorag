# AutoRAG — Agent Guide

Autonomous RAG pipeline optimizer: an LLM proposes `PipelineConfig` mutations, each is evaluated on a **development** golden split, and the best config is kept under a hard experiment/spend budget. Final numbers are reported on frozen held-out splits.

## Layout

| Path | Role |
|------|------|
| `src/autorag/config.py` | `PipelineConfig` (tunable knobs), `Settings` (env), search space, objective |
| `src/autorag/loop.py` | Research loop: propose → eval(dev) → keep/discard; held-out report |
| `src/autorag/eval.py` | Golden-eval harness, fingerprinted judges, distinct metrics |
| `src/autorag/budget.py` | Hard-cap reservation estimates and spend ledger |
| `src/autorag/pipeline.py` | LCEL RAG chain (chunk → embed → retrieve → rerank → answer) |
| `src/autorag/store.py` | SQLite experiment history + config-hash dedup |
| `src/autorag/llm.py` | OpenRouter client (all LLM calls) |
| `src/autorag/embeddings.py` | Sentence-transformers embeddings |
| `src/autorag/vector_store.py` | `pgvector` (default) or `faiss` backend |
| `src/autorag/api.py` | FastAPI dashboard + query API |
| `program.md` | System prompt for the researcher agent |
| `data/dev.jsonl` | Tuning / keep-discard set |
| `data/test.jsonl` | Frozen held-out test set |
| `data/adversarial.jsonl` | Adversarial refusal traps |
| `.github/workflows/ci.yml` | Offline format, types, tests, coverage |
| `.github/workflows/live-eval.yml` | Manual hosted-model eval artifact |

## Commands

```bash
make setup      # uv sync + copy .env.example
make up         # docker compose (Postgres/pgvector)
make run-once   # single demo query
make eval       # held-out golden eval (VECTOR_BACKEND=faiss)
make research   # full autoresearch loop
make serve      # FastAPI on :8000
make lint       # ruff + mypy
make test       # pytest — must pass with no API key
```

## Invariants (do not break silently)

1. **`PipelineConfig` validation** — Pydantic bounds on numeric knobs (`chunk_size ≥ 50`, `retrieval_k ≥ 1`, `mmr_lambda` / `refusal_threshold` in `[0, 1]`, etc.). Invalid proposals must be rejected, not coerced.
2. **Hard spend cap** — `run_research` must not *start* a researcher or eval call unless remaining `MAX_USD_SPEND` covers a conservative max-cost estimate (plus held-out holdback). Session spend is tracked separately for researcher, answering, and judging.
3. **Config dedup** — `propose_next_config` skips configs whose hash already exists in the experiment store (`store.hash_config` / `config_hash`).
4. **Eval cache fingerprint** — judge caches must include config, normalized question, answer, context digest, corpus digest, judge model/provider, prompt digest, and schema version. Changing any of those is a cache miss.
5. **Split isolation** — keep/discard uses `dev` only. Headline reporting uses `test` and `adversarial`, never the tuning rows.
6. **Distinct metrics** — retrieval hit@k / MRR / NDCG, citation precision / recall / claim-citation correctness, groundedness, and answer correctness are separate. A retrieved-but-uncited gold source is not a citation success.
7. **Eval stability (local only)** — `EvalHarness` on a frozen mini-corpus with stubbed LLM/embeddings must return deterministic metrics (no network in tests). Hosted models are not deterministic.

## Testing

- Run `make test` (or `uv run pytest`). Tests must pass **without** `OPENROUTER_API_KEY`.
- Mock all model calls in tests (`StubLLM`, `StubEmbeddings` in `tests/stubs.py`).
- Use `VECTOR_BACKEND=faiss` in tests to avoid Postgres.
- Critical coverage lives in:
  - `tests/test_config.py` — validation bounds
  - `tests/test_loop.py` — hard cap at proposal and eval boundaries + dedup
  - `tests/test_eval.py` — fingerprint cache invalidation, correctness cases, harness
  - `tests/test_budget.py` — reservation ledger

## Conventions

- Tunable pipeline fields belong on `PipelineConfig`; runtime/env on `Settings`.
- All LLM traffic goes through `LLMClient`; embeddings through `EmbeddingClient`.
- Experiment objective: `compute_objective(metrics)` = weighted quality − 50 × `avg_cost_per_query`.
- Keep/discard: objective must **beat** current best to be kept (dev split only).
- Prefer targeted 1–3 knob mutations per research proposal (`program.md`).

## Environment

Copy `.env.example` → `.env`. Key vars: `OPENROUTER_API_KEY`, `DATABASE_URL`, `VECTOR_BACKEND` (`pgvector` | `faiss`), `MAX_EXPERIMENTS`, `MAX_USD_SPEND` (hard session cap).
