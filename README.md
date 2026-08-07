# AutoRAG

**Autoresearch for RAG:** an agent that self-optimizes a retrieval pipeline against a golden-eval harness under a fixed cost budget.

The premise is that RAG tuning is a search problem, not a craft. Instead of hand-tweaking chunk sizes and rerankers, AutoRAG has an LLM propose configuration mutations, scores each one on a frozen golden set, and keeps only the configs that measurably win. **Everything hinges on the eval harness** — see [Evaluation](#evaluation), which is the core of this project.

## Results (sample corpus, 3 experiments, $0.12 spend)

| Metric | Baseline | Best | Δ |
|--------|----------|------|---|
| **Objective** | 0.772 | **0.803** | +0.031 |
| Groundedness | 0.820 | **0.880** | +0.060 |
| Answerable recall | 0.875 | 0.875 | — |
| Expected source hit | 0.800 | 0.800 | — |
| Avg cost / query | $0.0018 | **$0.0016** | −$0.0002 |

**Winning mutation:** `chunk_size` 400→512, `rerank_strategy` cosine→mmr (everything else unchanged).

## How it works

```mermaid
flowchart LR
  A[program.md<br/>researcher prompt] --> B[LLM proposes<br/>PipelineConfig]
  B --> C[Golden eval<br/>harness]
  C --> D{Objective beats<br/>best-so-far?}
  D -->|yes| E[Keep config]
  D -->|no| F[Discard]
  E --> G{Budget left?}
  F --> G
  G -->|yes| B
  G -->|no| H[Best config → API / dashboard]
```

The loop mutates eight pipeline knobs (chunking, embeddings, retrieval *k*, reranking, prompts, refusal threshold), scores each config on a frozen golden set, and keeps only improvements until `MAX_EXPERIMENTS` or `MAX_USD_SPEND` is hit.

## Quick start

Requires **Python 3.11+**, [`uv`](https://docs.astral.sh/uv/), and Docker (only if you want the pgvector backend).

```bash
make setup             # uv sync + create .env from .env.example
# edit .env and add your OPENROUTER_API_KEY
make up                # start Postgres/pgvector (skip if using FAISS)
make research          # runs baseline + agent search (FAISS, no Postgres required)
make serve             # dashboard + live query API at http://localhost:8000
```

![AutoRAG dashboard — objective chart, leaderboard, live query](assets/dashboard.png)

---

## Evaluation

Evaluation is the heart of AutoRAG. The optimizer is only as trustworthy as the harness that scores it, so the eval is designed to be **deterministic, cheap, and hard to game** — an improvement in the objective has to reflect a real improvement in retrieval quality, not eval noise.

Run it standalone at any time:

```bash
make eval              # VECTOR_BACKEND=faiss uv run python -m autorag.eval
```

### The golden set

`data/golden.jsonl` holds 10 hand-written rows over a synthetic sci-fi corpus (`data/corpus.txt`). A synthetic corpus is deliberate: the facts appear nowhere in any pretraining data, so a model cannot answer from memory and every correct answer must come through retrieval.

Each row is a `GoldenRow`:

```json
{
  "question": "How long is Lyra-7's orbital period around Keth?",
  "expected_answer": "18 Earth days.",
  "expected_source_ids": ["corpus-chunk-0"],
  "answerable": true
}
```

Crucially, **2 of the 10 rows are unanswerable** (`"answerable": false`) — questions about margherita pizza and the 2020 Nobel Prize that the corpus cannot support. These are the trap rows. A pipeline that answers everything confidently scores well on recall and then loses hard on `correct_refusal_rate`, which keeps the optimizer from drifting toward a configuration that hallucinates fluently.

### Metrics

`EvalHarness.run()` executes every golden row through the pipeline and aggregates seven metrics:

| Metric | What it measures |
|--------|------------------|
| `answerable_recall` | Fraction of answerable questions the pipeline actually attempted (did not refuse) |
| `citation_rate` | Fraction of non-refused answers that cite at least one source |
| `expected_source_hit_rate` | Fraction of rows where an expected chunk was cited or retrieved — pure retrieval quality |
| `groundedness` | LLM-as-judge: is every claim in the answer supported by the retrieved context? |
| `correct_refusal_rate` | Fraction of unanswerable questions correctly refused |
| `avg_cost_per_query` | Mean USD spend per query |
| `avg_latency` | Mean wall-clock latency in milliseconds |

### The objective

The knobs trade off against each other — a larger `retrieval_k` usually buys groundedness but costs tokens — so the loop collapses the metrics into a single scalar that prices that trade-off explicitly:

```
objective = 0.20·answerable_recall
          + 0.10·citation_rate
          + 0.20·expected_source_hit_rate
          + 0.35·groundedness
          + 0.15·correct_refusal_rate
          − 50·avg_cost_per_query
```

Groundedness carries the most weight because a confidently wrong answer is the most expensive failure mode. The `50×` cost penalty is what makes the search economically honest: every additional **$0.001 per query must buy at least +0.05 of weighted quality** to be worth keeping. Since quality is capped at 1.0 and queries run around $0.0018, that penalty bites quickly — without it the agent would simply crank `retrieval_k` to the ceiling and declare victory.

Weights live in `QUALITY_WEIGHTS` and `COST_WEIGHT` in `src/autorag/config.py`. A candidate is kept only if it **strictly beats** the current best.

### Determinism and cost control

Two properties make eval results comparable across experiments:

- **Fixed seeds.** `set_deterministic_seeds(42)` pins `random` and `numpy` before every run, so retrieval and reranking are reproducible.
- **Cached judgments.** `GroundednessJudge` calls the LLM at `temperature=0.0` with a JSON-only response format and caches each verdict on disk under `.cache/eval/groundedness/<config_hash>/<question_hash>.json`. Re-running an identical config is free and returns byte-identical scores; only genuinely new configs cost money.

Because the judge is keyed by config hash, changing any pipeline knob correctly invalidates the cache — you never get a stale score attributed to a new config.

### Testing the harness

The eval harness is itself under test. `tests/test_eval.py` runs `EvalHarness` twice against a frozen mini-corpus (`tests/fixtures/`) with stubbed LLM and embedding clients, asserting both runs return byte-identical metrics with no network access:

```bash
make test              # OPENROUTER_API_KEY= uv run pytest
```

The full suite (40 tests) must pass **without** an API key. All model calls are mocked through `StubLLM` / `StubEmbeddings` in `tests/stubs.py`, and tests use `VECTOR_BACKEND=faiss` to avoid needing Postgres. Coverage is concentrated where silent breakage would be most damaging:

| File | Guards |
|------|--------|
| `tests/test_config.py` | `PipelineConfig` validation bounds — invalid proposals are rejected, never coerced |
| `tests/test_loop.py` | Budget caps (`MAX_EXPERIMENTS`, `MAX_USD_SPEND`) and config-hash dedup |
| `tests/test_eval.py` | Deterministic metrics, judge caching, metric aggregation |
| `tests/test_store.py` | SQLite experiment history |
| `tests/test_api.py` | Dashboard and query endpoints |

---

## Tuning search behavior

Edit [`program.md`](program.md) to change how the agent searches. It is loaded as the **system prompt** for every proposal: objective weights, mutation rules (1–3 knobs at a time), strategy hints, and output schema. Restart `make research` to pick up changes — no code edits required.

The tunable search space is defined by `get_search_space()` in `src/autorag/config.py`:

| Knob | Range / choices | Default |
|------|-----------------|---------|
| `chunk_size` | 50–800 | 400 |
| `chunk_overlap` | 0–300 | 80 |
| `embedding_model` | `all-MiniLM-L6-v2`, `all-mpnet-base-v2` | `all-MiniLM-L6-v2` |
| `retrieval_k` | 1–12 | 4 |
| `rerank_strategy` | `none`, `cosine`, `mmr` | `cosine` |
| `mmr_lambda` | 0.0–1.0 (1 = relevance, 0 = diversity) | 0.5 |
| `prompt_template_id` | `grounded_v1`, `grounded_concise` | `grounded_v1` |
| `refusal_threshold` | 0.0–1.0 | 0.25 |

## Commands

| Command | Description |
|---------|-------------|
| `make setup` | `uv sync` + create `.env` from the example |
| `make up` / `make down` | Start / stop Postgres + pgvector via Docker Compose |
| `make run-once` | Single demo query through the pipeline |
| `make eval` | Golden eval on the current config (FAISS) |
| `make research` | Full autoresearch loop (`BUDGET=15` by default) |
| `make serve` | FastAPI dashboard + query API on `:8000` |
| `make test` | pytest — must pass with no API key |

The research loop also has a CLI: `--once`, `--budget N`, `--fresh` (clear the experiment store first), and `--store PATH`.

## Configuration

Copy `.env.example` → `.env`. `.env` is gitignored — never commit your key.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | — | Required for live runs; all LLM traffic goes through OpenRouter |
| `LLM_MODEL` | `meta-llama/llama-3.3-70b-instruct` | Model used for answers, proposals, and the groundedness judge |
| `DATABASE_URL` | `postgresql://autorag:autorag@localhost:5432/autorag` | Postgres connection for the pgvector backend |
| `VECTOR_BACKEND` | `pgvector` | `pgvector` or `faiss` (FAISS needs no database) |
| `MAX_EXPERIMENTS` | `15` | Session experiment budget |
| `MAX_USD_SPEND` | `5.0` | Session spend cap |

The loop halts as soon as **either** budget is reached.

## API

`make serve` exposes the dashboard plus a small JSON API:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard: objective chart, leaderboard, live query box |
| `GET /health` | Liveness check |
| `GET /experiments` | Full experiment history, chronological |
| `GET /leaderboard` | Experiments ranked by objective |
| `GET /best` | Winning config and its metrics |
| `POST /query` | Answer a question using the best config |

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How long is Lyra-7'\''s orbital period around Keth?"}'
```

## Project layout

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
| `tests/` | Test suite, stubs, and frozen eval fixtures |
