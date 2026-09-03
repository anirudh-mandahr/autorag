# AutoRAG

**Autoresearch for RAG:** an agent that proposes retrieval-pipeline mutations and scores them on a development eval set under a **hard** USD cap. Final performance is reported on frozen held-out splits, not on the data used to pick the winner.

The premise is that RAG tuning is a search problem. AutoRAG has an LLM propose `PipelineConfig` mutations, scores each one on `data/dev.jsonl`, and keeps only configs that beat the current best. **The optimizer is only as trustworthy as the harness.** This repo no longer treats cached judge scores, a ten-row mix of train-and-test, or `temperature=0` as guarantees of correctness or determinism.

## Results (what we can actually claim)

The previous sample table (objective 0.772 → 0.803 on ten mixed rows, $0.12 spend) is **withdrawn**. It tuned and reported on the same items, mixed retrieval with citation, did not score expected answers, and treated a stale groundedness cache as a frozen truth.

### Offline (CI, stubbed providers)

These numbers come from `EvalHarness` on the frozen mini-corpus in `tests/fixtures/` with `StubLLM` / `StubEmbeddings`. They prove the evaluator is stable locally. They are **not** evidence that a hosted model improved.

| Metric | Mini-corpus (n=2) |
|--------|-------------------|
| Answerable recall | 1.000 |
| Retrieval hit@k | 1.000 |
| Retrieval MRR | 1.000 |
| Retrieval NDCG | 1.000 |
| Citation precision | 1.000 |
| Citation recall | 1.000 |
| Claim-citation correctness | 1.000 |
| Groundedness | 1.000 |
| Answer correctness | 1.000 |
| Correct refusal rate | 1.000 |

n=2 is tiny. Bootstrap intervals over two examples are not a generalization claim. CI re-runs this harness on every push.

### Live hosted-model results

Not published in this README. Hosted inference is not deterministic, even at `temperature=0`. To produce a versioned artifact:

1. Store `OPENROUTER_API_KEY` as a GitHub Actions secret.
2. Run the **Live eval** workflow (`.github/workflows/live-eval.yml`).
3. Download `results/eval-<split>-<timestamp>.json`.

That artifact records model id, provider, prompt/corpus/config digests, raw judge output, timestamps, and whether each verdict was cached.

## How it works

```mermaid
flowchart LR
  A[program.md<br/>researcher prompt] --> B[LLM proposes<br/>PipelineConfig]
  B --> C[Dev-split eval]
  C --> D{Dev objective beats<br/>best-so-far?}
  D -->|yes| E[Keep config]
  D -->|no| F[Discard]
  E --> G{Hard budget left?}
  F --> G
  G -->|yes| B
  G -->|no| H[Held-out test + adversarial report]
```

Keep/discard uses **dev only**. After search stops, the winner is scored on `data/test.jsonl` and `data/adversarial.jsonl` if the reserved budget remains. Those held-out numbers are not fed back into selection.

## Quick start

Requires **Python 3.11+**, [`uv`](https://docs.astral.sh/uv/), and Docker (only if you want the pgvector backend).

```bash
make setup             # uv sync + create .env from .env.example
# edit .env and add your OPENROUTER_API_KEY
make up                # start Postgres/pgvector (skip if using FAISS)
make research          # baseline + agent search (FAISS, no Postgres required)
make serve             # dashboard + live query API at http://localhost:8000
```

![AutoRAG dashboard — objective chart, leaderboard, live query](assets/dashboard.png)

---

## Evaluation

Run it standalone:

```bash
make eval              # VECTOR_BACKEND=faiss; reports held-out test + adversarial
```

### Splits

A synthetic sci-fi corpus (`data/corpus.txt`) is still used so facts are unlikely to be memorized. Rows are **not** one pile:

| Split | File | Role |
|-------|------|------|
| Development | `data/dev.jsonl` | Tuning / keep-discard (6 answerable + 1 unanswerable) |
| Test | `data/test.jsonl` | Frozen held-out reporting (4 answerable + 1 unanswerable) |
| Adversarial | `data/adversarial.jsonl` | Refusal traps that mention corpus entities or look answerable (5 unanswerable) |

Selecting a config on ten mixed rows and quoting those same rows as the result overfits the evaluator even if the LLM never sees `expected_answer`.

Each row is a `GoldenRow`:

```json
{
  "question": "How long is Lyra-7's orbital period around Keth?",
  "expected_answer": "18 Earth days.",
  "expected_source_ids": ["corpus-chunk-0"],
  "answerable": true
}
```

### Metrics

`EvalHarness.run()` scores each split separately. Retrieval and citation are not merged. Groundedness and correctness are not merged.

| Metric | What it measures |
|--------|------------------|
| `answerable_recall` | Fraction of answerable questions not refused |
| `retrieval_hit_at_k` | Gold source appears in `retrieved_ids` |
| `retrieval_mrr` | Reciprocal rank of the first gold source in the retrieved list |
| `retrieval_ndcg` | NDCG with binary relevance on retrieved ranks |
| `citation_precision` | Fraction of **cited** ids that are gold sources |
| `citation_recall` | Fraction of gold sources that were **cited** (retrieved-but-uncited is 0) |
| `claim_citation_correctness` | Fraction of cited ids that are both retrieved and gold. Citation-id-level, not NLP claim extraction |
| `groundedness` | Judge: are the answer's claims supported by retrieved context? |
| `answer_correctness` | Exact-fact containment or a separately versioned correctness judge vs `expected_answer` |
| `correct_refusal_rate` | Fraction of unanswerable questions refused |
| `avg_cost_per_query` | Mean USD (answering + judging) per query |
| `avg_latency` | Mean wall-clock latency in milliseconds |

A response can be grounded and wrong, or correct and unsupported. Those cases are tested independently.

Headline reports include **bootstrap 95% CIs over examples** in the split. That is sampling uncertainty for a small set, not a claim that hosted-model draws are stable. Optional `--repeats N` measures run-to-run variance (and costs N times). A handful of experiments on a handful of rows is not evidence of a general improvement.

### The objective (development split only)

```
objective = 0.10·answerable_recall
          + 0.15·retrieval_hit_at_k
          + 0.10·citation_precision
          + 0.10·citation_recall
          + 0.20·groundedness
          + 0.20·answer_correctness
          + 0.15·correct_refusal_rate
          − 50·avg_cost_per_query
```

Weights live in `QUALITY_WEIGHTS` and `COST_WEIGHT` in `src/autorag/config.py`. A candidate is kept only if it **strictly beats** the current best **on dev**.

### Judge cache

`GroundednessJudge` and `CorrectnessJudge` cache under `.cache/eval/judgements/<kind>/<fingerprint>.json`. The fingerprint includes:

* pipeline config
* normalized question
* generated answer
* retrieved-context digest
* corpus digest
* judge model id and provider
* judge prompt digest
* evaluation schema version
* (correctness) expected answer

Changing any of those is a cache miss. Each stored verdict records model, provider, prompt/corpus/config digests, raw judge output, timestamp, and whether the value was served from cache.

### Repeatability vs determinism

| What | Repeatable? |
|------|-------------|
| Local chunking, FAISS, stub embeddings, metric aggregation with fixed seeds | Yes, in tests |
| Hosted LLM answers and judges at `temperature=0` | **No.** Providers do not guarantee bit-identical outputs |
| Disk cache of a previous judge call | Repeatable **stored** scores; can hide model drift |

`set_deterministic_seeds(42)` only pins `random` and `numpy`.

### Testing the harness

```bash
make test              # OPENROUTER_API_KEY= VECTOR_BACKEND=faiss uv run pytest
make lint              # ruff format/check + mypy
```

GitHub Actions (`.github/workflows/ci.yml`) runs format, lint, typecheck, offline tests (including cache-invalidation and budget-boundary cases), and a coverage gate from a fresh clone. Live eval is a separate, manually triggered workflow.

| File | Guards |
|------|--------|
| `tests/test_config.py` | `PipelineConfig` bounds; objective weights sum to 1 |
| `tests/test_loop.py` | Hard cap at proposal and eval boundaries; held-out not stored as experiments |
| `tests/test_eval.py` | Fingerprint invalidation, correctness cases, retrieval ≠ citation, mini-harness |
| `tests/test_budget.py` | Reservation ledger and cost estimates |
| `tests/test_store.py` | SQLite experiment history |
| `tests/test_api.py` | Dashboard and query endpoints |

---

## Tuning search behavior

Edit [`program.md`](program.md) to change how the agent searches. It is loaded as the **system prompt** for every proposal.

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
| `make eval` | Held-out + adversarial eval on the current config (FAISS) |
| `make research` | Full autoresearch loop (`BUDGET=15` by default) |
| `make serve` | FastAPI dashboard + query API on `:8000` |
| `make lint` | Ruff format/check + mypy |
| `make test` | pytest — must pass with no API key |

The research loop also has a CLI: `--once`, `--budget N`, `--fresh` (clear the experiment store first), and `--store PATH`.

## Configuration

Copy `.env.example` → `.env`. `.env` is gitignored — never commit your key.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | — | Required for live runs; all LLM traffic goes through OpenRouter |
| `LLM_MODEL` | `meta-llama/llama-3.3-70b-instruct` | Model used for answers, proposals, and judges |
| `LLM_PROVIDER` | `openrouter` | Recorded on judge provenance |
| `DATABASE_URL` | `postgresql://autorag:autorag@localhost:5432/autorag` | Postgres connection for the pgvector backend |
| `VECTOR_BACKEND` | `pgvector` | `pgvector` or `faiss` (FAISS needs no database) |
| `MAX_EXPERIMENTS` | `15` | Session experiment budget |
| `MAX_USD_SPEND` | `5.0` | **Hard** per-session spend cap |

`MAX_USD_SPEND` is enforced by reserving a conservative maximum (provider `max_tokens` × published rates × query count) **before** a researcher call or a development eval. Historical rows already in the SQLite store do not count toward the session cap. A held-out eval reserve is held back so reporting can still run. If a provider ignores `max_tokens`, that in-flight call can still exceed its estimate; the loop records it and starts no further paid work.

Researcher, answering, and judging costs are accumulated separately on the session ledger.

## API

`make serve` exposes the dashboard plus a small JSON API:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard: objective chart, leaderboard, live query box |
| `GET /health` | Liveness check |
| `GET /experiments` | Full experiment history, chronological |
| `GET /leaderboard` | Experiments ranked by objective (dev) |
| `GET /best` | Winning config and its (dev) metrics |
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
| `src/autorag/loop.py` | Research loop: propose → dev eval → keep/discard |
| `src/autorag/eval.py` | Golden-eval harness, judges, metrics, fingerprints |
| `src/autorag/budget.py` | Hard-cap estimates and spend ledger |
| `src/autorag/pipeline.py` | LCEL RAG chain (chunk → embed → retrieve → rerank → answer) |
| `src/autorag/store.py` | SQLite experiment history + config-hash dedup |
| `src/autorag/llm.py` | OpenRouter client (all LLM calls) |
| `src/autorag/embeddings.py` | Sentence-transformers embeddings |
| `src/autorag/vector_store.py` | `pgvector` (default) or `faiss` backend |
| `src/autorag/api.py` | FastAPI dashboard + query API |
| `program.md` | System prompt for the researcher agent |
| `data/` | Corpus plus `dev` / `test` / `adversarial` JSONL |
| `tests/` | Test suite, stubs, and frozen eval fixtures |
| `.github/workflows/` | Offline CI and manual live-eval artifacts |
