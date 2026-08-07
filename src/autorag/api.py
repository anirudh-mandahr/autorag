"""FastAPI application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from autorag import __version__
from autorag.config import Settings
from autorag.pipeline import RAGPipeline, QueryResult, load_sample_corpus
from autorag.store import DEFAULT_DB_PATH, ExperimentStore

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

app = FastAPI(title="AutoRAG", version=__version__)

_pipeline_cache: tuple[str, RAGPipeline] | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


def _open_store() -> ExperimentStore:
    return ExperimentStore(DEFAULT_DB_PATH)


def _summary_with_rank(record, rank: int) -> dict[str, Any]:
    return {**record.to_summary(), "rank": rank}


def _get_query_pipeline() -> RAGPipeline:
    """Return a corpus-indexed pipeline for the current best config."""
    global _pipeline_cache

    settings = Settings()
    store = _open_store()
    try:
        best = store.best()
        if best is None:
            raise HTTPException(
                status_code=404,
                detail="No winning configuration yet. Run research first.",
            )

        config_hash = store.hash_config(best.config)
        if _pipeline_cache is not None and _pipeline_cache[0] == config_hash:
            return _pipeline_cache[1]

        pipeline = RAGPipeline(settings, best.config)
        pipeline.index_corpus(load_sample_corpus())
        _pipeline_cache = (config_hash, pipeline)
        return pipeline
    finally:
        store.close()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Single-page dashboard (leaderboard, chart, live query)."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail="Dashboard page missing")
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/experiments")
def list_experiments() -> dict[str, list[dict[str, Any]]]:
    """Experiment history in chronological order."""
    store = _open_store()
    try:
        experiments = [record.to_summary() for record in store.history()]
        return {"experiments": experiments}
    finally:
        store.close()


@app.get("/leaderboard")
def leaderboard() -> dict[str, list[dict[str, Any]]]:
    """Experiments ranked by objective (descending)."""
    store = _open_store()
    try:
        ranked = sorted(store.history(), key=lambda record: (-record.objective, record.id))
        return {
            "leaderboard": [
                _summary_with_rank(record, rank=index + 1)
                for index, record in enumerate(ranked)
            ]
        }
    finally:
        store.close()


@app.get("/best")
def best_experiment() -> dict[str, Any]:
    """Winning kept configuration and its metrics."""
    store = _open_store()
    try:
        record = store.best()
        if record is None:
            raise HTTPException(
                status_code=404,
                detail="No winning configuration yet. Run research first.",
            )
        return {"experiment": record.to_summary()}
    finally:
        store.close()


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    """Answer a question using the best pipeline configuration."""
    pipeline = _get_query_pipeline()
    result: QueryResult = pipeline.query(request.question)
    return result.model_dump()
