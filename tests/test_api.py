"""Tests for FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from autorag.config import PipelineConfig, compute_objective
from autorag.pipeline import QueryResult
from autorag.store import ExperimentStore
from tests.helpers import sample_metrics


def _seed_store(db_path: Path) -> ExperimentStore:
    store = ExperimentStore(db_path)
    metrics = sample_metrics()
    objective = compute_objective(metrics)

    store.insert(
        config=PipelineConfig(),
        metrics=metrics,
        objective=objective,
        status="kept",
        parent_id=None,
        cost_usd=0.05,
        is_baseline=True,
    )
    store.insert(
        config=PipelineConfig(chunk_size=512),
        metrics=metrics,
        objective=objective + 0.02,
        status="kept",
        parent_id=1,
        cost_usd=0.04,
    )
    store.insert(
        config=PipelineConfig(retrieval_k=8),
        metrics=metrics,
        objective=objective - 0.03,
        status="discarded",
        parent_id=1,
        cost_usd=0.03,
    )
    return store


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "experiments.db"
    _seed_store(db_path)
    monkeypatch.setattr("autorag.api.DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr("autorag.api._pipeline_cache", None)

    from autorag.api import app

    return TestClient(app)


def test_experiments_history(api_client: TestClient) -> None:
    response = api_client.get("/experiments")
    assert response.status_code == 200
    experiments = response.json()["experiments"]
    assert len(experiments) == 3
    assert [item["id"] for item in experiments] == [1, 2, 3]


def test_leaderboard_ranked_by_objective(api_client: TestClient) -> None:
    response = api_client.get("/leaderboard")
    assert response.status_code == 200
    rows = response.json()["leaderboard"]
    assert rows[0]["rank"] == 1
    assert rows[0]["id"] == 2
    assert rows[0]["objective"] >= rows[1]["objective"]


def test_best_returns_winning_config(api_client: TestClient) -> None:
    response = api_client.get("/best")
    assert response.status_code == 200
    experiment = response.json()["experiment"]
    assert experiment["id"] == 2
    assert experiment["config"]["chunk_size"] == 512
    assert "groundedness" in experiment["metrics"]


def test_best_404_when_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "empty.db"
    ExperimentStore(db_path).close()
    monkeypatch.setattr("autorag.api.DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr("autorag.api._pipeline_cache", None)

    from autorag.api import app

    client = TestClient(app)
    assert client.get("/best").status_code == 404


def test_query_uses_best_pipeline(api_client: TestClient) -> None:
    mock_result = QueryResult(
        question="What is AutoRAG?",
        answer="An autonomous RAG optimizer.",
        source_ids=["corpus-chunk-0"],
        refused=False,
        top_similarity=0.91,
    )
    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = mock_result

    with patch("autorag.api._get_query_pipeline", return_value=mock_pipeline):
        response = api_client.post("/query", json={"question": "What is AutoRAG?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "An autonomous RAG optimizer."
    mock_pipeline.query.assert_called_once_with("What is AutoRAG?")


def test_dashboard_served(api_client: TestClient) -> None:
    response = api_client.get("/")
    assert response.status_code == 200
    assert "AutoRAG Dashboard" in response.text
    assert "objective-chart" in response.text
