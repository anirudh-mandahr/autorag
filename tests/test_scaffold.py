"""Smoke tests for project scaffold."""

import importlib

import pytest
from fastapi.testclient import TestClient

MODULES = [
    "autorag",
    "autorag.config",
    "autorag.llm",
    "autorag.embeddings",
    "autorag.vector_store",
    "autorag.store",
    "autorag.pipeline",
    "autorag.eval",
    "autorag.loop",
    "autorag.api",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import_module(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_health_endpoint() -> None:
    from autorag.api import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_pipeline_config_is_pydantic_model() -> None:
    from autorag.config import PipelineConfig

    cfg = PipelineConfig()
    dumped = cfg.model_dump()
    assert dumped["chunk_size"] == 400
    assert dumped["rerank_strategy"] == "cosine"
    assert dumped["prompt_template_id"] == "grounded_v1"
