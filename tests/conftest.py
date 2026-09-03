"""Shared pytest fixtures — no network, no API keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from autorag.config import Settings
from autorag.usage import UsageTracker
from tests.stubs import StubEmbeddings, StubLLM

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on real credentials."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")


@pytest.fixture(autouse=True)
def _faiss_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        openrouter_api_key="",
        vector_backend="faiss",
        max_experiments=15,
        max_usd_spend=5.0,
    )


@pytest.fixture
def stub_usage() -> UsageTracker:
    return UsageTracker()


@pytest.fixture
def stub_llm(stub_usage: UsageTracker) -> StubLLM:
    return StubLLM(usage=stub_usage)


@pytest.fixture
def stub_embeddings(stub_usage: UsageTracker) -> StubEmbeddings:
    return StubEmbeddings(usage=stub_usage)


@pytest.fixture
def mini_corpus() -> str:
    return (FIXTURES_DIR / "mini_corpus.txt").read_text(encoding="utf-8")


@pytest.fixture
def mini_golden_path() -> Path:
    return FIXTURES_DIR / "mini_golden.jsonl"
