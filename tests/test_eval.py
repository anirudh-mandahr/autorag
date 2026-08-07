"""Unit tests for the golden eval harness."""

from __future__ import annotations

import json
import time
from itertools import count
from pathlib import Path

import pytest

from autorag.config import PipelineConfig, Settings
from autorag.usage import UsageTracker
from autorag.eval import (
    EvalHarness,
    GoldenRow,
    GroundednessJudge,
    compute_metrics,
    config_hash,
    load_golden,
    print_metrics_table,
    question_cache_key,
)
from autorag.pipeline import RAGPipeline, QueryResult
from tests.stubs import StubEmbeddings, StubLLM


def test_load_golden_has_ten_rows() -> None:
    rows = load_golden()
    assert len(rows) == 10
    assert sum(1 for row in rows if not row.answerable) == 2


def test_config_hash_is_stable() -> None:
    config = PipelineConfig()
    assert config_hash(config) == config_hash(PipelineConfig())
    assert len(config_hash(config)) == 16


def test_question_cache_key_is_stable() -> None:
    assert question_cache_key("hello") == question_cache_key("hello")
    assert question_cache_key("hello") != question_cache_key("world")


def test_compute_metrics() -> None:
    rows = [
        GoldenRow(
            question="q1",
            expected_answer="a1",
            expected_source_ids=["corpus-chunk-0"],
            answerable=True,
        ),
        GoldenRow(
            question="q2",
            expected_answer="",
            expected_source_ids=[],
            answerable=False,
        ),
    ]
    results = [
        QueryResult(
            question="q1",
            answer="answer",
            source_ids=["corpus-chunk-0"],
            refused=False,
            retrieved_ids=["corpus-chunk-0"],
            context="ctx",
        ),
        QueryResult(
            question="q2",
            answer="refused",
            source_ids=[],
            refused=True,
            retrieved_ids=[],
            context="",
        ),
    ]
    metrics = compute_metrics(
        rows,
        results,
        groundedness_scores=[1.0, None],
        query_costs=[0.01, 0.0],
        query_latencies=[100.0, 50.0],
    )
    assert metrics["answerable_recall"] == 1.0
    assert metrics["citation_rate"] == 1.0
    assert metrics["expected_source_hit_rate"] == 1.0
    assert metrics["groundedness"] == 1.0
    assert metrics["correct_refusal_rate"] == 1.0
    assert metrics["avg_cost_per_query"] == pytest.approx(0.005)
    assert metrics["avg_latency"] == pytest.approx(75.0)


def test_groundedness_judge_uses_cache(tmp_path: Path) -> None:
    config = PipelineConfig()
    calls: list[str] = []

    class FakeLLM:
        def chat(self, messages, *, temperature=0.0, response_format=None) -> str:
            calls.append(messages[-1]["content"])
            return json.dumps({"supported": True})

    judge = GroundednessJudge(FakeLLM(), config, cache_dir=tmp_path)
    score_first = judge.score("q", "a", "ctx")
    score_second = judge.score("q", "a", "ctx")
    assert score_first == 1.0
    assert score_second == 1.0
    assert len(calls) == 1
    cache_file = tmp_path / config_hash(config) / f"{question_cache_key('q')}.json"
    assert cache_file.exists()


def test_eval_harness_stable_metrics_on_mini_corpus(
    test_settings: Settings,
    mini_corpus: str,
    mini_golden_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frozen mini-corpus + stubbed clients => deterministic metrics, no network."""
    tick = count(1_000_000, 1)
    monkeypatch.setattr(time, "perf_counter", lambda: next(tick) / 1000.0)

    usage = UsageTracker()
    llm = StubLLM(usage=usage)
    embeddings = StubEmbeddings(usage=usage)
    config = PipelineConfig(refusal_threshold=0.25)

    pipeline = RAGPipeline(
        test_settings,
        config,
        llm=llm,
        embeddings=embeddings,
        usage=usage,
    )
    pipeline.index_corpus(mini_corpus)

    class StubJudge:
        def score(self, question: str, answer: str, context: str) -> float:
            return 1.0

    harness = EvalHarness(
        pipeline,
        config,
        golden_path=mini_golden_path,
        judge=StubJudge(),
        seed=42,
    )

    first = harness.run()
    second = harness.run()

    expected = {
        "answerable_recall": 1.0,
        "citation_rate": 1.0,
        "expected_source_hit_rate": 1.0,
        "groundedness": 1.0,
        "correct_refusal_rate": 1.0,
        "avg_cost_per_query": 0.001,
        "avg_latency": 1.0,
    }
    assert first == expected
    assert second == expected


def test_print_metrics_table(capsys: pytest.CaptureFixture[str]) -> None:
    metrics = {
        "answerable_recall": 0.875,
        "citation_rate": 1.0,
        "expected_source_hit_rate": 0.9,
        "groundedness": 0.8,
        "correct_refusal_rate": 1.0,
        "avg_cost_per_query": 0.001234,
        "avg_latency": 123.45,
    }
    print_metrics_table(metrics)
    output = capsys.readouterr().out
    assert "answerable_recall" in output
    assert "0.8750" in output
    assert "$0.001234" in output
    assert "123.45 ms" in output
