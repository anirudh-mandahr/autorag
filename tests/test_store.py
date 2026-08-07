"""Tests for experiment store and objective."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autorag.config import PipelineConfig, compute_objective
from autorag.store import ExperimentStore


def test_compute_objective() -> None:
    metrics = {
        "answerable_recall": 1.0,
        "citation_rate": 1.0,
        "expected_source_hit_rate": 1.0,
        "groundedness": 1.0,
        "correct_refusal_rate": 1.0,
        "avg_cost_per_query": 0.01,
        "avg_latency": 100.0,
    }
    assert compute_objective(metrics) == pytest.approx(1.0 - 0.5)


def test_experiment_store_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "experiments.db"
    store = ExperimentStore(db_path)
    config = PipelineConfig(chunk_size=512)
    metrics = {
        "answerable_recall": 0.9,
        "citation_rate": 0.8,
        "expected_source_hit_rate": 0.7,
        "groundedness": 0.85,
        "correct_refusal_rate": 1.0,
        "avg_cost_per_query": 0.002,
        "avg_latency": 120.0,
    }
    objective = compute_objective(metrics)

    baseline = store.insert(
        config=PipelineConfig(),
        metrics=metrics,
        objective=objective,
        status="kept",
        parent_id=None,
        cost_usd=0.05,
        is_baseline=True,
    )
    candidate = store.insert(
        config=config,
        metrics=metrics,
        objective=objective + 0.01,
        status="kept",
        parent_id=baseline.id,
        cost_usd=0.04,
    )
    store.insert(
        config=PipelineConfig(retrieval_k=8),
        metrics=metrics,
        objective=objective - 0.02,
        status="discarded",
        parent_id=baseline.id,
        cost_usd=0.03,
    )

    assert store.count() == 3
    assert store.has_config(config)
    assert not store.has_config(PipelineConfig(retrieval_k=99))
    assert store.baseline() is not None
    assert store.best() is not None
    assert store.best().id == candidate.id
    assert len(store.history()) == 3
    assert store.total_cost_usd() == pytest.approx(0.12)
    assert store.hash_config(config) in store.seen_config_hashes()

    summary = candidate.to_summary()
    assert summary["config"]["chunk_size"] == 512
    assert json.loads(json.dumps(summary))["status"] == "kept"
    store.close()
