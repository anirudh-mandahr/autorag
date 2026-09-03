"""Shared test metrics that include every objective key."""

from __future__ import annotations

from autorag.pipeline import QueryResult

SAMPLE_METRICS: dict[str, float] = {
    "answerable_recall": 0.9,
    "retrieval_hit_at_k": 0.8,
    "retrieval_mrr": 0.7,
    "retrieval_ndcg": 0.75,
    "citation_precision": 0.8,
    "citation_recall": 0.7,
    "claim_citation_correctness": 0.7,
    "groundedness": 0.85,
    "answer_correctness": 0.8,
    "correct_refusal_rate": 1.0,
    "avg_cost_per_query": 0.002,
    "avg_latency": 100.0,
    "answering_cost_usd": 0.01,
    "judging_cost_usd": 0.005,
}


def sample_metrics(**overrides: float) -> dict[str, float]:
    metrics = dict(SAMPLE_METRICS)
    metrics.update(overrides)
    return metrics


def query_result(**overrides: object) -> QueryResult:
    payload: dict[str, object] = {
        "question": "q",
        "answer": "a",
        "source_ids": [],
        "refused": False,
        "retrieved_ids": [],
        "context": "",
        "top_similarity": 0.0,
    }
    payload.update(overrides)
    return QueryResult.model_validate(payload)
