"""Unit tests for the golden eval harness."""

from __future__ import annotations

import json
import time
from itertools import count
from pathlib import Path

import pytest

from autorag.config import PipelineConfig, Settings
from autorag.eval import (
    CORRECTNESS_PROMPT,
    GROUNDEDNESS_PROMPT,
    CorrectnessJudge,
    EvalHarness,
    GoldenRow,
    GroundednessJudge,
    compute_metrics,
    config_hash,
    digest_text,
    evaluation_fingerprint,
    load_golden,
    load_split,
    normalize_question,
    print_metrics_table,
    question_cache_key,
)
from autorag.pipeline import RAGPipeline
from autorag.usage import UsageTracker
from tests.helpers import query_result
from tests.stubs import StubEmbeddings, StubLLM


def test_splits_are_partitioned() -> None:
    dev = load_split("dev")
    test = load_split("test")
    adversarial = load_split("adversarial")
    assert len(dev) == 7
    assert len(test) == 5
    assert len(adversarial) == 5
    assert sum(1 for row in dev if not row.answerable) == 1
    assert sum(1 for row in test if not row.answerable) == 1
    assert all(not row.answerable for row in adversarial)
    questions = [row.question for row in (*dev, *test, *adversarial)]
    assert len(questions) == len(set(questions))
    assert len(load_golden()) == 17


def test_config_hash_is_stable() -> None:
    config = PipelineConfig()
    assert config_hash(config) == config_hash(PipelineConfig())
    assert len(config_hash(config)) == 16


def test_question_cache_key_normalizes_whitespace() -> None:
    assert question_cache_key("Hello  World") == question_cache_key("hello world")
    assert question_cache_key("hello") != question_cache_key("world")


def _fingerprint(**overrides: object) -> str:
    payload = {
        "config": PipelineConfig(),
        "question": "How long is the orbit?",
        "answer": "18 Earth days.",
        "context": "[corpus-chunk-0] Lyra-7 orbits Keth.",
        "corpus_digest": "corpus-v1",
        "judge_model": "meta-llama/llama-3.3-70b-instruct",
        "judge_provider": "openrouter",
        "judge_prompt": GROUNDEDNESS_PROMPT,
        "schema_version": "groundedness_v2",
    }
    payload.update(overrides)
    return evaluation_fingerprint(**payload)  # type: ignore[arg-type]


def test_evaluation_fingerprint_changes_with_each_relevant_input() -> None:
    base = _fingerprint()
    assert _fingerprint(answer="19 Earth days.") != base
    assert _fingerprint(context="different evidence") != base
    assert _fingerprint(corpus_digest="corpus-v2") != base
    assert _fingerprint(judge_model="other-model") != base
    assert _fingerprint(judge_provider="other-provider") != base
    assert _fingerprint(judge_prompt=CORRECTNESS_PROMPT) != base
    assert _fingerprint(schema_version="groundedness_v3") != base
    assert _fingerprint(config=PipelineConfig(retrieval_k=8)) != base
    assert _fingerprint(question="A different question?") != base
    assert _fingerprint(question="HOW LONG IS THE ORBIT?") == base


def test_compute_metrics_separates_retrieval_from_citation() -> None:
    rows = [
        GoldenRow(
            question="q1",
            expected_answer="18 Earth days.",
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
        query_result(
            question="q1",
            answer="18 Earth days.",
            source_ids=["corpus-chunk-0"],
            retrieved_ids=["corpus-chunk-0"],
            context="ctx",
        ),
        query_result(question="q2", answer="refused", refused=True),
    ]
    metrics, _ = compute_metrics(
        rows,
        results,
        groundedness_scores=[1.0, None],
        correctness_scores=[1.0, None],
        query_costs=[0.01, 0.0],
        query_latencies=[100.0, 50.0],
        answering_cost_usd=0.01,
        judging_cost_usd=0.002,
    )
    assert metrics["answerable_recall"] == 1.0
    assert metrics["retrieval_hit_at_k"] == 1.0
    assert metrics["citation_recall"] == 1.0
    assert metrics["citation_precision"] == 1.0
    assert metrics["claim_citation_correctness"] == 1.0
    assert metrics["groundedness"] == 1.0
    assert metrics["answer_correctness"] == 1.0
    assert metrics["correct_refusal_rate"] == 1.0
    assert metrics["avg_cost_per_query"] == pytest.approx(0.005)
    assert metrics["answering_cost_usd"] == pytest.approx(0.01)


def test_retrieved_but_uncited_is_not_citation_success() -> None:
    rows = [
        GoldenRow(
            question="q",
            expected_answer="18 Earth days.",
            expected_source_ids=["corpus-chunk-0"],
            answerable=True,
        )
    ]
    results = [
        query_result(
            answer="18 Earth days.",
            source_ids=[],
            retrieved_ids=["corpus-chunk-0", "corpus-chunk-1"],
        )
    ]
    metrics, _ = compute_metrics(
        rows,
        results,
        groundedness_scores=[1.0],
        correctness_scores=[1.0],
        query_costs=[0.0],
        query_latencies=[1.0],
    )
    assert metrics["retrieval_hit_at_k"] == 1.0
    assert metrics["retrieval_mrr"] == 1.0
    assert metrics["citation_recall"] == 0.0
    assert metrics["citation_precision"] == 0.0
    assert metrics["claim_citation_correctness"] == 0.0


def test_correctness_cases_are_independent_of_groundedness() -> None:
    def row(question: str, expected: str) -> GoldenRow:
        return GoldenRow(
            question=question,
            expected_answer=expected,
            expected_source_ids=["corpus-chunk-0"],
            answerable=True,
        )

    rows = [
        row("correct-grounded", "18 Earth days."),
        row("incorrect-grounded", "18 Earth days."),
        row("correct-unsupported", "18 Earth days."),
        row("partial", "18 Earth days and 0.92 g."),
        row("refused-answerable", "18 Earth days."),
    ]
    results = [
        query_result(
            question="correct-grounded",
            answer="18 Earth days.",
            source_ids=["corpus-chunk-0"],
            retrieved_ids=["corpus-chunk-0"],
        ),
        query_result(
            question="incorrect-grounded",
            answer="The orbit lasts 40 years.",
            source_ids=["corpus-chunk-0"],
            retrieved_ids=["corpus-chunk-0"],
        ),
        query_result(
            question="correct-unsupported",
            answer="18 Earth days.",
            source_ids=["corpus-chunk-9"],
            retrieved_ids=["corpus-chunk-9"],
        ),
        query_result(
            question="partial",
            answer="18 Earth days.",
            source_ids=["corpus-chunk-0"],
            retrieved_ids=["corpus-chunk-0"],
        ),
        query_result(question="refused-answerable", answer="refused", refused=True),
    ]
    metrics, per_row = compute_metrics(
        rows,
        results,
        groundedness_scores=[1.0, 1.0, 0.0, 1.0, None],
        correctness_scores=[1.0, 0.0, 1.0, 0.5, 0.0],
        query_costs=[0.0] * 5,
        query_latencies=[1.0] * 5,
    )
    assert per_row["answer_correctness"] == [1.0, 0.0, 1.0, 0.5, 0.0]
    assert per_row["groundedness"] == [1.0, 1.0, 0.0, 1.0]
    assert metrics["answerable_recall"] == 0.8
    assert metrics["answer_correctness"] == pytest.approx(0.5)
    assert metrics["groundedness"] == pytest.approx(0.75)
    assert metrics["retrieval_hit_at_k"] == pytest.approx(0.6)
    assert metrics["citation_recall"] == pytest.approx(0.75)


def test_groundedness_judge_uses_cache(tmp_path: Path) -> None:
    config = PipelineConfig()
    calls: list[str] = []

    class FakeLLM:
        model_id = "fake-model"
        provider = "fake"

        def chat(self, messages, **kwargs) -> str:
            calls.append(messages[-1]["content"])
            return json.dumps({"supported": True})

    judge = GroundednessJudge(
        FakeLLM(),
        config,
        cache_dir=tmp_path,
        corpus_digest="corpus-v1",
        model_id="fake-model",
        provider="fake",
    )
    first = judge.score("q", "a", "ctx")
    second = judge.score("q", "a", "ctx")
    assert first.score == 1.0
    assert second.score == 1.0
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.raw_output
    assert second.timestamp
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("answer", {"answer": "changed-answer"}),
        ("context", {"context": "changed-context"}),
        ("corpus", {"corpus_digest": "corpus-v2"}),
        ("model", {"model_id": "other-model"}),
        ("prompt", {"prompt": "a different judge prompt"}),
        ("schema", {"schema_version": "groundedness_v9"}),
        ("config", {"config": PipelineConfig(chunk_size=512)}),
    ],
)
def test_groundedness_cache_invalidates_on_relevant_change(
    tmp_path: Path, field: str, kwargs: dict
) -> None:
    calls: list[int] = []

    class FakeLLM:
        def chat(self, messages, **kw) -> str:
            calls.append(1)
            return json.dumps({"supported": True})

    def make_judge(**overrides: object) -> GroundednessJudge:
        params = {
            "corpus_digest": "corpus-v1",
            "model_id": "model-a",
            "provider": "openrouter",
            "cache_dir": tmp_path,
        }
        skip = {"config", "prompt", "schema_version", "answer", "context"}
        params.update({k: v for k, v in overrides.items() if k not in skip})
        judge = GroundednessJudge(
            FakeLLM(),
            overrides.get("config", PipelineConfig()),  # type: ignore[arg-type]
            **params,  # type: ignore[arg-type]
        )
        if "prompt" in overrides:
            judge.prompt = str(overrides["prompt"])
        if "schema_version" in overrides:
            judge.schema_version = str(overrides["schema_version"])
        return judge

    base = make_judge()
    base.score("q", "a", "ctx")
    changed = make_judge(**kwargs)
    changed.score(
        "q",
        str(kwargs.get("answer", "a")),
        str(kwargs.get("context", "ctx")),
    )
    assert len(calls) == 2, f"{field} should miss the cache"


def test_correctness_cache_invalidates_when_expected_answer_changes(tmp_path: Path) -> None:
    calls: list[str] = []

    class FakeLLM:
        def chat(self, messages, **kwargs) -> str:
            calls.append(messages[-1]["content"])
            return json.dumps({"label": "incorrect"})

    judge = CorrectnessJudge(
        FakeLLM(),
        PipelineConfig(),
        cache_dir=tmp_path,
        corpus_digest="c",
        model_id="m",
        provider="p",
    )
    # Non-containment so the LLM is consulted.
    judge.score("q", "blue", "ctx", expected_answer="red")
    judge.score("q", "blue", "ctx", expected_answer="green")
    assert len(calls) == 2


def test_eval_harness_stable_metrics_on_mini_corpus(
    test_settings: Settings,
    mini_corpus: str,
    mini_golden_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frozen mini-corpus + stubbed clients => deterministic local metrics, no network."""
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
        def score(self, question: str, answer: str, context: str, **kwargs) -> float:
            return 1.0

    harness = EvalHarness(
        pipeline,
        config,
        golden_path=mini_golden_path,
        groundedness_judge=StubJudge(),
        correctness_judge=StubJudge(),
        corpus_digest=digest_text(mini_corpus),
        seed=42,
    )

    first = harness.run()
    second = harness.run()

    expected = {
        "answerable_recall": 1.0,
        "retrieval_hit_at_k": 1.0,
        "retrieval_mrr": 1.0,
        "retrieval_ndcg": 1.0,
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "claim_citation_correctness": 1.0,
        "groundedness": 1.0,
        "answer_correctness": 1.0,
        "correct_refusal_rate": 1.0,
        "avg_cost_per_query": 0.001,
        "avg_latency": 1.0,
        "answering_cost_usd": 0.002,
        "judging_cost_usd": 0.0,
    }
    assert first.metrics == expected
    assert second.metrics == expected
    assert first.n == 2
    assert "answer_correctness" in first.intervals


def test_print_metrics_table(capsys: pytest.CaptureFixture[str]) -> None:
    metrics = {
        "answerable_recall": 0.875,
        "retrieval_hit_at_k": 0.8,
        "citation_recall": 0.7,
        "groundedness": 0.8,
        "answer_correctness": 0.6,
        "correct_refusal_rate": 1.0,
        "avg_cost_per_query": 0.001234,
        "avg_latency": 123.45,
    }
    print_metrics_table(metrics)
    output = capsys.readouterr().out
    assert "answerable_recall" in output
    assert "answer_correctness" in output
    assert "0.8750" in output
    assert "$0.001234" in output
    assert "123.45 ms" in output


def test_normalize_question() -> None:
    assert normalize_question("  Hello   WORLD ") == "hello world"
