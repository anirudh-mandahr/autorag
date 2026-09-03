"""Edge-case coverage for metrics, parsers, pipeline, and CLI helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autorag.budget import CostBreakdown
from autorag.config import PipelineConfig, Settings
from autorag.eval import (
    CorrectnessJudge,
    EvalHarness,
    EvalResult,
    GoldenRow,
    GroundednessJudge,
    bootstrap_mean_ci,
    citation_precision,
    compute_metrics,
    deterministic_correctness,
    load_golden,
    print_intervals_table,
    repeat_run_variance,
    retrieval_mrr,
    retrieval_ndcg,
    split_path,
)
from autorag.eval import main as eval_main
from autorag.llm import LLMClient
from autorag.loop import (
    format_config_delta,
    held_out_reserve_usd,
    parse_pipeline_config,
    run_experiment,
)
from autorag.pipeline import RAGPipeline, load_sample_questions
from autorag.prompts import get_prompt_template
from autorag.usage import CallMetrics, UsageTracker
from tests.helpers import query_result, sample_metrics
from tests.stubs import StubEmbeddings, StubLLM


def test_split_path_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        split_path("train")


def test_load_golden_named_split() -> None:
    rows = load_golden(split="adversarial")
    assert rows
    assert all(not row.answerable for row in rows)


def test_load_golden_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '\n{"question": "q", "expected_answer": "a", "answerable": true}\n\n',
        encoding="utf-8",
    )
    assert len(load_golden(path)) == 1


def test_retrieval_rank_metrics() -> None:
    assert retrieval_mrr(["gold"], ["other", "gold"]) == pytest.approx(0.5)
    assert retrieval_mrr(["gold"], []) == 0.0
    assert retrieval_ndcg(["gold"], []) == 0.0
    assert retrieval_ndcg(["gold"], ["gold"]) == 1.0
    assert citation_precision(["gold"], []) == 0.0


def test_bootstrap_and_repeat_variance_empty() -> None:
    empty = bootstrap_mean_ci([])
    assert empty["n"] == 0.0
    assert repeat_run_variance([]) == {}
    variance = repeat_run_variance([{"a": 1.0}, {"a": 1.0}])
    assert variance["a"]["std"] == 0.0


def test_compute_metrics_rejects_length_mismatch() -> None:
    row = GoldenRow(question="q", expected_answer="a")
    result = query_result()
    with pytest.raises(ValueError, match="length mismatch"):
        compute_metrics(
            [row],
            [],
            groundedness_scores=[],
            correctness_scores=[],
            query_costs=[],
            query_latencies=[],
        )
    with pytest.raises(ValueError, match="per-row"):
        compute_metrics(
            [row],
            [result],
            groundedness_scores=[1.0],
            correctness_scores=[],
            query_costs=[0.0],
            query_latencies=[1.0],
        )


def test_deterministic_correctness_and_intervals(capsys: pytest.CaptureFixture[str]) -> None:
    assert deterministic_correctness("The period is 18 Earth days.", "18 Earth days.") == 1.0
    assert deterministic_correctness("nope", "18 Earth days.") is None
    assert deterministic_correctness("x", "  ") is None
    print_intervals_table({"groundedness": bootstrap_mean_ci([1.0, 0.0, 1.0])})
    assert "groundedness" in capsys.readouterr().out


def test_judge_parsers(tmp_path: Path) -> None:
    class FakeLLM:
        def __init__(self, raw: str) -> None:
            self.raw = raw

        def chat(self, messages, **kwargs) -> str:
            return self.raw

    grounded = GroundednessJudge(
        FakeLLM("not json true"),
        PipelineConfig(),
        cache_dir=tmp_path / "g1",
        corpus_digest="c",
        model_id="m",
        provider="p",
    )
    assert grounded.score("q", "a", "ctx").score == 1.0

    grounded_false = GroundednessJudge(
        FakeLLM(json.dumps({"supported": False})),
        PipelineConfig(),
        cache_dir=tmp_path / "g2",
        corpus_digest="c",
        model_id="m",
        provider="p",
    )
    assert grounded_false.score("q", "b", "ctx").score == 0.0

    correct = CorrectnessJudge(
        FakeLLM(json.dumps({"label": "partial"})),
        PipelineConfig(),
        cache_dir=tmp_path / "c1",
        corpus_digest="c",
        model_id="m",
        provider="p",
    )
    assert correct.score("q", "maybe", "ctx", expected_answer="full fact").score == 0.5

    malformed = CorrectnessJudge(
        FakeLLM("this is incorrect"),
        PipelineConfig(),
        cache_dir=tmp_path / "c2",
        corpus_digest="c",
        model_id="m",
        provider="p",
    )
    assert malformed.score("q", "x", "ctx", expected_answer="y").score == 0.0

    unknown_label = CorrectnessJudge(
        FakeLLM(json.dumps({"label": "whatever"})),
        PipelineConfig(),
        cache_dir=tmp_path / "c3",
        corpus_digest="c",
        model_id="m",
        provider="p",
    )
    assert unknown_label.score("q", "x", "ctx", expected_answer="z").score == 0.0


def test_eval_harness_empty_and_repeated(
    test_settings: Settings,
    mini_corpus: str,
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    usage = UsageTracker()
    pipeline = RAGPipeline(
        test_settings,
        PipelineConfig(),
        llm=StubLLM(usage=usage),
        embeddings=StubEmbeddings(usage=usage),
        usage=usage,
    )
    pipeline.index_corpus(mini_corpus)

    class StubJudge:
        def score(self, *args, **kwargs) -> float:
            return 1.0

    harness = EvalHarness(
        pipeline,
        PipelineConfig(),
        golden_path=empty,
        groundedness_judge=StubJudge(),
        correctness_judge=StubJudge(),
    )
    result = harness.run()
    assert result.n == 0
    assert result.metrics["answerable_recall"] == 0.0

    mini = tmp_path / "mini.jsonl"
    mini.write_text(
        json.dumps(
            {
                "question": "How long is Alpha station's orbital period?",
                "expected_answer": "18 Earth days",
                "expected_source_ids": ["corpus-chunk-0"],
                "answerable": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    harness2 = EvalHarness(
        pipeline,
        PipelineConfig(refusal_threshold=0.0),
        golden_path=mini,
        groundedness_judge=StubJudge(),
        correctness_judge=StubJudge(),
    )
    repeated = harness2.run_repeated(2)
    assert len(repeated.repeats) == 2
    with pytest.raises(ValueError):
        harness2.run_repeated(0)


def test_eval_cli_writes_json(tmp_path: Path, test_settings: Settings, mini_corpus: str) -> None:

    fake = EvalResult(
        metrics=sample_metrics(),
        split="test",
        n=2,
        intervals={"groundedness": bootstrap_mean_ci([1.0, 1.0])},
        cost=CostBreakdown(),
        repeats=[sample_metrics(), sample_metrics()],
    )
    out = tmp_path / "eval.json"
    with (
        patch("autorag.eval.Settings", return_value=test_settings),
        patch("autorag.eval.run_eval", return_value=fake),
        patch("autorag.eval.load_sample_corpus", return_value=mini_corpus),
    ):
        assert eval_main(["--split", "test", "--repeats", "2", "--json", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "test" in payload["splits"]
    assert payload["splits"]["test"]["n"] == 2


def test_parse_pipeline_config_and_delta() -> None:
    wrapped = 'Sure.\n{"chunk_size": 512, "chunk_overlap": 40, "embedding_model": '
    wrapped += '"sentence-transformers/all-MiniLM-L6-v2", "retrieval_k": 4, '
    wrapped += '"rerank_strategy": "cosine", "mmr_lambda": 0.5, '
    wrapped += '"prompt_template_id": "grounded_v1", "refusal_threshold": 0.25}\n'
    parsed = parse_pipeline_config(wrapped)
    assert parsed.chunk_size == 512
    with pytest.raises(ValueError, match="did not return JSON"):
        parse_pipeline_config("nope")
    with pytest.raises(ValueError, match="object"):
        parse_pipeline_config("[1]")
    assert format_config_delta(None, PipelineConfig()) == "baseline"
    assert "chunk_size" in format_config_delta(PipelineConfig(), PipelineConfig(chunk_size=512))
    assert format_config_delta(PipelineConfig(), PipelineConfig()) == "(unchanged)"


def test_held_out_reserve_and_run_experiment_wrapper() -> None:
    settings = Settings(_env_file=None, llm_model="meta-llama/llama-3.3-70b-instruct")
    assert held_out_reserve_usd(settings) > 0
    with patch(
        "autorag.loop.evaluate_config",
        return_value=type(
            "R",
            (),
            {
                "metrics": sample_metrics(),
                "cost": type("C", (), {"total_usd": 0.02})(),
            },
        )(),
    ):
        metrics, cost = run_experiment(settings, PipelineConfig(), split="dev")
    assert cost == 0.02
    assert metrics["groundedness"] == 0.85


def test_llm_parse_grounded_response() -> None:
    answer, ids = LLMClient._parse_grounded_response(
        'prefix {"answer": "hi", "source_ids": ["corpus-chunk-0"]} suffix'
    )
    assert answer == "hi"
    assert ids == ["corpus-chunk-0"]
    answer, ids = LLMClient._parse_grounded_response('{"answer": "x", "source_ids": "nope"}')
    assert ids == []
    with pytest.raises(ValueError):
        LLMClient._parse_grounded_response("no json here")


def test_pipeline_mmr_empty_corpus_and_guard(
    test_settings: Settings,
    mini_corpus: str,
) -> None:
    usage = UsageTracker()
    config = PipelineConfig(rerank_strategy="mmr", retrieval_k=2, refusal_threshold=0.0)
    pipeline = RAGPipeline(
        test_settings,
        config,
        llm=StubLLM(usage=usage),
        embeddings=StubEmbeddings(usage=usage),
        usage=usage,
    )
    with pytest.raises(RuntimeError, match="index_corpus"):
        pipeline.query("q")
    assert pipeline.index_corpus("") == 0
    pipeline.index_corpus(mini_corpus)
    result = pipeline.query("How long is Alpha station's orbital period?")
    assert result.answer
    assert pipeline.metrics["calls"] >= 1
    assert load_sample_questions()
    with pytest.raises(ValueError):
        get_prompt_template("missing")
    usage.record(CallMetrics(operation="other", cost_usd=0.1))
    assert usage.cost_for("other") == pytest.approx(0.1)
    assert "other" in usage.cost_by_operation()
    assert usage.summary()["calls"] >= 1
    assert usage.total_tokens >= 0
    assert CallMetrics(input_tokens=1, output_tokens=2).total_tokens == 3
