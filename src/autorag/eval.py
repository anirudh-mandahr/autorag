"""Golden-eval harness for scoring pipeline outputs."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from autorag.config import PipelineConfig, Settings
from autorag.pipeline import DATA_DIR, RAGPipeline, QueryResult, load_sample_corpus

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "eval" / "groundedness"

METRIC_NAMES = (
    "answerable_recall",
    "citation_rate",
    "expected_source_hit_rate",
    "groundedness",
    "correct_refusal_rate",
    "avg_cost_per_query",
    "avg_latency",
)


class GoldenRow(BaseModel):
    question: str
    expected_answer: str
    expected_source_ids: list[str] = Field(default_factory=list)
    answerable: bool = True


def load_golden(path: Path | None = None) -> list[GoldenRow]:
    """Load golden eval rows from JSONL."""
    golden_path = path or (DATA_DIR / "golden.jsonl")
    rows: list[GoldenRow] = []
    for line in golden_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(GoldenRow.model_validate(json.loads(stripped)))
    return rows


def config_hash(config: PipelineConfig) -> str:
    """Stable short hash of pipeline knobs for groundedness caching."""
    payload = json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def question_cache_key(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def set_deterministic_seeds(seed: int = 42) -> None:
    """Fix RNG seeds used by eval and downstream numeric code."""
    random.seed(seed)
    np.random.seed(seed)


class GroundednessJudge:
    """Cheap LLM-as-judge for whether an answer is supported by context."""

    _PROMPT = (
        "You are a strict evaluator. Given the retrieved context and a candidate answer, "
        "decide whether every factual claim in the answer is directly supported by the "
        "context. Reply with JSON only: {\"supported\": true} or {\"supported\": false}."
    )

    def __init__(
        self,
        llm,
        config: PipelineConfig,
        *,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self._llm = llm
        self._config_hash = config_hash(config)
        self._cache_dir = cache_dir / self._config_hash
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def score(self, question: str, answer: str, context: str) -> float:
        """Return 1.0 if supported, else 0.0. Cached by (config_hash, question)."""
        cache_path = self._cache_dir / f"{question_cache_key(question)}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return float(cached["score"])

        user_content = (
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            f"Answer: {answer}"
        )
        raw = self._llm.chat(
            messages=[
                {"role": "system", "content": self._PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        supported = self._parse_supported(raw)
        score = 1.0 if supported else 0.0
        cache_path.write_text(
            json.dumps({"score": score, "question": question}, indent=2),
            encoding="utf-8",
        )
        return score

    @staticmethod
    def _parse_supported(raw: str) -> bool:
        text = raw.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            lowered = text.lower()
            if "true" in lowered and "false" not in lowered:
                return True
            return False
        return bool(data.get("supported", False))


def _has_expected_source_hit(row: GoldenRow, result: QueryResult) -> bool:
    if not row.expected_source_ids:
        return True
    cited_or_retrieved = set(result.source_ids) | set(result.retrieved_ids)
    return bool(cited_or_retrieved.intersection(row.expected_source_ids))


def compute_metrics(
    rows: list[GoldenRow],
    results: list[QueryResult],
    *,
    groundedness_scores: list[float | None],
    query_costs: list[float],
    query_latencies: list[float],
) -> dict[str, float]:
    """Aggregate per-row outcomes into the eval metric dict."""
    if len(rows) != len(results):
        raise ValueError("rows and results length mismatch")

    answerable = [row for row in rows if row.answerable]
    answerable_results = [
        result for row, result in zip(rows, results, strict=True) if row.answerable
    ]
    unanswerable = [row for row in rows if not row.answerable]
    unanswerable_results = [
        result for row, result in zip(rows, results, strict=True) if not row.answerable
    ]

    answered_answerable = sum(1 for result in answerable_results if not result.refused)
    answerable_recall = (
        answered_answerable / len(answerable_results) if answerable_results else 1.0
    )

    non_refused = [result for result in results if not result.refused]
    citation_rate = (
        sum(1 for result in non_refused if result.source_ids) / len(non_refused)
        if non_refused
        else 0.0
    )

    source_rows = [
        (row, result)
        for row, result in zip(rows, results, strict=True)
        if row.expected_source_ids
    ]
    expected_source_hit_rate = (
        sum(1 for row, result in source_rows if _has_expected_source_hit(row, result))
        / len(source_rows)
        if source_rows
        else 1.0
    )

    judged = [score for score in groundedness_scores if score is not None]
    groundedness = sum(judged) / len(judged) if judged else 0.0

    correct_refusals = sum(1 for result in unanswerable_results if result.refused)
    correct_refusal_rate = (
        correct_refusals / len(unanswerable_results) if unanswerable_results else 1.0
    )

    query_count = len(results)
    avg_cost_per_query = sum(query_costs) / query_count if query_count else 0.0
    avg_latency = sum(query_latencies) / query_count if query_count else 0.0

    return {
        "answerable_recall": round(answerable_recall, 4),
        "citation_rate": round(citation_rate, 4),
        "expected_source_hit_rate": round(expected_source_hit_rate, 4),
        "groundedness": round(groundedness, 4),
        "correct_refusal_rate": round(correct_refusal_rate, 4),
        "avg_cost_per_query": round(avg_cost_per_query, 6),
        "avg_latency": round(avg_latency, 2),
    }


def print_metrics_table(metrics: dict[str, float]) -> None:
    """Print a fixed-width metrics table to stdout."""
    name_width = max(len(name) for name in METRIC_NAMES)
    print(f"{'Metric':<{name_width}}  Value")
    print(f"{'-' * name_width}  -----")
    for name in METRIC_NAMES:
        value = metrics.get(name, 0.0)
        if name.startswith("avg_cost"):
            print(f"{name:<{name_width}}  ${value:.6f}")
        elif name == "avg_latency":
            print(f"{name:<{name_width}}  {value:.2f} ms")
        else:
            print(f"{name:<{name_width}}  {value:.4f}")


class EvalHarness:
    """Runs golden examples and returns aggregate scores."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        config: PipelineConfig,
        *,
        golden_path: Path | None = None,
        judge: GroundednessJudge | None = None,
        seed: int = 42,
    ) -> None:
        self._pipeline = pipeline
        self._config = config
        self._golden_path = golden_path
        self._judge = judge or GroundednessJudge(pipeline._llm, config)
        self._seed = seed

    def run(self) -> dict[str, float]:
        """Execute eval suite and return metric dict."""
        set_deterministic_seeds(self._seed)
        rows = load_golden(self._golden_path)
        if not rows:
            return {name: 0.0 for name in METRIC_NAMES}

        results: list[QueryResult] = []
        groundedness_scores: list[float | None] = []
        query_costs: list[float] = []
        query_latencies: list[float] = []

        for row in rows:
            cost_before = self._pipeline.usage.total_cost_usd
            started = time.perf_counter()
            result = self._pipeline.query(row.question)

            if result.refused:
                groundedness_scores.append(None)
            else:
                groundedness_scores.append(
                    self._judge.score(result.question, result.answer, result.context)
                )

            elapsed_ms = (time.perf_counter() - started) * 1000
            cost_after = self._pipeline.usage.total_cost_usd

            results.append(result)
            query_costs.append(cost_after - cost_before)
            query_latencies.append(elapsed_ms)

        return compute_metrics(
            rows,
            results,
            groundedness_scores=groundedness_scores,
            query_costs=query_costs,
            query_latencies=query_latencies,
        )


def run_eval(
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
    *,
    golden_path: Path | None = None,
) -> dict[str, float]:
    """Index the sample corpus and run the golden eval suite."""
    set_deterministic_seeds()
    settings = settings or Settings()
    config = config or PipelineConfig()
    pipeline = RAGPipeline(settings, config)
    pipeline.index_corpus(load_sample_corpus())
    harness = EvalHarness(pipeline, config, golden_path=golden_path)
    return harness.run()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m autorag.eval`."""
    _ = argv
    metrics = run_eval()
    print_metrics_table(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
