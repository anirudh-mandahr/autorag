"""Golden-eval harness: distinct metrics, fingerprinted judges, split datasets."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autorag.budget import (
    JUDGE_MAX_OUTPUT_TOKENS,
    CostBreakdown,
)
from autorag.config import PipelineConfig, Settings
from autorag.pipeline import DATA_DIR, QueryResult, RAGPipeline, load_sample_corpus

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "eval" / "judgements"

SPLITS = ("dev", "test", "adversarial")

GROUNDEDNESS_SCHEMA_VERSION = "groundedness_v2"
CORRECTNESS_SCHEMA_VERSION = "correctness_v1"

GROUNDEDNESS_PROMPT = (
    "You are a strict groundedness evaluator. Given the retrieved context and a "
    "candidate answer, decide whether every factual claim in the answer is directly "
    "supported by the context. Do not score correctness against an expected answer. "
    'Reply with JSON only: {"supported": true} or {"supported": false}.'
)

CORRECTNESS_PROMPT = (
    "You are a strict answer-correctness evaluator. Compare the candidate answer to "
    "the expected answer. Ignore style and hedging. Score factual overlap only. "
    "Do not consider whether the answer is supported by retrieved context. "
    'Reply with JSON only: {"label": "correct"}, {"label": "partial"}, '
    'or {"label": "incorrect"}.'
)

METRIC_NAMES = (
    "answerable_recall",
    "retrieval_hit_at_k",
    "retrieval_mrr",
    "retrieval_ndcg",
    "citation_precision",
    "citation_recall",
    "claim_citation_correctness",
    "groundedness",
    "answer_correctness",
    "correct_refusal_rate",
    "avg_cost_per_query",
    "avg_latency",
    "answering_cost_usd",
    "judging_cost_usd",
)


class GoldenRow(BaseModel):
    question: str
    expected_answer: str
    expected_source_ids: list[str] = Field(default_factory=list)
    answerable: bool = True


class JudgeVerdict(BaseModel):
    """One judge decision plus provenance for cache/audit."""

    model_config = ConfigDict(extra="ignore")

    score: float
    from_cache: bool
    model: str
    provider: str
    prompt_digest: str
    corpus_digest: str
    config_digest: str
    raw_output: str = ""
    timestamp: str
    schema_version: str
    fingerprint: str


@dataclass
class EvalResult:
    """Harness output: aggregates, intervals, spend, and judge provenance."""

    metrics: dict[str, float]
    split: str
    n: int
    intervals: dict[str, dict[str, float]] = field(default_factory=dict)
    per_row: dict[str, list[float]] = field(default_factory=dict)
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    repeats: list[dict[str, float]] = field(default_factory=list)


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


def config_hash(config: PipelineConfig) -> str:
    """Stable short hash of pipeline knobs (store dedup + config digest)."""
    payload = json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def question_cache_key(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()[:16]


def evaluation_fingerprint(
    *,
    config: PipelineConfig,
    question: str,
    answer: str,
    context: str,
    corpus_digest: str,
    judge_model: str,
    judge_provider: str,
    judge_prompt: str,
    schema_version: str,
    extra: dict[str, str] | None = None,
) -> str:
    """Hash of every input that can change a judge verdict."""
    payload: dict[str, Any] = {
        "config": config.model_dump(),
        "question": normalize_question(question),
        "answer": answer,
        "context_digest": digest_text(context),
        "corpus_digest": corpus_digest,
        "judge_model": judge_model,
        "judge_provider": judge_provider,
        "judge_prompt_digest": digest_text(judge_prompt),
        "schema_version": schema_version,
    }
    if extra:
        payload.update(extra)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def split_path(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}. Expected one of {SPLITS}")
    return DATA_DIR / f"{split}.jsonl"


def _load_jsonl(path: Path) -> list[GoldenRow]:
    rows: list[GoldenRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(GoldenRow.model_validate(json.loads(stripped)))
    return rows


def load_split(split: str) -> list[GoldenRow]:
    return _load_jsonl(split_path(split))


def load_golden(path: Path | None = None, *, split: str | None = None) -> list[GoldenRow]:
    """Load golden rows from a path, a named split, or all splits (dev, test, adversarial)."""
    if path is not None:
        return _load_jsonl(path)
    if split is not None:
        return load_split(split)
    rows: list[GoldenRow] = []
    for name in SPLITS:
        rows.extend(load_split(name))
    return rows


def set_deterministic_seeds(seed: int = 42) -> None:
    """Fix local RNGs. Does not make hosted-model inference deterministic."""
    random.seed(seed)
    np.random.seed(seed)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(index + 1) for index, rel in enumerate(relevances, start=1))


def retrieval_hit_at_k(expected_ids: list[str], retrieved_ids: list[str]) -> float:
    if not expected_ids:
        return 1.0
    return 1.0 if set(expected_ids).intersection(retrieved_ids) else 0.0


def retrieval_mrr(expected_ids: list[str], retrieved_ids: list[str]) -> float:
    if not expected_ids:
        return 1.0
    expected = set(expected_ids)
    for rank, source_id in enumerate(retrieved_ids, start=1):
        if source_id in expected:
            return 1.0 / rank
    return 0.0


def retrieval_ndcg(expected_ids: list[str], retrieved_ids: list[str]) -> float:
    if not expected_ids:
        return 1.0
    if not retrieved_ids:
        return 0.0
    expected = set(expected_ids)
    rels = [1.0 if item in expected else 0.0 for item in retrieved_ids]
    dcg = _dcg(rels)
    ideal = [1.0] * min(len(expected), len(retrieved_ids))
    idcg = _dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def citation_precision(expected_ids: list[str], cited_ids: list[str]) -> float:
    if not cited_ids:
        return 0.0
    hits = sum(1 for cited in cited_ids if cited in expected_ids)
    return hits / len(cited_ids)


def citation_recall(expected_ids: list[str], cited_ids: list[str]) -> float:
    if not expected_ids:
        return 1.0
    hits = sum(1 for expected in expected_ids if expected in cited_ids)
    return hits / len(expected_ids)


def claim_citation_correctness(
    expected_ids: list[str],
    cited_ids: list[str],
    retrieved_ids: list[str],
) -> float:
    """Fraction of cited IDs that appear in retrieved context and are gold sources.

    This is citation-id-level, not linguistic claim extraction. A retrieved-but-uncited
    gold source scores 0 here (and 0 citation recall) even if retrieval hit@k is 1.
    """
    if not cited_ids:
        return 0.0
    retrieved = set(retrieved_ids)
    expected = set(expected_ids)
    correct = 0
    for cited in cited_ids:
        if cited in retrieved and cited in expected:
            correct += 1
    return correct / len(cited_ids)


def deterministic_correctness(answer: str, expected: str) -> float | None:
    """Return 1.0 when the expected fact is contained in the answer; else None (use judge)."""
    expected_norm = normalize_question(expected)
    if not expected_norm:
        return None
    answer_norm = normalize_question(answer)
    if answer_norm == expected_norm or expected_norm in answer_norm:
        return 1.0
    return None


def bootstrap_mean_ci(
    values: list[float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap 95% CI of the mean. Width reflects example-set sampling, not model draws."""
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    rng = np.random.default_rng(seed)
    n = len(arr)
    samples = np.array([float(arr[rng.integers(0, n, n)].mean()) for _ in range(n_boot)])
    low = float(np.percentile(samples, 100.0 * alpha / 2.0))
    high = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "mean": round(mean, 4),
        "ci95_low": round(low, 4),
        "ci95_high": round(high, 4),
        "n": float(n),
        "std": round(std, 4),
    }


def repeat_run_variance(run_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Per-metric mean/std across independent harness repeats."""
    if not run_metrics:
        return {}
    keys = run_metrics[0].keys()
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(run[key]) for run in run_metrics]
        arr = np.asarray(values, dtype=float)
        out[key] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else 0.0,
            "n": float(len(arr)),
        }
    return out


def compute_metrics(
    rows: list[GoldenRow],
    results: list[QueryResult],
    *,
    groundedness_scores: list[float | None],
    correctness_scores: list[float | None],
    query_costs: list[float],
    query_latencies: list[float],
    answering_cost_usd: float = 0.0,
    judging_cost_usd: float = 0.0,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Aggregate per-row outcomes. Retrieval and citation are scored separately."""
    if len(rows) != len(results):
        raise ValueError("rows and results length mismatch")
    if not (
        len(groundedness_scores)
        == len(correctness_scores)
        == len(query_costs)
        == len(query_latencies)
        == len(rows)
    ):
        raise ValueError("per-row score lists must match rows")

    per_row: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}

    for row, result, grounded, correct in zip(
        rows, results, groundedness_scores, correctness_scores, strict=True
    ):
        if row.answerable:
            per_row["answerable_recall"].append(0.0 if result.refused else 1.0)
            per_row["answer_correctness"].append(0.0 if correct is None else float(correct))
        else:
            per_row["correct_refusal_rate"].append(1.0 if result.refused else 0.0)

        if row.expected_source_ids:
            per_row["retrieval_hit_at_k"].append(
                retrieval_hit_at_k(row.expected_source_ids, result.retrieved_ids)
            )
            per_row["retrieval_mrr"].append(
                retrieval_mrr(row.expected_source_ids, result.retrieved_ids)
            )
            per_row["retrieval_ndcg"].append(
                retrieval_ndcg(row.expected_source_ids, result.retrieved_ids)
            )
            if not result.refused:
                per_row["citation_precision"].append(
                    citation_precision(row.expected_source_ids, result.source_ids)
                )
                per_row["citation_recall"].append(
                    citation_recall(row.expected_source_ids, result.source_ids)
                )
                per_row["claim_citation_correctness"].append(
                    claim_citation_correctness(
                        row.expected_source_ids,
                        result.source_ids,
                        result.retrieved_ids,
                    )
                )

        if grounded is not None:
            per_row["groundedness"].append(float(grounded))

    query_count = len(results)
    metrics = {
        "answerable_recall": round(_mean(per_row["answerable_recall"]), 4)
        if per_row["answerable_recall"]
        else 1.0,
        "retrieval_hit_at_k": round(_mean(per_row["retrieval_hit_at_k"]), 4)
        if per_row["retrieval_hit_at_k"]
        else 1.0,
        "retrieval_mrr": round(_mean(per_row["retrieval_mrr"]), 4)
        if per_row["retrieval_mrr"]
        else 1.0,
        "retrieval_ndcg": round(_mean(per_row["retrieval_ndcg"]), 4)
        if per_row["retrieval_ndcg"]
        else 1.0,
        "citation_precision": round(_mean(per_row["citation_precision"]), 4)
        if per_row["citation_precision"]
        else 0.0,
        "citation_recall": round(_mean(per_row["citation_recall"]), 4)
        if per_row["citation_recall"]
        else 0.0,
        "claim_citation_correctness": round(_mean(per_row["claim_citation_correctness"]), 4)
        if per_row["claim_citation_correctness"]
        else 0.0,
        "groundedness": round(_mean(per_row["groundedness"]), 4),
        "answer_correctness": round(_mean(per_row["answer_correctness"]), 4)
        if per_row["answer_correctness"]
        else 0.0,
        "correct_refusal_rate": round(_mean(per_row["correct_refusal_rate"]), 4)
        if per_row["correct_refusal_rate"]
        else 1.0,
        "avg_cost_per_query": round(sum(query_costs) / query_count, 6) if query_count else 0.0,
        "avg_latency": round(sum(query_latencies) / query_count, 2) if query_count else 0.0,
        "answering_cost_usd": round(answering_cost_usd, 6),
        "judging_cost_usd": round(judging_cost_usd, 6),
    }
    return metrics, per_row


def print_metrics_table(metrics: dict[str, float]) -> None:
    """Print a fixed-width metrics table to stdout."""
    names = [name for name in METRIC_NAMES if name in metrics]
    name_width = max(len(name) for name in names)
    print(f"{'Metric':<{name_width}}  Value")
    print(f"{'-' * name_width}  -----")
    for name in names:
        value = metrics[name]
        if "cost" in name:
            print(f"{name:<{name_width}}  ${value:.6f}")
        elif name == "avg_latency":
            print(f"{name:<{name_width}}  {value:.2f} ms")
        else:
            print(f"{name:<{name_width}}  {value:.4f}")


def print_intervals_table(intervals: dict[str, dict[str, float]]) -> None:
    print()
    print("Example-bootstrap 95% CIs (sampling over this split, not model reruns):")
    name_width = max((len(name) for name in intervals), default=8)
    print(f"{'Metric':<{name_width}}  Mean    95% CI              n")
    for name, stats in intervals.items():
        print(
            f"{name:<{name_width}}  {stats['mean']:.4f}  "
            f"[{stats['ci95_low']:.4f}, {stats['ci95_high']:.4f}]  {int(stats['n'])}"
        )


class _CachedJudge:
    """Shared fingerprint cache for LLM-as-judge calls."""

    kind: str = "judge"
    prompt: str = ""
    schema_version: str = ""

    def __init__(
        self,
        llm: Any,
        config: PipelineConfig,
        *,
        corpus_digest: str = "",
        cache_dir: Path = CACHE_DIR,
        model_id: str | None = None,
        provider: str | None = None,
    ) -> None:
        self._llm = llm
        self._config = config
        self._corpus_digest = corpus_digest
        self._cache_dir = cache_dir / self.kind
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model_id = str(model_id or getattr(llm, "model_id", "unknown") or "unknown")
        self._provider = str(provider or getattr(llm, "provider", "unknown") or "unknown")

    def _fingerprint(self, question: str, answer: str, context: str, extra: dict[str, str]) -> str:
        return evaluation_fingerprint(
            config=self._config,
            question=question,
            answer=answer,
            context=context,
            corpus_digest=self._corpus_digest,
            judge_model=self._model_id,
            judge_provider=self._provider,
            judge_prompt=self.prompt,
            schema_version=self.schema_version,
            extra=extra,
        )

    def _load(self, path: Path, fingerprint: str) -> JudgeVerdict | None:
        if not path.exists():
            return None
        cached = json.loads(path.read_text(encoding="utf-8"))
        cached["from_cache"] = True
        cached["fingerprint"] = fingerprint
        return JudgeVerdict.model_validate(cached)

    def _store(self, path: Path, verdict: JudgeVerdict) -> None:
        payload = verdict.model_dump()
        payload["from_cache"] = False
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _complete(self, user_content: str) -> str:
        return str(
            self._llm.chat(
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                operation="judging",
                max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
            )
        )

    def _verdict(self, score: float, fingerprint: str, raw: str) -> JudgeVerdict:
        return JudgeVerdict(
            score=score,
            from_cache=False,
            model=self._model_id,
            provider=self._provider,
            prompt_digest=digest_text(self.prompt),
            corpus_digest=self._corpus_digest,
            config_digest=config_hash(self._config),
            raw_output=raw,
            timestamp=datetime.now(UTC).isoformat(),
            schema_version=self.schema_version,
            fingerprint=fingerprint,
        )


class GroundednessJudge(_CachedJudge):
    """Judge whether an answer is supported by retrieved context (not correctness)."""

    kind = "groundedness"
    prompt = GROUNDEDNESS_PROMPT
    schema_version = GROUNDEDNESS_SCHEMA_VERSION

    def score(self, question: str, answer: str, context: str, **_: Any) -> JudgeVerdict:
        fingerprint = self._fingerprint(question, answer, context, extra={})
        cache_path = self._cache_dir / f"{fingerprint}.json"
        cached = self._load(cache_path, fingerprint)
        if cached is not None:
            return cached

        user_content = f"Question: {question}\n\nContext:\n{context}\n\nAnswer: {answer}"
        raw = self._complete(user_content)
        supported = self._parse_supported(raw)
        verdict = self._verdict(1.0 if supported else 0.0, fingerprint, raw)
        self._store(cache_path, verdict)
        return verdict

    @staticmethod
    def _parse_supported(raw: str) -> bool:
        text = raw.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            lowered = text.lower()
            return "true" in lowered and "false" not in lowered
        return bool(data.get("supported", False))


class CorrectnessJudge(_CachedJudge):
    """Judge factual overlap with the expected answer (not groundedness)."""

    kind = "correctness"
    prompt = CORRECTNESS_PROMPT
    schema_version = CORRECTNESS_SCHEMA_VERSION

    def score(
        self,
        question: str,
        answer: str,
        context: str,
        *,
        expected_answer: str = "",
        **_: Any,
    ) -> JudgeVerdict:
        extra = {"expected_answer": expected_answer}
        fingerprint = self._fingerprint(question, answer, context, extra=extra)
        cache_path = self._cache_dir / f"{fingerprint}.json"
        cached = self._load(cache_path, fingerprint)
        if cached is not None:
            return cached

        exact = deterministic_correctness(answer, expected_answer)
        if exact is not None:
            verdict = self._verdict(exact, fingerprint, raw="deterministic:containment")
            verdict.model = "deterministic"
            verdict.provider = "local"
            self._store(cache_path, verdict)
            return verdict

        user_content = (
            f"Question: {question}\n\n"
            f"Expected answer: {expected_answer}\n\n"
            f"Candidate answer: {answer}"
        )
        raw = self._complete(user_content)
        label = self._parse_label(raw)
        score = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(label, 0.0)
        verdict = self._verdict(score, fingerprint, raw)
        self._store(cache_path, verdict)
        return verdict

    @staticmethod
    def _parse_label(raw: str) -> str:
        text = raw.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            lowered = text.lower()
            for label in ("incorrect", "partial", "correct"):
                if label in lowered:
                    return label
            return "incorrect"
        label = str(data.get("label", "incorrect")).lower()
        if label not in {"correct", "partial", "incorrect"}:
            return "incorrect"
        return label


def _as_score(value: JudgeVerdict | float) -> float:
    return float(value.score) if isinstance(value, JudgeVerdict) else float(value)


def _zero_metrics() -> dict[str, float]:
    return {name: 0.0 for name in METRIC_NAMES}


class EvalHarness:
    """Runs a named split (or JSONL path) and returns aggregate scores."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        config: PipelineConfig,
        *,
        golden_path: Path | None = None,
        split: str | None = None,
        judge: Any = None,
        groundedness_judge: Any = None,
        correctness_judge: Any = None,
        corpus_digest: str = "",
        seed: int = 42,
    ) -> None:
        self._pipeline = pipeline
        self._config = config
        self._golden_path = golden_path
        self._split = split
        self._seed = seed
        self._corpus_digest = corpus_digest
        self._groundedness_judge = (
            groundedness_judge
            or judge
            or GroundednessJudge(pipeline._llm, config, corpus_digest=corpus_digest)
        )
        self._correctness_judge = correctness_judge or CorrectnessJudge(
            pipeline._llm, config, corpus_digest=corpus_digest
        )

    def run(self) -> EvalResult:
        """Execute the eval split and return metrics plus provenance."""
        set_deterministic_seeds(self._seed)
        if self._golden_path is not None:
            rows = load_golden(self._golden_path)
            split_name = "path"
        else:
            split_name = self._split or "dev"
            rows = load_split(split_name)
        if not rows:
            return EvalResult(metrics=_zero_metrics(), split=split_name, n=0)

        results: list[QueryResult] = []
        groundedness_scores: list[float | None] = []
        correctness_scores: list[float | None] = []
        query_costs: list[float] = []
        query_latencies: list[float] = []
        provenance: list[dict[str, Any]] = []
        usage = self._pipeline.usage
        answering_before = usage.cost_for("answering")
        judging_before = usage.cost_for("judging")

        for row in rows:
            cost_before = usage.total_cost_usd
            started = time.perf_counter()
            result = self._pipeline.query(row.question)

            if result.refused:
                groundedness_scores.append(None)
                correctness_scores.append(0.0 if row.answerable else None)
            else:
                g_verdict = self._groundedness_judge.score(
                    result.question, result.answer, result.context
                )
                groundedness_scores.append(_as_score(g_verdict))
                if isinstance(g_verdict, JudgeVerdict):
                    provenance.append({"kind": "groundedness", **g_verdict.model_dump()})
                if row.answerable:
                    c_verdict = self._correctness_judge.score(
                        result.question,
                        result.answer,
                        result.context,
                        expected_answer=row.expected_answer,
                    )
                    correctness_scores.append(_as_score(c_verdict))
                    if isinstance(c_verdict, JudgeVerdict):
                        provenance.append({"kind": "correctness", **c_verdict.model_dump()})
                else:
                    correctness_scores.append(None)

            elapsed_ms = (time.perf_counter() - started) * 1000
            results.append(result)
            query_costs.append(usage.total_cost_usd - cost_before)
            query_latencies.append(elapsed_ms)

        answering_cost = usage.cost_for("answering") - answering_before
        judging_cost = usage.cost_for("judging") - judging_before
        metrics, per_row = compute_metrics(
            rows,
            results,
            groundedness_scores=groundedness_scores,
            correctness_scores=correctness_scores,
            query_costs=query_costs,
            query_latencies=query_latencies,
            answering_cost_usd=answering_cost,
            judging_cost_usd=judging_cost,
        )
        intervals = {
            name: bootstrap_mean_ci(values)
            for name, values in per_row.items()
            if values
            and name
            not in {"avg_cost_per_query", "avg_latency", "answering_cost_usd", "judging_cost_usd"}
        }
        return EvalResult(
            metrics=metrics,
            split=split_name,
            n=len(rows),
            intervals=intervals,
            per_row=per_row,
            cost=CostBreakdown(answering_usd=answering_cost, judging_usd=judging_cost),
            provenance=provenance,
        )

    def run_repeated(self, n_repeats: int) -> EvalResult:
        """Run the split ``n_repeats`` times and attach run-to-run variance."""
        if n_repeats < 1:
            raise ValueError("n_repeats must be >= 1")
        last = self.run()
        repeats = [last.metrics]
        for _ in range(n_repeats - 1):
            repeats.append(self.run().metrics)
        last.repeats = repeats
        return last


def evaluate_config(
    settings: Settings,
    config: PipelineConfig,
    *,
    split: str = "dev",
    golden_path: Path | None = None,
    corpus: str | None = None,
) -> EvalResult:
    """Index the sample corpus and evaluate ``config`` on a split."""
    set_deterministic_seeds()
    corpus_text = corpus if corpus is not None else load_sample_corpus()
    corpus_digest = digest_text(corpus_text)
    pipeline = RAGPipeline(settings, config)
    pipeline.index_corpus(corpus_text)
    model_id = getattr(pipeline._llm, "model_id", settings.llm_model)
    provider = getattr(pipeline._llm, "provider", settings.llm_provider)
    harness = EvalHarness(
        pipeline,
        config,
        golden_path=golden_path,
        split=None if golden_path is not None else split,
        corpus_digest=corpus_digest,
        groundedness_judge=GroundednessJudge(
            pipeline._llm,
            config,
            corpus_digest=corpus_digest,
            model_id=model_id,
            provider=provider,
        ),
        correctness_judge=CorrectnessJudge(
            pipeline._llm,
            config,
            corpus_digest=corpus_digest,
            model_id=model_id,
            provider=provider,
        ),
    )
    return harness.run()


def run_eval(
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
    *,
    golden_path: Path | None = None,
    split: str = "test",
) -> EvalResult:
    """Index the sample corpus and run a golden split (default: held-out test)."""
    settings = settings or Settings()
    config = config or PipelineConfig()
    return evaluate_config(settings, config, split=split, golden_path=golden_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m autorag.eval`."""
    import argparse

    parser = argparse.ArgumentParser(description="AutoRAG golden eval")
    parser.add_argument(
        "--split",
        choices=[*SPLITS, "held-out"],
        default="held-out",
        help="dev = tuning set; test/adversarial = reporting; held-out = test + adversarial",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write versioned JSON result")
    parser.add_argument("--repeats", type=int, default=1, help="Independent harness repeats")
    args = parser.parse_args(argv)

    settings = Settings()
    config = PipelineConfig()
    payload: dict[str, Any] = {
        "schema_version": f"{GROUNDEDNESS_SCHEMA_VERSION}+{CORRECTNESS_SCHEMA_VERSION}",
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config.model_dump(),
        "model": settings.llm_model,
        "provider": settings.llm_provider,
        "corpus_digest": digest_text(load_sample_corpus()),
        "splits": {},
        "note": (
            "Local chunking/embeddings/metrics can be seeded. Hosted LLM inference is not "
            "deterministic even at temperature=0. Cached judge scores repeat stored values "
            "and can hide model drift."
        ),
    }

    splits = ["test", "adversarial"] if args.split == "held-out" else [args.split]
    for name in splits:
        result = run_eval(settings, config, split=name)
        if args.repeats > 1:
            # Re-run via a fresh evaluate for variance; first result already captured.
            repeats = [result.metrics]
            for _ in range(args.repeats - 1):
                repeats.append(run_eval(settings, config, split=name).metrics)
            result.repeats = repeats
        print()
        print(f"=== split={name} n={result.n} ===")
        print_metrics_table(result.metrics)
        print_intervals_table(result.intervals)
        if result.repeats and len(result.repeats) > 1:
            print()
            print("Repeat-run variance:")
            for metric, stats in repeat_run_variance(result.repeats).items():
                print(
                    f"  {metric}: mean={stats['mean']:.4f} std={stats['std']:.4f} n={int(stats['n'])}"
                )
        payload["splits"][name] = {
            "n": result.n,
            "metrics": result.metrics,
            "intervals": result.intervals,
            "cost": {
                "answering_usd": result.cost.answering_usd,
                "judging_usd": result.cost.judging_usd,
            },
            "repeat_variance": repeat_run_variance(result.repeats) if result.repeats else {},
            "provenance": result.provenance,
        }

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
