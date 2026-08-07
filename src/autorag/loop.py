"""Autoresearch loop: LLM-proposed PipelineConfig mutations under a fixed budget."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from autorag.config import (
    PipelineConfig,
    Settings,
    compute_objective,
    get_search_space,
)
from autorag.eval import EvalHarness
from autorag.llm import LLMClient
from autorag.pipeline import RAGPipeline, load_sample_corpus, load_sample_questions
from autorag.store import ExperimentRecord, ExperimentStore
from autorag.usage import UsageTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
PROGRAM_MD = REPO_ROOT / "program.md"
MAX_RESEARCHER_RETRIES = 5


def run_once(settings: Settings, config: PipelineConfig) -> int:
    """Index the sample corpus and answer the first golden question."""
    pipeline = RAGPipeline(settings, config)
    corpus = load_sample_corpus()
    questions = load_sample_questions()
    if not questions:
        print("No questions found in data/questions.json", file=sys.stderr)
        return 1

    chunk_count = pipeline.index_corpus(corpus)
    question = questions[0]
    result = pipeline.query(question["question"])
    metrics = pipeline.metrics

    print(f"Question ({question['id']}): {result.question}")
    print(f"Answer: {result.answer}")
    print(f"Citations: {', '.join(result.source_ids) if result.source_ids else '(none)'}")
    print(f"Refused: {result.refused}")
    print(f"Top similarity: {result.top_similarity:.4f}")
    print(f"Indexed chunks: {chunk_count}")
    print(f"Latency (ms): {metrics['latency_ms']}")
    print(f"Tokens: {metrics['total_tokens']} (in={metrics['input_tokens']}, out={metrics['output_tokens']})")
    print(f"Cost (USD): ${metrics['cost_usd']:.6f}")
    return 0


def load_program_md(path: Path = PROGRAM_MD) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Research program not found: {path}")
    return path.read_text(encoding="utf-8")


def format_config_delta(previous: PipelineConfig | None, current: PipelineConfig) -> str:
    if previous is None:
        return "baseline"
    parts: list[str] = []
    for key, value in current.model_dump().items():
        old = getattr(previous, key)
        if old != value:
            parts.append(f"{key}:{old}->{value}")
    return " ".join(parts) if parts else "(unchanged)"


def log_experiment_line(
    *,
    index: int,
    delta: str,
    objective: float,
    delta_obj: float | None,
    status: str,
    cost_usd: float,
    session_spend: float,
) -> None:
    delta_str = f"{delta_obj:+.4f}" if delta_obj is not None else "n/a"
    print(
        f"exp={index:02d}  {delta:<48}  "
        f"objective={objective:.4f} ({delta_str})  {status.upper():8}  "
        f"cost=${cost_usd:.4f}  spend=${session_spend:.4f}",
        flush=True,
    )


def build_researcher_context(
    *,
    search_space: dict,
    best: ExperimentRecord | None,
    recent: list[ExperimentRecord],
) -> str:
    payload = {
        "search_space": search_space,
        "best_so_far": best.to_summary() if best else None,
        "recent_experiments": [record.to_summary() for record in recent],
    }
    return json.dumps(payload, indent=2)


def parse_pipeline_config(raw: str) -> PipelineConfig:
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise ValueError(f"Researcher did not return JSON: {raw!r}") from None
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Researcher JSON must be an object")
    return PipelineConfig.model_validate(data)


def propose_next_config(
    *,
    settings: Settings,
    program_md: str,
    store: ExperimentStore,
    best: ExperimentRecord | None,
    usage: UsageTracker,
) -> PipelineConfig:
    llm = LLMClient(settings, PipelineConfig(), usage=usage)
    search_space = get_search_space()
    seen = store.seen_config_hashes()
    parent = best.config if best else PipelineConfig()

    for attempt in range(1, MAX_RESEARCHER_RETRIES + 1):
        context = build_researcher_context(
            search_space=search_space,
            best=best,
            recent=store.recent(limit=12),
        )
        duplicate_note = ""
        if attempt > 1:
            duplicate_note = (
                "\n\nYour last proposal was invalid or already tried. "
                "Propose a different config."
            )
        messages = [
            {"role": "system", "content": program_md},
            {
                "role": "user",
                "content": (
                    "Given the context below, output the NEXT PipelineConfig as strict JSON only.\n\n"
                    f"{context}{duplicate_note}"
                ),
            },
        ]
        raw = llm.chat(
            messages=messages,
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        try:
            config = parse_pipeline_config(raw)
        except (ValidationError, ValueError):
            continue
        if store.hash_config(config) in seen:
            continue
        if config.chunk_overlap >= config.chunk_size:
            continue
        return config

    raise RuntimeError(
        f"Researcher failed to propose a novel valid config after {MAX_RESEARCHER_RETRIES} attempts"
    )


def run_experiment(
    settings: Settings,
    config: PipelineConfig,
) -> tuple[dict[str, float], float]:
    """Run golden eval; return metrics and total USD cost for the experiment."""
    pipeline = RAGPipeline(settings, config)
    cost_before = pipeline.usage.total_cost_usd
    pipeline.index_corpus(load_sample_corpus())
    metrics = EvalHarness(pipeline, config).run()
    cost_usd = pipeline.usage.total_cost_usd - cost_before
    return metrics, cost_usd


def print_final_report(
    *,
    winner: ExperimentRecord,
    baseline: ExperimentRecord,
) -> None:
    print()
    print("=" * 72)
    print("RESEARCH COMPLETE")
    print("=" * 72)
    print()
    print("Winning config:")
    print(json.dumps(winner.config.model_dump(), indent=2))
    print()
    print(f"{'Metric':<28} {'Winner':>12} {'Baseline':>12} {'Delta':>12}")
    print(f"{'-' * 28} {'-' * 12} {'-' * 12} {'-' * 12}")
    for name in winner.metrics:
        win_val = winner.metrics[name]
        base_val = baseline.metrics[name]
        delta = win_val - base_val
        if name == "avg_cost_per_query":
            print(f"{name:<28} ${win_val:>11.6f} ${base_val:>11.6f} ${delta:>+11.6f}")
        elif name == "avg_latency":
            print(f"{name:<28} {win_val:>11.2f}ms {base_val:>11.2f}ms {delta:>+11.2f}ms")
        else:
            print(f"{name:<28} {win_val:>12.4f} {base_val:>12.4f} {delta:>+12.4f}")
    print()
    print(
        f"Objective: {winner.objective:.4f} (baseline {baseline.objective:.4f}, "
        f"delta {winner.objective - baseline.objective:+.4f})"
    )
    print(f"Status: {'IMPROVED' if winner.objective > baseline.objective else 'NO IMPROVEMENT'}")


def run_research(
    settings: Settings,
    *,
    budget: int,
    store_path: Path | None = None,
    fresh: bool = False,
) -> int:
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 1

    program_md = load_program_md()
    if fresh and store_path and store_path.exists():
        store_path.unlink()

    store = ExperimentStore(store_path)
    session_spend = 0.0
    session_experiments = 0
    researcher_usage = UsageTracker()

    try:
        baseline_record = store.baseline()
        if baseline_record is None:
            baseline_config = PipelineConfig()
            print("Running baseline eval...", flush=True)
            metrics, cost_usd = run_experiment(settings, baseline_config)
            objective = compute_objective(metrics)
            session_spend += cost_usd
            session_experiments += 1
            baseline_record = store.insert(
                config=baseline_config,
                metrics=metrics,
                objective=objective,
                status="kept",
                parent_id=None,
                cost_usd=cost_usd,
                is_baseline=True,
            )
            log_experiment_line(
                index=session_experiments,
                delta="baseline",
                objective=objective,
                delta_obj=None,
                status="keep",
                cost_usd=cost_usd,
                session_spend=session_spend,
            )
            if session_experiments >= budget or session_spend >= settings.max_usd_spend:
                print_final_report(winner=baseline_record, baseline=baseline_record)
                return 0

        best = store.best() or baseline_record
        previous_config = best.config

        while session_experiments < budget and session_spend < settings.max_usd_spend:
            researcher_cost_before = researcher_usage.total_cost_usd
            try:
                config = propose_next_config(
                    settings=settings,
                    program_md=program_md,
                    store=store,
                    best=best,
                    usage=researcher_usage,
                )
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                break

            researcher_cost = researcher_usage.total_cost_usd - researcher_cost_before
            session_spend += researcher_cost
            if session_spend >= settings.max_usd_spend:
                print("Spend cap reached during researcher call.", flush=True)
                break

            metrics, eval_cost = run_experiment(settings, config)
            cost_usd = eval_cost + researcher_cost
            session_spend += eval_cost
            session_experiments += 1

            objective = compute_objective(metrics)
            best_before = best.objective
            kept = objective > best.objective
            status = "kept" if kept else "discarded"
            parent_id = best.id

            record = store.insert(
                config=config,
                metrics=metrics,
                objective=objective,
                status=status,
                parent_id=parent_id,
                cost_usd=cost_usd,
            )

            log_experiment_line(
                index=session_experiments,
                delta=format_config_delta(previous_config, config),
                objective=objective,
                delta_obj=objective - best_before,
                status=status,
                cost_usd=cost_usd,
                session_spend=session_spend,
            )

            if kept:
                best = record
                previous_config = config

            if session_spend >= settings.max_usd_spend:
                print("Spend cap reached.", flush=True)
                break

        winner = store.best() or baseline_record
        print_final_report(winner=winner, baseline=baseline_record)
        return 0
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    """Entry point for run-once and research targets."""
    parser = argparse.ArgumentParser(description="AutoRAG research loop")
    parser.add_argument("--once", action="store_true", help="Run a single demo query")
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Max experiments this session (overrides MAX_EXPERIMENTS)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear experiment store before starting",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Path to SQLite experiment database",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    budget = args.budget if args.budget is not None else settings.max_experiments

    if args.once:
        return run_once(settings, PipelineConfig())

    return run_research(
        settings,
        budget=budget,
        store_path=args.store,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    sys.exit(main())
