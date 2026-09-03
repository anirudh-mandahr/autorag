"""Autoresearch loop: LLM-proposed PipelineConfig mutations under a fixed budget."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from autorag.budget import (
    RESEARCHER_MAX_OUTPUT_TOKENS,
    CostBreakdown,
    SpendLedger,
    estimate_eval_cost_usd,
    estimate_researcher_cost_usd,
)
from autorag.config import (
    PipelineConfig,
    Settings,
    compute_objective,
    get_search_space,
)
from autorag.eval import (
    METRIC_NAMES,
    EvalResult,
    evaluate_config,
    load_split,
    print_intervals_table,
    print_metrics_table,
)
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
    print(
        f"Tokens: {metrics['total_tokens']} (in={metrics['input_tokens']}, out={metrics['output_tokens']})"
    )
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

    for attempt in range(1, MAX_RESEARCHER_RETRIES + 1):
        context = build_researcher_context(
            search_space=search_space,
            best=best,
            recent=store.recent(limit=12),
        )
        duplicate_note = ""
        if attempt > 1:
            duplicate_note = (
                "\n\nYour last proposal was invalid or already tried. Propose a different config."
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
            operation="researcher",
            max_tokens=RESEARCHER_MAX_OUTPUT_TOKENS,
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


def held_out_reserve_usd(settings: Settings) -> float:
    """Budget held back so final test + adversarial eval can still run."""
    n_held_out = len(load_split("test")) + len(load_split("adversarial"))
    return estimate_eval_cost_usd(settings.llm_model, n_held_out)


def run_experiment(
    settings: Settings,
    config: PipelineConfig,
    *,
    split: str = "dev",
) -> tuple[dict[str, float], float]:
    """Evaluate on a split; return metrics and total USD (answering + judging)."""
    result = evaluate_config(settings, config, split=split)
    return result.metrics, result.cost.total_usd


def _print_metric_delta_table(
    winner_metrics: dict[str, float], baseline_metrics: dict[str, float]
) -> None:
    print(f"{'Metric':<32} {'Winner':>12} {'Baseline':>12} {'Delta':>12}")
    print(f"{'-' * 32} {'-' * 12} {'-' * 12} {'-' * 12}")
    names = [name for name in METRIC_NAMES if name in winner_metrics and name in baseline_metrics]
    extra = [name for name in winner_metrics if name not in names and name in baseline_metrics]
    for name in names + extra:
        win_val = winner_metrics[name]
        base_val = baseline_metrics[name]
        delta = win_val - base_val
        if "cost" in name:
            print(f"{name:<32} ${win_val:>11.6f} ${base_val:>11.6f} ${delta:>+11.6f}")
        elif name == "avg_latency":
            print(f"{name:<32} {win_val:>11.2f}ms {base_val:>11.2f}ms {delta:>+11.2f}ms")
        else:
            print(f"{name:<32} {win_val:>12.4f} {base_val:>12.4f} {delta:>+12.4f}")


def print_final_report(
    *,
    winner: ExperimentRecord,
    baseline: ExperimentRecord,
    ledger: SpendLedger,
    held_out: dict[str, EvalResult] | None = None,
) -> None:
    print()
    print("=" * 72)
    print("RESEARCH COMPLETE")
    print("=" * 72)
    print()
    print("Winning config (selected on the development split):")
    print(json.dumps(winner.config.model_dump(), indent=2))
    print()
    print("Development-split metrics used for keep/discard (not a generalization claim):")
    _print_metric_delta_table(winner.metrics, baseline.metrics)
    print()
    print(
        f"Dev objective: {winner.objective:.4f} (baseline {baseline.objective:.4f}, "
        f"delta {winner.objective - baseline.objective:+.4f})"
    )
    print(
        f"Session spend: ${ledger.total_usd:.4f} "
        f"(researcher ${ledger.researcher_usd:.4f}, "
        f"answering ${ledger.answering_usd:.4f}, "
        f"judging ${ledger.judging_usd:.4f})"
    )
    if held_out:
        print()
        print("Held-out reporting (not used to pick the winner):")
        for split_name, result in held_out.items():
            print()
            print(f"--- {split_name}  n={result.n} ---")
            print_metrics_table(result.metrics)
            print_intervals_table(result.intervals)
        print()
        print(
            "Intervals are bootstrap 95% CIs over examples in that split. "
            "They do not measure hosted-model nondeterminism. Small n means "
            "wide intervals; a higher dev objective is not evidence of a "
            "general improvement."
        )
    else:
        print()
        print("Held-out eval skipped (insufficient remaining budget or disabled).")
    print()
    print(
        "Hosted LLM calls are not deterministic even at temperature=0. "
        "Cached judge verdicts replay stored scores and can hide model drift."
    )


def _dev_eval_estimate(settings: Settings) -> float:
    return estimate_eval_cost_usd(settings.llm_model, len(load_split("dev")))


def run_research(
    settings: Settings,
    *,
    budget: int,
    store_path: Path | None = None,
    fresh: bool = False,
    report_held_out: bool = True,
) -> int:
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 1

    program_md = load_program_md()
    if fresh and store_path and store_path.exists():
        store_path.unlink()

    store = ExperimentStore(store_path)
    ledger = SpendLedger()
    session_experiments = 0
    researcher_usage = UsageTracker()
    holdback = held_out_reserve_usd(settings) if report_held_out else 0.0
    cap = settings.max_usd_spend
    researcher_est = estimate_researcher_cost_usd(settings.llm_model)
    eval_est = _dev_eval_estimate(settings)

    try:
        baseline_record = store.baseline()
        if baseline_record is None:
            if not ledger.can_reserve(cap, eval_est, holdback=holdback):
                print(
                    "Spend cap: cannot reserve baseline evaluation under the hard cap.",
                    file=sys.stderr,
                )
                return 1
            baseline_config = PipelineConfig()
            print("Running baseline eval on the development split...", flush=True)
            result = evaluate_config(settings, baseline_config, split="dev")
            ledger.add(result.cost)
            session_experiments += 1
            baseline_record = store.insert(
                config=baseline_config,
                metrics=result.metrics,
                objective=compute_objective(result.metrics),
                status="kept",
                parent_id=None,
                cost_usd=result.cost.total_usd,
                is_baseline=True,
            )
            log_experiment_line(
                index=session_experiments,
                delta="baseline",
                objective=baseline_record.objective,
                delta_obj=None,
                status="keep",
                cost_usd=result.cost.total_usd,
                session_spend=ledger.total_usd,
            )

        best = store.best() or baseline_record
        previous_config = best.config

        while session_experiments < budget:
            if not ledger.can_reserve(cap, researcher_est, holdback=holdback):
                print("Spend cap: cannot reserve researcher proposal.", flush=True)
                break

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
            ledger.add(CostBreakdown(researcher_usd=researcher_cost))
            if not ledger.can_reserve(cap, 0.0, holdback=0.0) and ledger.remaining(cap) < 0:
                print("Spend cap reached during researcher call.", flush=True)
                break

            if not ledger.can_reserve(cap, eval_est, holdback=holdback):
                print("Spend cap: cannot reserve development evaluation.", flush=True)
                break

            result = evaluate_config(settings, config, split="dev")
            ledger.add(result.cost)
            session_experiments += 1
            cost_usd = result.cost.total_usd + researcher_cost

            objective = compute_objective(result.metrics)
            best_before = best.objective
            kept = objective > best.objective
            status = "kept" if kept else "discarded"

            record = store.insert(
                config=config,
                metrics=result.metrics,
                objective=objective,
                status=status,
                parent_id=best.id,
                cost_usd=cost_usd,
            )

            log_experiment_line(
                index=session_experiments,
                delta=format_config_delta(previous_config, config),
                objective=objective,
                delta_obj=objective - best_before,
                status=status,
                cost_usd=cost_usd,
                session_spend=ledger.total_usd,
            )

            if kept:
                best = record
                previous_config = config

        winner = store.best() or baseline_record
        held_out: dict[str, EvalResult] | None = None
        if report_held_out:
            held_out = {}
            for split_name in ("test", "adversarial"):
                n = len(load_split(split_name))
                split_est = estimate_eval_cost_usd(settings.llm_model, n)
                if not ledger.can_reserve(cap, split_est, holdback=0.0):
                    print(
                        f"Spend cap: cannot reserve held-out split {split_name}.",
                        flush=True,
                    )
                    if not held_out:
                        held_out = None
                    break
                split_result = evaluate_config(settings, winner.config, split=split_name)
                ledger.add(split_result.cost)
                held_out[split_name] = split_result

        print_final_report(
            winner=winner,
            baseline=baseline_record,
            ledger=ledger,
            held_out=held_out,
        )
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
    parser.add_argument(
        "--no-held-out",
        action="store_true",
        help="Skip held-out test/adversarial reporting (still holds no reserve)",
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
        report_held_out=not args.no_held_out,
    )


if __name__ == "__main__":
    sys.exit(main())
