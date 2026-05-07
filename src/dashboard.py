"""
Results Dashboard

Stores eval run results to JSON and prints rich terminal output.
Tracks history across multiple runs so you can answer:
  "When did Telugu quality drop, and which commit caused it?"

Each eval run is saved as: eval_results/run_{timestamp}.json
A summary history is maintained in: eval_results/history.json
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

RESULTS_DIR = Path(__file__).parent.parent / "eval_results"
HISTORY_FILE = RESULTS_DIR / "history.json"

console = Console()


@dataclass
class RunSummary:
    run_id: str
    timestamp: str
    baseline_provider: str
    baseline_model: str
    baseline_prompt: str
    candidate_provider: str
    candidate_model: str
    candidate_prompt: str
    gate_decision: str   # GO / NO-GO / INCONCLUSIVE
    deal_copy_baseline_mean: float
    deal_copy_candidate_mean: float
    deal_copy_diff: float
    insurance_baseline_mean: float
    insurance_candidate_mean: float
    insurance_diff: float
    credit_baseline_mean: float
    credit_candidate_mean: float
    credit_diff: float
    total_cost_usd: float
    total_cases: int
    git_commit: str = "none"
    notes: str = ""


def save_run(
    run_id: str,
    baseline_results: dict[str, list],
    candidate_results: dict[str, list],
    comparison_results: dict,
    gate_decision_obj: Any,
    metadata: dict,
) -> Path:
    """Persist full run data to JSON."""
    RESULTS_DIR.mkdir(exist_ok=True)

    def _serialize_list(lst):
        out = []
        for item in lst:
            if hasattr(item, "__dataclass_fields__"):
                out.append(asdict(item))
            else:
                out.append(item)
        return out

    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "gate_decision": gate_decision_obj.decision if gate_decision_obj else "UNKNOWN",
        "gate_summary": gate_decision_obj.summary if gate_decision_obj else "",
        "baseline_results": {k: _serialize_list(v) for k, v in baseline_results.items()},
        "candidate_results": {k: _serialize_list(v) for k, v in candidate_results.items()},
        "comparison_results": comparison_results,
    }

    run_file = RESULTS_DIR / f"run_{run_id}.json"
    run_file.write_text(json.dumps(payload, indent=2, default=str))
    return run_file


def _mean(lst, field):
    vals = [getattr(r, field) for r in lst if hasattr(r, field)]
    return sum(vals) / len(vals) if vals else 0.0


def update_history(summary: RunSummary) -> None:
    """Append this run summary to the history file."""
    RESULTS_DIR.mkdir(exist_ok=True)
    history = []
    if HISTORY_FILE.exists():
        history = json.loads(HISTORY_FILE.read_text())
    history.append(asdict(summary))
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def print_run_report(
    run_id: str,
    baseline_results: dict,
    candidate_results: dict,
    comparison: dict,
    gate,
    metadata: dict,
) -> None:
    """Print a rich terminal report of the eval run."""

    # ── Header ─────────────────────────────────────────────────────────────────
    decision_color = {"GO": "green", "NO-GO": "red", "INCONCLUSIVE": "yellow"}.get(
        gate.decision, "white"
    )
    console.print(
        Panel(
            f"[bold {decision_color}]  {gate.decision}[/]\n\n{gate.summary}",
            title=f"[bold]The Guard — Eval Run {run_id}[/]",
            border_style=decision_color,
        )
    )

    # ── Per-task scores ─────────────────────────────────────────────────────────
    table = Table(title="Task Scores", box=box.ROUNDED)
    table.add_column("Task", style="cyan")
    table.add_column("Scorer", style="blue")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Verdict")

    for task_key, comp_data in comparison.items():
        for scorer_key, result_data in comp_data.items():
            baseline_mean = result_data.get("baseline_mean", "—")
            candidate_mean = result_data.get("candidate_mean", "—")
            diff = result_data.get("absolute_diff", 0)
            p_val = result_data.get("p_value", 1.0)
            verdict = result_data.get("verdict", "—")

            diff_str = f"{diff:+.4f}" if isinstance(diff, float) else "—"
            diff_color = "red" if isinstance(diff, float) and diff < -0.02 else (
                "green" if isinstance(diff, float) and diff > 0.02 else "white"
            )
            verdict_color = {"IMPROVED": "green", "REGRESSED": "red", "NO_CHANGE": "yellow"}.get(
                verdict, "white"
            )

            table.add_row(
                task_key,
                scorer_key,
                f"{baseline_mean:.4f}" if isinstance(baseline_mean, float) else str(baseline_mean),
                f"{candidate_mean:.4f}" if isinstance(candidate_mean, float) else str(candidate_mean),
                f"[{diff_color}]{diff_str}[/]",
                f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val),
                f"[{verdict_color}]{verdict}[/]",
            )

    console.print(table)

    # ── Cost summary ────────────────────────────────────────────────────────────
    all_costs = []
    for res_dict in [baseline_results, candidate_results]:
        for res_list in res_dict.values():
            all_costs.extend(getattr(r, "cost_usd", 0) for r in res_list)
    total_cost = sum(all_costs)

    console.print(
        f"\n[dim]Total cost this run: ${total_cost:.4f} USD | "
        f"Run ID: {run_id} | "
        f"Metadata: {metadata.get('baseline_model', '?')} vs {metadata.get('candidate_model', '?')}[/]"
    )


def print_history_table() -> None:
    """Print a summary of all past eval runs."""
    if not HISTORY_FILE.exists():
        console.print("[yellow]No history found. Run an eval first.[/]")
        return

    history = json.loads(HISTORY_FILE.read_text())
    table = Table(title="Eval History", box=box.SIMPLE)
    table.add_column("Run ID", style="dim")
    table.add_column("Timestamp")
    table.add_column("Baseline")
    table.add_column("Candidate")
    table.add_column("Deal Δ", justify="right")
    table.add_column("Ins Δ", justify="right")
    table.add_column("Credit Δ", justify="right")
    table.add_column("Decision")

    for run in history[-20:]:  # Show last 20 runs
        dec = run.get("gate_decision", "?")
        dec_color = {"GO": "green", "NO-GO": "red", "INCONCLUSIVE": "yellow"}.get(dec, "white")
        d_diff = run.get("deal_copy_diff", 0)
        i_diff = run.get("insurance_diff", 0)
        c_diff = run.get("credit_diff", 0)

        def fmt_diff(v):
            if not isinstance(v, float):
                return "—"
            color = "red" if v < -0.01 else ("green" if v > 0.01 else "white")
            return f"[{color}]{v:+.3f}[/]"

        table.add_row(
            run.get("run_id", "?")[:12],
            run.get("timestamp", "")[:16],
            f"{run.get('baseline_model', '?')} / {run.get('baseline_prompt', '?')}",
            f"{run.get('candidate_model', '?')} / {run.get('candidate_prompt', '?')}",
            fmt_diff(d_diff),
            fmt_diff(i_diff),
            fmt_diff(c_diff),
            f"[{dec_color}]{dec}[/]",
        )

    console.print(table)
