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


def _compute_worst_cases(
    baseline_results: dict,
    candidate_results: dict,
    comparison: dict,
) -> str:
    """
    Compute per-task worst-performing case IDs for regressed tasks.
    Returns a formatted string listing case IDs where candidate lost, per task.
    """
    task_map = {
        "deal_copy": ("composite_score", "deal_copy"),
        "insurance": ("intent_score",    "insurance"),
        "credit":    ("composite_score", "credit"),
    }
    lines = []
    for task_name, (score_field, res_key) in task_map.items():
        task_comp = comparison.get(task_name, {})
        task_has_regression = any(v.get("verdict") == "REGRESSED" for v in task_comp.values())
        if not task_has_regression:
            continue
        b_list = baseline_results.get(res_key, [])
        c_list = candidate_results.get(res_key, [])
        worse_cases = []
        for b, c in zip(b_list, c_list):
            b_score = getattr(b, score_field, None)
            c_score = getattr(c, score_field, None)
            if b_score is None or c_score is None:
                continue
            if c_score - b_score < -0.02:
                worse_cases.append(getattr(b, "case_id", "?"))
        if worse_cases:
            lines.append(f"  {task_name}: {', '.join(worse_cases)}")
    return "\n".join(lines)


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

    worst_cases = _compute_worst_cases(baseline_results, candidate_results, comparison_results)
    gate_summary = gate_decision_obj.summary if gate_decision_obj else ""
    if gate_decision_obj and gate_decision_obj.decision == "NO-GO" and worst_cases:
        gate_summary += f"\n\nWorst performing cases (candidate lost):\n{worst_cases}"

    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "gate_decision": gate_decision_obj.decision if gate_decision_obj else "UNKNOWN",
        "gate_summary": gate_summary,
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

    # ── Pre-compute worst cases so they can appear in the gate panel ───────────
    worst_cases_text = _compute_worst_cases(baseline_results, candidate_results, comparison)

    # ── Header ─────────────────────────────────────────────────────────────────
    decision_color = {"GO": "green", "NO-GO": "red", "INCONCLUSIVE": "yellow"}.get(
        gate.decision, "white"
    )
    panel_body = f"[bold {decision_color}]  {gate.decision}[/]\n\n{gate.summary}"
    if gate.decision == "NO-GO" and worst_cases_text:
        panel_body += f"\n\nWorst performing cases (candidate lost):\n{worst_cases_text}"
    console.print(
        Panel(
            panel_body,
            title=f"[bold]The Guard — Eval Run {run_id}[/]",
            border_style=decision_color,
        )
    )

    # ── API Error Summary (shown prominently if any calls failed) ──────────────
    all_errors = []
    for role, res_dict in [("BASELINE", baseline_results), ("CANDIDATE", candidate_results)]:
        for task_name, res_list in res_dict.items():
            for r in res_list:
                for err in getattr(r, "errors", []):
                    case_id = getattr(r, "case_id", "?")
                    model   = getattr(r, "model", "?")
                    # Classify the error type
                    if "billing" in err.lower() or "not active" in err.lower():
                        kind = "BILLING NOT ACTIVE"
                        color = "red"
                    elif "404" in err or "not found" in err.lower():
                        kind = "MODEL NOT FOUND"
                        color = "yellow"
                    elif "429" in err or "quota" in err.lower() or "rate" in err.lower():
                        kind = "RATE LIMIT / QUOTA"
                        color = "red"
                    elif "auth" in err.lower() or "api key" in err.lower():
                        kind = "AUTH / API KEY"
                        color = "red"
                    else:
                        kind = "API ERROR"
                        color = "yellow"
                    all_errors.append((role, task_name, case_id, model, kind, color, err[:160]))

    if all_errors:
        console.print(
            Panel(
                f"[bold red]{len(all_errors)} API call(s) failed — scores for these cases are ZERO (not real scores)[/]\n"
                f"[dim]This means the model never ran — results below are misleading for affected cases.[/]",
                title="[bold red]⚠  API FAILURES DETECTED[/]",
                border_style="red",
            )
        )
        err_table = Table(box=box.SIMPLE, show_header=True)
        err_table.add_column("Role",    width=10)
        err_table.add_column("Task",    width=14)
        err_table.add_column("Case",    width=10)
        err_table.add_column("Model",   width=26)
        err_table.add_column("Type",    width=22)
        err_table.add_column("Detail",  width=60)
        for role, task, case_id, model, kind, color, detail in all_errors:
            err_table.add_row(
                f"[{color}]{role}[/]",
                task, case_id, model,
                f"[{color}]{kind}[/]",
                f"[dim]{detail}[/]",
            )
        console.print(err_table)
        console.print()

    # ── Model routing summary ───────────────────────────────────────────────────
    console.print("\n[bold cyan]── Model Routing: Who Did What ──[/]")
    routing_table = Table(box=box.SIMPLE, show_header=True)
    routing_table.add_column("Task", style="cyan", width=20)
    routing_table.add_column("Role", width=12)
    routing_table.add_column("Provider", width=12)
    routing_table.add_column("Model", width=28)
    routing_table.add_column("Cases Run", justify="right", width=10)

    task_display = {
        "deal_copy": "Deal Copy",
        "insurance": "Insurance",
        "credit": "Credit Narrative",
    }
    for task_key, res_key in [("deal_copy", "deal_copy"), ("insurance", "insurance"), ("credit", "credit")]:
        b_list = baseline_results.get(res_key, [])
        c_list = candidate_results.get(res_key, [])
        b_model = b_list[0].model if b_list and hasattr(b_list[0], "model") else "unknown"
        c_model = c_list[0].model if c_list and hasattr(c_list[0], "model") else "unknown"
        b_prov = b_list[0].provider if b_list and hasattr(b_list[0], "provider") else metadata.get("baseline_provider", "?")
        c_prov = c_list[0].provider if c_list and hasattr(c_list[0], "provider") else metadata.get("candidate_provider", "?")
        routing_table.add_row(task_display.get(task_key, task_key), "Baseline", b_prov, b_model, str(len(b_list)))
        routing_table.add_row("", "Candidate", c_prov, c_model, str(len(c_list)))
        routing_table.add_row("", "", "", "", "")
    console.print(routing_table)

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

    # ── Detailed per-case walkthrough (always shown) ───────────────────────────
    console.print("\n[bold cyan]── Detailed Per-Case Breakdown ──[/]")
    console.print("[dim]Shows exactly what each model generated and how each score was calculated.[/]\n")

    task_configs = [
        ("Deal Copy",       "deal_copy",  "deal_copy",  "composite_score",
         ["format_score", "grounding_score", "judge_score", "similarity_score"]),
        ("Insurance",       "insurance",  "insurance",  "intent_score",
         ["intent_score"]),
        ("Credit Narrative","credit",     "credit",     "composite_score",
         ["grounding_score", "judge_score", "similarity_score"]),
    ]

    for display_name, task_key, res_key, primary_score, score_fields in task_configs:
        b_list = baseline_results.get(res_key, [])
        c_list = candidate_results.get(res_key, [])
        if not b_list:
            continue

        console.print(f"[bold yellow]{'='*70}[/]")
        console.print(f"[bold yellow]TASK: {display_name}[/]")
        console.print(f"[bold yellow]{'='*70}[/]")

        for i, (b, c) in enumerate(zip(b_list, c_list)):
            case_id = getattr(b, "case_id", f"case_{i+1}")
            console.print(f"\n[bold]  Case {i+1}/{len(b_list)}: {case_id}[/]")

            # Input description
            if res_key == "deal_copy":
                console.print(f"  [dim]Input:[/] {getattr(b,'merchant','?')} — channel: {getattr(b,'channel','?')}")
                console.print(f"  [dim]Source data:[/] {getattr(b,'source_data',{})}")
            elif res_key == "insurance":
                console.print(f"  [dim]Input:[/] deal_object → {getattr(b,'deal_object',{})}")
                console.print(f"  [dim]Correct answer:[/] [green]{getattr(b,'ground_truth','?')}[/]")
            elif res_key == "credit":
                console.print(f"  [dim]Input:[/] user={getattr(b,'user_id','?')}, eligible={getattr(b,'ground_truth_eligible','?')}")

            # Baseline output
            b_text = getattr(b, "generated_text", None) or getattr(b, "generated_narrative", None) or getattr(b, "raw_output", "")
            b_errors = getattr(b, "errors", [])
            console.print(f"\n  [green]BASELINE ({getattr(b,'model','?')}):[/]")
            if b_errors:
                console.print(f"    [red]Error: {b_errors[0][:120]}[/]")
            else:
                console.print(f"    Output: {b_text.strip() if b_text else '(empty)'}")

            # Baseline scores
            score_parts = []
            for sf in score_fields:
                val = getattr(b, sf, None)
                if val is not None:
                    score_parts.append(f"{sf}={val:.3f}")
            if res_key == "insurance":
                score_parts.append(f"predicted={getattr(b,'predicted_label','?')}")
                score_parts.append(f"correct={getattr(b,'correct','?')}")
            b_primary = getattr(b, primary_score, 0)
            console.print(f"    Scores: {' | '.join(score_parts)}")
            console.print(f"    [green]Final score: {b_primary:.3f}[/]")

            # Candidate output
            c_text = getattr(c, "generated_text", None) or getattr(c, "generated_narrative", None) or getattr(c, "raw_output", "")
            c_errors = getattr(c, "errors", [])
            console.print(f"\n  [yellow]CANDIDATE ({getattr(c,'model','?')}):[/]")
            if c_errors:
                console.print(f"    [red]Error: {c_errors[0][:120]}[/]")
            else:
                console.print(f"    Output: {c_text.strip() if c_text else '(empty)'}")

            # Candidate scores
            score_parts = []
            for sf in score_fields:
                val = getattr(c, sf, None)
                if val is not None:
                    score_parts.append(f"{sf}={val:.3f}")
            if res_key == "insurance":
                score_parts.append(f"predicted={getattr(c,'predicted_label','?')}")
                score_parts.append(f"correct={getattr(c,'correct','?')}")
            c_primary = getattr(c, primary_score, 0)
            console.print(f"    Scores: {' | '.join(score_parts)}")
            console.print(f"    [yellow]Final score: {c_primary:.3f}[/]")

            # Winner for this case
            diff = c_primary - b_primary
            if diff < -0.02:
                console.print(f"  [red]  → Baseline won this case by {abs(diff):.3f}[/]")
            elif diff > 0.02:
                console.print(f"  [green]  → Candidate won this case by {diff:.3f}[/]")
            else:
                console.print(f"  [dim]  → Tied (diff={diff:+.3f})[/]")

        console.print()

    # ── Per-case breakdown (shown when any task REGRESSED) ─────────────────────
    regressed_tasks = {
        task: data
        for task, comp_data in comparison.items()
        for scorer, data in comp_data.items()
        if data.get("verdict") == "REGRESSED"
    }

    if regressed_tasks or gate.decision == "NO-GO":
        console.print("\n[bold red]── Regression Detail: Which Cases Failed ──[/]")

        task_map = {
            "deal_copy": ("composite_score", "deal_copy"),
            "insurance": ("intent_score", "insurance"),
            "credit":    ("composite_score", "credit"),
        }

        for task_name, (score_field, res_key) in task_map.items():
            b_list = baseline_results.get(res_key, [])
            c_list = candidate_results.get(res_key, [])
            if not b_list or not c_list:
                continue

            # Check if this task has any REGRESSED scorer
            task_comp = comparison.get(task_name, {})
            task_has_regression = any(
                v.get("verdict") == "REGRESSED" for v in task_comp.values()
            )
            if not task_has_regression:
                continue

            console.print(f"\n[bold cyan]{task_name.upper()} — case-by-case comparison:[/]")

            case_table = Table(box=box.SIMPLE, show_header=True)
            case_table.add_column("Case ID", style="dim", width=14)
            case_table.add_column("Description", width=28)
            case_table.add_column("Baseline", justify="right", width=10)
            case_table.add_column("Candidate", justify="right", width=10)
            case_table.add_column("Δ", justify="right", width=8)
            case_table.add_column("Winner", width=12)

            worse_cases = []
            better_cases = []
            same_cases = []

            for b, c in zip(b_list, c_list):
                b_score = getattr(b, score_field, None)
                c_score = getattr(c, score_field, None)
                if b_score is None or c_score is None:
                    continue

                case_id = getattr(b, "case_id", "?")

                # Build a human-readable description per task
                if task_name == "deal_copy":
                    desc = f"{getattr(b, 'merchant', '?')} ({getattr(b, 'channel', '?')})"
                elif task_name == "insurance":
                    truth = getattr(b, "ground_truth", "?")
                    b_pred = getattr(b, "predicted_label", "?")
                    c_pred = getattr(c, "predicted_label", "?")
                    desc = f"truth={truth}"
                elif task_name == "credit":
                    uid = getattr(b, "user_id", "?")
                    eligible = getattr(b, "ground_truth_eligible", "?")
                    desc = f"user={uid} eligible={eligible}"
                else:
                    desc = ""

                diff = c_score - b_score
                if diff < -0.02:
                    color = "red"
                    winner = "[red]BASELINE[/]"
                    worse_cases.append(case_id)
                elif diff > 0.02:
                    color = "green"
                    winner = "[green]CANDIDATE[/]"
                    better_cases.append(case_id)
                else:
                    color = "white"
                    winner = "[dim]TIED[/]"
                    same_cases.append(case_id)

                # For insurance, show label prediction details too
                if task_name == "insurance":
                    b_pred = getattr(b, "predicted_label", "?")
                    c_pred = getattr(c, "predicted_label", "?")
                    desc = f"{desc} | B={b_pred} C={c_pred}"

                case_table.add_row(
                    case_id,
                    desc,
                    f"{b_score:.3f}",
                    f"[{color}]{c_score:.3f}[/]",
                    f"[{color}]{diff:+.3f}[/]",
                    winner,
                )

            console.print(case_table)
            console.print(
                f"  [red]Candidate worse on {len(worse_cases)} case(s):[/] {', '.join(worse_cases) or 'none'}"
            )
            console.print(
                f"  [green]Candidate better on {len(better_cases)} case(s):[/] {', '.join(better_cases) or 'none'}"
            )
            console.print(
                f"  [dim]Tied on {len(same_cases)} case(s)[/]"
            )

        console.print()

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
