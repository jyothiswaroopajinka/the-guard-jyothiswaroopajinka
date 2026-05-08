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
        "failure_reasons": getattr(gate_decision_obj, "failure_reasons", []),
        "baseline_results": {k: _serialize_list(v) for k, v in baseline_results.items()},
        "candidate_results": {k: _serialize_list(v) for k, v in candidate_results.items()},
        "comparison_results": comparison_results,
    }

    run_file = RESULTS_DIR / f"run_{run_id}.json"
    run_file.write_text(json.dumps(payload, indent=2, default=str))

    # Save to history so --history flag works
    def _comp_mean(task, scorer):
        c = comparison_results.get(task, {}).get(scorer, {})
        return c.get("baseline_mean", 0.0), c.get("candidate_mean", 0.0), c.get("absolute_diff", 0.0)

    b_deal, c_deal, d_deal = _comp_mean("deal_copy", "composite")
    b_ins,  c_ins,  d_ins  = _comp_mean("insurance",  "intent_score")
    b_cred, c_cred, d_cred = _comp_mean("credit",      "composite")

    all_costs = [
        getattr(r, "cost_usd", 0)
        for res_dict in [baseline_results, candidate_results]
        for res_list in res_dict.values()
        for r in res_list
    ]
    total_cases = sum(len(v) for v in baseline_results.values())

    summary = RunSummary(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        baseline_provider=metadata.get("baseline_provider", ""),
        baseline_model=metadata.get("baseline_model") or "default",
        baseline_prompt=metadata.get("baseline_prompt", ""),
        candidate_provider=metadata.get("candidate_provider", ""),
        candidate_model=metadata.get("candidate_model") or "default",
        candidate_prompt=metadata.get("candidate_prompt", ""),
        gate_decision=gate_decision_obj.decision if gate_decision_obj else "UNKNOWN",
        deal_copy_baseline_mean=b_deal,
        deal_copy_candidate_mean=c_deal,
        deal_copy_diff=d_deal,
        insurance_baseline_mean=b_ins,
        insurance_candidate_mean=c_ins,
        insurance_diff=d_ins,
        credit_baseline_mean=b_cred,
        credit_candidate_mean=c_cred,
        credit_diff=d_cred,
        total_cost_usd=round(sum(all_costs), 6),
        total_cases=total_cases,
    )
    update_history(summary)

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


def _save_regression_detail(
    run_id: str,
    baseline_results: dict,
    candidate_results: dict,
    comparison: dict,
) -> None:
    """
    For every regressed task, build a per-case JSON with:
    - case_id, scores, score drop, and a human-readable reason for the failure.
    Saved as eval_results/regression_<run_id>.json
    """
    task_map = {
        "deal_copy": ("composite_score", "deal_copy"),
        "insurance": ("intent_score",    "insurance"),
        "credit":    ("composite_score", "credit"),
    }

    regression_detail = {}

    for task_name, (score_field, res_key) in task_map.items():
        task_comp = comparison.get(task_name, {})
        if not any(v.get("verdict") == "REGRESSED" for v in task_comp.values()):
            continue

        b_list = baseline_results.get(res_key, [])
        c_list = candidate_results.get(res_key, [])
        cases = []

        for b, c in zip(b_list, c_list):
            b_score = getattr(b, score_field, None)
            c_score = getattr(c, score_field, None)
            if b_score is None or c_score is None:
                continue

            diff = c_score - b_score
            case_id = getattr(b, "case_id", "?")

            if diff < -0.02:
                outcome = "CANDIDATE_WORSE"
            elif diff > 0.02:
                outcome = "CANDIDATE_BETTER"
            else:
                outcome = "TIED"

            # Build a human-readable reason explaining WHY the candidate lost
            reason = ""
            if outcome == "CANDIDATE_WORSE":
                if task_name == "deal_copy":
                    b_fmt = getattr(b, "format_score", None)
                    c_fmt = getattr(c, "format_score", None)
                    b_grnd = getattr(b, "grounding_score", None)
                    c_grnd = getattr(c, "grounding_score", None)
                    b_jdg = getattr(b, "judge_score", None)
                    c_jdg = getattr(c, "judge_score", None)
                    parts = []
                    if c_fmt is not None and b_fmt is not None and c_fmt < b_fmt - 0.1:
                        parts.append(f"format compliance dropped ({b_fmt:.2f} → {c_fmt:.2f}) — likely exceeded channel character limit")
                    if c_grnd is not None and b_grnd is not None and c_grnd < b_grnd - 0.1:
                        parts.append(f"factual grounding dropped ({b_grnd:.2f} → {c_grnd:.2f}) — candidate missed or hallucinated deal details")
                    if c_jdg is not None and b_jdg is not None and c_jdg < b_jdg - 0.1:
                        parts.append(f"judge score dropped ({b_jdg:.2f} → {c_jdg:.2f}) — candidate copy was less persuasive or channel-appropriate")
                    reason = "; ".join(parts) if parts else "composite score dropped — multiple scorers contributed"

                elif task_name == "insurance":
                    b_pred = getattr(b, "predicted_label", "?")
                    c_pred = getattr(c, "predicted_label", "?")
                    truth  = getattr(b, "ground_truth", "?")
                    reason = (
                        f"baseline correctly predicted '{truth}', "
                        f"candidate predicted '{c_pred}' (wrong)"
                    )

                elif task_name == "credit":
                    b_grnd = getattr(b, "grounding_score", None)
                    c_grnd = getattr(c, "grounding_score", None)
                    b_jdg  = getattr(b, "judge_score", None)
                    c_jdg  = getattr(c, "judge_score", None)
                    parts = []
                    if c_grnd is not None and b_grnd is not None and c_grnd < b_grnd - 0.1:
                        parts.append(f"grounding dropped ({b_grnd:.2f} → {c_grnd:.2f}) — candidate narrative cited stats not found in user data")
                    if c_jdg is not None and b_jdg is not None and c_jdg < b_jdg - 0.1:
                        parts.append(f"judge score dropped ({b_jdg:.2f} → {c_jdg:.2f}) — narrative was less professional or complete")
                    reason = "; ".join(parts) if parts else "composite score dropped"

            cases.append({
                "case_id": case_id,
                "outcome": outcome,
                "baseline_score": round(b_score, 4),
                "candidate_score": round(c_score, 4),
                "score_drop": round(diff, 4),
                "reason": reason,
            })

        regression_detail[task_name] = {
            "score_field": score_field,
            "total_cases": len(cases),
            "candidate_worse": sum(1 for c in cases if c["outcome"] == "CANDIDATE_WORSE"),
            "candidate_better": sum(1 for c in cases if c["outcome"] == "CANDIDATE_BETTER"),
            "tied": sum(1 for c in cases if c["outcome"] == "TIED"),
            "cases": cases,
        }

    if regression_detail:
        RESULTS_DIR.mkdir(exist_ok=True)
        reg_file = RESULTS_DIR / f"regression_{run_id}.json"
        reg_file.write_text(json.dumps(regression_detail, indent=2))


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
    if gate.decision == "NO-GO":
        failed = [tv.task_name for tv in gate.regressions]
        passed = [t for t in ["deal_copy", "insurance", "credit"] if t not in failed]
        failed_str = ", ".join(failed)
        passed_str = ", ".join(passed) if passed else "none"
        panel_body += (
            f"\n\n[bold red]{'─' * 50}[/]"
            f"\n[bold red]DEPLOYMENT BLOCKED[/]"
            f"\n  Failed task(s)  : {failed_str}"
            f"\n  Passed task(s)  : {passed_str}"
            f"\n  Even though {passed_str} passed, the entire deployment"
            f"\n  is blocked because {failed_str} failed."
            f"\n  You cannot deploy a partial fix — it's all or nothing."
        )
    console.print(
        Panel(
            panel_body,
            title=f"[bold]The Guard — Eval Run {run_id}[/]",
            border_style=decision_color,
        )
    )

    # ── Per-task deployment readiness (always shown) ───────────────────────────
    console.print()
    failure_reasons = getattr(gate, "failure_reasons", [])
    failed_tasks = {fr["task"] for fr in failure_reasons}
    regressed_tasks = {tv.task_name for tv in getattr(gate, "regressions", [])}
    improved_tasks  = {tv.task_name for tv in getattr(gate, "improvements", [])}

    task_order = ["deal_copy", "insurance", "credit"]
    task_labels = {"deal_copy": "Deal Copy", "insurance": "Insurance", "credit": "Credit Narrative"}

    console.print("[bold cyan]── Deployment Readiness — Per Task ──[/]")
    for task_name in task_order:
        task_comp = comparison.get(task_name, {})
        if not task_comp:
            continue

        if task_name in regressed_tasks:
            status = "[bold red]BLOCKED — DO NOT DEPLOY[/]"
            border = "red"
            fr = next((f for f in failure_reasons if f["task"] == task_name), None)
            if fr:
                trigger_label = (
                    "Practical threshold breached (drop too large)"
                    if fr["trigger"] == "practical_threshold"
                    else "Statistically significant regression"
                )
                detail = (
                    f"[bold]Why blocked:[/] {trigger_label}\n"
                    f"[bold]What happened:[/] {fr['reason']}\n"
                    f"[bold]Score drop:[/] [red]{fr['score_drop']:+.4f}[/]  "
                    f"[bold]Confidence:[/] {fr['confidence_pct']}%  "
                    f"[bold]p-value:[/] {fr['p_value']:.4f}"
                )
            else:
                detail = "Regression detected."
        elif task_name in improved_tasks:
            status = "[bold green]READY — IMPROVED[/]"
            border = "green"
            tv = next((t for t in gate.improvements if t.task_name == task_name), None)
            r = tv.result if tv else None
            b = getattr(r, "baseline_mean", "?")
            c = getattr(r, "candidate_mean", "?")
            detail = (
                f"[bold]What happened:[/] Candidate improved over baseline\n"
                f"[bold]Score:[/] {b:.4f} → [green]{c:.4f}[/]  "
                f"[bold]p-value:[/] {getattr(r, 'p_value', '?'):.4f}"
            ) if r else "Improvement detected."
        else:
            status = "[bold yellow]READY — NO SIGNIFICANT CHANGE[/]"
            border = "yellow"
            scorer_data = list(task_comp.values())[0]
            b = scorer_data.get("baseline_mean", "?")
            c = scorer_data.get("candidate_mean", "?")
            diff = scorer_data.get("absolute_diff", 0)
            p = scorer_data.get("p_value", "?")
            detail = (
                f"[bold]What happened:[/] Candidate performance is similar to baseline — no regression\n"
                f"[bold]Score:[/] {b:.4f} → {c:.4f}  "
                f"[bold]Change:[/] {diff:+.4f}  "
                f"[bold]p-value:[/] {p:.4f}"
            ) if isinstance(b, float) else "No significant change detected."

        console.print(Panel(
            f"{status}\n\n{detail}",
            title=f"[bold]{task_labels.get(task_name, task_name)}[/]",
            border_style=border,
        ))

    console.print()

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

    # ── Regression detail — save to JSON + print concise summary ─────────────
    _save_regression_detail(run_id, baseline_results, candidate_results, comparison)

    if gate.decision == "NO-GO":
        task_map = {
            "deal_copy": ("composite_score", "deal_copy"),
            "insurance": ("intent_score",    "insurance"),
            "credit":    ("composite_score", "credit"),
        }
        console.print("\n[bold red]── Regression Detail: Which Cases Failed ──[/]")
        for task_name, (score_field, res_key) in task_map.items():
            task_comp = comparison.get(task_name, {})
            if not any(v.get("verdict") == "REGRESSED" for v in task_comp.values()):
                continue
            b_list = baseline_results.get(res_key, [])
            c_list = candidate_results.get(res_key, [])
            worse, better, tied = [], [], []
            for b, c in zip(b_list, c_list):
                b_score = getattr(b, score_field, None)
                c_score = getattr(c, score_field, None)
                if b_score is None or c_score is None:
                    continue
                diff = c_score - b_score
                cid = getattr(b, "case_id", "?")
                if diff < -0.02:
                    worse.append(cid)
                elif diff > 0.02:
                    better.append(cid)
                else:
                    tied.append(cid)
            console.print(f"\n  [bold]{task_name.upper()}[/]")
            console.print(f"  [red]Failed cases ({len(worse)}):[/] {', '.join(worse) or 'none'}")
            console.print(f"  [green]Improved cases ({len(better)}):[/] {', '.join(better) or 'none'}")
            console.print(f"  [dim]Tied ({len(tied)})[/]")
        reg_file = RESULTS_DIR / f"regression_{run_id}.json"
        console.print(f"\n  [dim]Full regression detail saved to: {reg_file}[/]")
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
