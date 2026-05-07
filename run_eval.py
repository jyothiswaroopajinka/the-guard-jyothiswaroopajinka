"""
The Guard — Main Eval Runner

Usage:
  python run_eval.py                         # Full eval (baseline=v1 vs candidate=v2, Claude vs GPT)
  python run_eval.py --quick                 # 5 cases per task, skip judge (fast demo)
  python run_eval.py --history               # Show past eval runs
  python run_eval.py --diff deal_copy v1 v2  # Diff two prompt versions

What it does:
  1. Registers all prompt versions (for versioning/audit)
  2. Runs baseline eval (Claude Sonnet + prompt v1)
  3. Runs candidate eval (OpenAI GPT-4o-mini + prompt v1, OR same model + prompt v2)
  4. Computes statistical comparison (paired bootstrap + McNemar's)
  5. Decides GO / NO-GO / INCONCLUSIVE
  6. Saves results + prints report
  7. Exits with code 0 (GO), 1 (NO-GO), 2 (INCONCLUSIVE)

The exit code is what the GitHub Action checks.
"""

import argparse
import json
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

from rich.console import Console

from src import (
    dashboard,
    gate as gate_module,
    statistical,
    versioning,
)
from src.gate import TaskVerdicts
from src.tasks import deal_copy_eval, insurance_eval, credit_eval

console = Console()


def run_full_eval(
    baseline_provider: str,
    baseline_model: str | None,
    baseline_prompt: str,
    candidate_provider: str,
    candidate_model: str | None,
    candidate_prompt: str,
    max_cases: int | None = None,
    use_judge: bool = True,
) -> int:
    """
    Core eval loop. Returns exit code (0=GO, 1=NO-GO, 2=INCONCLUSIVE).
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    console.print(f"\n[bold cyan]The Guard — Eval Run {run_id}[/]")
    console.print(f"Baseline: [green]{baseline_provider}/{baseline_model or 'default'}[/] + prompt [green]{baseline_prompt}[/]")
    console.print(f"Candidate: [yellow]{candidate_provider}/{candidate_model or 'default'}[/] + prompt [yellow]{candidate_prompt}[/]\n")

    # ── Step 1: Register prompt versions ───────────────────────────────────────
    console.print("[dim]Registering prompt versions...[/]")
    versioning.register_all_prompts()

    # ── Step 2: Run baseline ───────────────────────────────────────────────────
    console.print("[bold]Running baseline eval...[/]")

    console.print("  [1/3] Deal copy task...")
    baseline_deal = deal_copy_eval.run_all(
        provider=baseline_provider,
        model=baseline_model,
        prompt_file=baseline_prompt,
        max_cases=max_cases,
        use_judge=use_judge,
    )

    console.print("  [2/3] Insurance classification task...")
    baseline_ins = insurance_eval.run_all(
        provider=baseline_provider,
        model=baseline_model,
        prompt_file="v1.txt",
        max_cases=max_cases,
    )

    console.print("  [3/3] Credit narrative task...")
    baseline_credit = credit_eval.run_all(
        provider=baseline_provider,
        model=baseline_model,
        prompt_file="v1.txt",
        max_cases=max_cases,
        use_judge=use_judge,
    )

    # ── Step 3: Run candidate ──────────────────────────────────────────────────
    console.print("\n[bold]Running candidate eval...[/]")

    console.print("  [1/3] Deal copy task...")
    candidate_deal = deal_copy_eval.run_all(
        provider=candidate_provider,
        model=candidate_model,
        prompt_file=candidate_prompt,
        max_cases=max_cases,
        use_judge=use_judge,
    )

    console.print("  [2/3] Insurance classification task...")
    candidate_ins = insurance_eval.run_all(
        provider=candidate_provider,
        model=candidate_model,
        prompt_file="v1.txt",
        max_cases=max_cases,
    )

    console.print("  [3/3] Credit narrative task...")
    candidate_credit = credit_eval.run_all(
        provider=candidate_provider,
        model=candidate_model,
        prompt_file="v1.txt",
        max_cases=max_cases,
        use_judge=use_judge,
    )

    # ── Step 4: Statistical comparison ────────────────────────────────────────
    console.print("\n[bold]Computing statistical comparisons...[/]")

    comparison = {}
    all_verdicts: list[TaskVerdicts] = []

    # Deal copy: compare composite scores
    deal_comp = statistical.paired_bootstrap_test(
        [r.composite_score for r in baseline_deal],
        [r.composite_score for r in candidate_deal],
    )
    comparison["deal_copy"] = {"composite": asdict(deal_comp)}
    all_verdicts.append(TaskVerdicts(
        task_name="deal_copy",
        scorer_name="composite",
        result=deal_comp,
        baseline_version=f"{baseline_provider}/{baseline_prompt}",
        candidate_version=f"{candidate_provider}/{candidate_prompt}",
    ))

    # Insurance: McNemar's (binary correct/incorrect) + continuous intent score
    ins_mcnemar = statistical.mcnemar_test(
        [r.correct for r in baseline_ins],
        [r.correct for r in candidate_ins],
    )
    ins_comp = statistical.paired_bootstrap_test(
        [r.intent_score for r in baseline_ins],
        [r.intent_score for r in candidate_ins],
    )
    comparison["insurance"] = {
        "intent_score": asdict(ins_comp),
        "mcnemar": asdict(ins_mcnemar),
    }
    all_verdicts.append(TaskVerdicts(
        task_name="insurance",
        scorer_name="intent_score",
        result=ins_comp,
        baseline_version=f"{baseline_provider}/v1.txt",
        candidate_version=f"{candidate_provider}/v1.txt",
    ))

    # Credit: compare grounding scores (most critical)
    credit_comp = statistical.paired_bootstrap_test(
        [r.composite_score for r in baseline_credit],
        [r.composite_score for r in candidate_credit],
    )
    comparison["credit"] = {"composite": asdict(credit_comp)}
    all_verdicts.append(TaskVerdicts(
        task_name="credit",
        scorer_name="composite",
        result=credit_comp,
        baseline_version=f"{baseline_provider}/v1.txt",
        candidate_version=f"{candidate_provider}/v1.txt",
    ))

    # ── Step 5: GO/NO-GO gate ─────────────────────────────────────────────────
    gate_decision = gate_module.evaluate(all_verdicts)

    # ── Step 6: Save + print ───────────────────────────────────────────────────
    baseline_results = {"deal_copy": baseline_deal, "insurance": baseline_ins, "credit": baseline_credit}
    candidate_results = {"deal_copy": candidate_deal, "insurance": candidate_ins, "credit": candidate_credit}
    metadata = {
        "baseline_provider": baseline_provider,
        "baseline_model": baseline_model,
        "baseline_prompt": baseline_prompt,
        "candidate_provider": candidate_provider,
        "candidate_model": candidate_model,
        "candidate_prompt": candidate_prompt,
    }

    run_file = dashboard.save_run(
        run_id=run_id,
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        comparison_results=comparison,
        gate_decision_obj=gate_decision,
        metadata=metadata,
    )

    dashboard.print_run_report(
        run_id=run_id,
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        comparison=comparison,
        gate=gate_decision,
        metadata=metadata,
    )

    console.print(f"\n[dim]Results saved to: {run_file}[/]")

    # If regression detected, also print the prompt diff
    if gate_decision.decision == "NO-GO" and baseline_prompt != candidate_prompt:
        console.print("\n[bold red]Prompt diff (what changed):[/]")
        diff = versioning.diff_prompts("deal_copy", baseline_prompt, candidate_prompt)
        console.print(diff)

    return gate_decision.exit_code


def main():
    parser = argparse.ArgumentParser(description="The Guard — AI Eval Pipeline")
    parser.add_argument("--quick", action="store_true", help="Run 5 cases per task, skip judge")
    parser.add_argument("--history", action="store_true", help="Show eval history")
    parser.add_argument("--diff", nargs=3, metavar=("TASK", "FILE_A", "FILE_B"), help="Diff two prompts")

    # Baseline args
    parser.add_argument("--baseline-provider", default="anthropic", choices=["anthropic", "openai", "google"])
    parser.add_argument("--baseline-model", default=None, help="e.g. claude-sonnet-4-6")
    parser.add_argument("--baseline-prompt", default="v1.txt", help="Prompt file in prompts/deal_copy/")

    # Candidate args
    parser.add_argument("--candidate-provider", default="openai", choices=["anthropic", "openai", "google"])
    parser.add_argument("--candidate-model", default=None, help="e.g. gpt-4o-mini")
    parser.add_argument("--candidate-prompt", default="v1.txt", help="Prompt file")

    # Regression demo: compare bad prompt (v2) vs good (v1) to show gate catching it
    parser.add_argument("--demo-regression", action="store_true",
                        help="Demo: compare v1 vs v2 (bad prompt) to show NO-GO gate")

    args = parser.parse_args()

    if args.history:
        dashboard.print_history_table()
        sys.exit(0)

    if args.diff:
        task, file_a, file_b = args.diff
        diff = versioning.diff_prompts(task, file_a, file_b)
        console.print(diff)
        sys.exit(0)

    # Demo regression: use same model (Sonnet) but swap to degraded prompt v2
    if args.demo_regression:
        console.print("[bold yellow]DEMO MODE: Comparing v1 (good) vs v2 (degraded) prompt[/]")
        exit_code = run_full_eval(
            baseline_provider="anthropic",
            baseline_model=None,
            baseline_prompt="v1.txt",
            candidate_provider="anthropic",
            candidate_model=None,
            candidate_prompt="v2.txt",
            max_cases=10,
            use_judge=True,
        )
    else:
        exit_code = run_full_eval(
            baseline_provider=args.baseline_provider,
            baseline_model=args.baseline_model,
            baseline_prompt=args.baseline_prompt,
            candidate_provider=args.candidate_provider,
            candidate_model=args.candidate_model,
            candidate_prompt=args.candidate_prompt,
            max_cases=5 if args.quick else None,
            use_judge=not args.quick,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
