from __future__ import annotations

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
    candidate_provider: str,
    candidate_model: str | None,
    deal_copy_baseline_prompt: str = "v1.txt",
    deal_copy_candidate_prompt: str = "v1.txt",
    insurance_baseline_prompt: str = "v1.txt",
    insurance_candidate_prompt: str = "v1.txt",
    credit_baseline_prompt: str = "v1.txt",
    credit_candidate_prompt: str = "v1.txt",
    max_cases: int | None = None,
    use_judge: bool = True,
) -> int:
    """
    Core eval loop. Returns exit code (0=GO, 1=NO-GO).
    Each task has its own baseline and candidate prompt so any task's prompt
    can be tested independently without affecting the others.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    console.print(f"\n[bold cyan]The Guard — Eval Run {run_id}[/]")
    console.print(f"Baseline: [green]{baseline_provider}/{baseline_model or 'default'}[/]")
    console.print(f"Candidate: [yellow]{candidate_provider}/{candidate_model or 'default'}[/]")
    console.print(f"Prompts (baseline → candidate): deal_copy=[green]{deal_copy_baseline_prompt}[/]→[yellow]{deal_copy_candidate_prompt}[/]  insurance=[green]{insurance_baseline_prompt}[/]→[yellow]{insurance_candidate_prompt}[/]  credit=[green]{credit_baseline_prompt}[/]→[yellow]{credit_candidate_prompt}[/]\n")

    # ── Step 1: Register prompt versions ───────────────────────────────────────
    console.print("[dim]Registering prompt versions...[/]")
    versioning.register_all_prompts()

    # ── Step 2: Run baseline ───────────────────────────────────────────────────
    console.print("[bold]Running baseline eval...[/]")

    console.print("  [1/3] Deal copy task...")
    baseline_deal = deal_copy_eval.run_all(
        provider=baseline_provider,
        model=baseline_model,
        prompt_file=deal_copy_baseline_prompt,
        max_cases=max_cases,
        use_judge=use_judge,
    )

    console.print("  [2/3] Insurance classification task...")
    baseline_ins = insurance_eval.run_all(
        provider=baseline_provider,
        model=baseline_model,
        prompt_file=insurance_baseline_prompt,
        max_cases=max_cases,
    )

    console.print("  [3/3] Credit narrative task...")
    baseline_credit = credit_eval.run_all(
        provider=baseline_provider,
        model=baseline_model,
        prompt_file=credit_baseline_prompt,
        max_cases=max_cases,
        use_judge=use_judge,
    )

    # ── Step 3: Run candidate ──────────────────────────────────────────────────
    console.print("\n[bold]Running candidate eval...[/]")

    console.print("  [1/3] Deal copy task...")
    candidate_deal = deal_copy_eval.run_all(
        provider=candidate_provider,
        model=candidate_model,
        prompt_file=deal_copy_candidate_prompt,
        max_cases=max_cases,
        use_judge=use_judge,
    )

    console.print("  [2/3] Insurance classification task...")
    candidate_ins = insurance_eval.run_all(
        provider=candidate_provider,
        model=candidate_model,
        prompt_file=insurance_candidate_prompt,
        max_cases=max_cases,
    )

    console.print("  [3/3] Credit narrative task...")
    candidate_credit = credit_eval.run_all(
        provider=candidate_provider,
        model=candidate_model,
        prompt_file=credit_candidate_prompt,
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
        baseline_version=f"{baseline_provider}/{deal_copy_baseline_prompt}",
        candidate_version=f"{candidate_provider}/{deal_copy_candidate_prompt}",
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
        baseline_version=f"{baseline_provider}/{insurance_baseline_prompt}",
        candidate_version=f"{candidate_provider}/{insurance_candidate_prompt}",
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
        baseline_version=f"{baseline_provider}/{credit_baseline_prompt}",
        candidate_version=f"{candidate_provider}/{credit_candidate_prompt}",
    ))

    # ── Step 5: GO/NO-GO gate ─────────────────────────────────────────────────
    gate_decision = gate_module.evaluate(all_verdicts)

    # ── Step 6: Save + print ───────────────────────────────────────────────────
    baseline_results = {"deal_copy": baseline_deal, "insurance": baseline_ins, "credit": baseline_credit}
    candidate_results = {"deal_copy": candidate_deal, "insurance": candidate_ins, "credit": candidate_credit}
    metadata = {
        "baseline_provider": baseline_provider,
        "baseline_model": baseline_model,
        "candidate_provider": candidate_provider,
        "candidate_model": candidate_model,
        "deal_copy_baseline_prompt": deal_copy_baseline_prompt,
        "deal_copy_candidate_prompt": deal_copy_candidate_prompt,
        "insurance_baseline_prompt": insurance_baseline_prompt,
        "insurance_candidate_prompt": insurance_candidate_prompt,
        "credit_baseline_prompt": credit_baseline_prompt,
        "credit_candidate_prompt": credit_candidate_prompt,
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

    # If regression detected, show diffs for any tasks whose prompts changed
    if gate_decision.decision == "NO-GO":
        prompt_pairs = [
            ("deal_copy", deal_copy_baseline_prompt, deal_copy_candidate_prompt),
            ("insurance", insurance_baseline_prompt, insurance_candidate_prompt),
            ("credit", credit_baseline_prompt, credit_candidate_prompt),
        ]
        changed = [(task, b, c) for task, b, c in prompt_pairs if b != c]
        if changed:
            console.print("\n[bold red]Prompt diff (what changed):[/]")
            for task, b_prompt, c_prompt in changed:
                console.print(f"\n[bold]{task}[/]: {b_prompt} → {c_prompt}")
                diff = versioning.diff_prompts(task, b_prompt, c_prompt)
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

    # Candidate args
    parser.add_argument("--candidate-provider", default="google", choices=["anthropic", "openai", "google"])
    parser.add_argument("--candidate-model", default=None, help="e.g. gpt-4o-mini")

    # Per-task prompt flags (baseline and candidate independently)
    parser.add_argument("--deal-copy-prompt", default="v1.txt", help="Candidate prompt for deal_copy task")
    parser.add_argument("--insurance-prompt", default="v1.txt", help="Candidate prompt for insurance task")
    parser.add_argument("--credit-prompt", default="v1.txt", help="Candidate prompt for credit task")

    # Regression demo: compare bad prompt (v2) vs good (v1) to show gate catching it
    parser.add_argument("--demo-regression", action="store_true",
                        help="Demo: compare v1 vs v2 (bad prompt) to show NO-GO gate")
    # Fallback demo: break the primary provider mid-eval to show automatic recovery
    parser.add_argument("--demo-fallback", action="store_true",
                        help="Demo: simulate primary provider failure to show fallback chain live")
    parser.add_argument("--max-cases", type=int, default=30,
                        help="Max cases per task (default: 30)")

    args = parser.parse_args()

    if args.history:
        dashboard.print_history_table()
        sys.exit(0)

    if args.diff:
        task, file_a, file_b = args.diff
        diff = versioning.diff_prompts(task, file_a, file_b)
        console.print(diff)
        sys.exit(0)

    # Fallback demo: patch Anthropic to fail so Google takes over during the real eval
    if args.demo_fallback:
        import src.providers as _providers
        _real_claude = _providers.call_claude

        def _broken_claude(*a, **kw):
            raise Exception("auth: invalid_api_key — simulated failure for fallback demo")

        _providers.call_claude = _broken_claude
        console.print("[bold yellow]DEMO MODE: Anthropic is broken — watch Google Gemini take over[/]\n")
        try:
            exit_code = run_full_eval(
                baseline_provider="anthropic",
                baseline_model=None,
                candidate_provider="anthropic",
                candidate_model=None,
                max_cases=3,
                use_judge=False,
            )
        finally:
            _providers.call_claude = _real_claude  # always restore
        sys.exit(exit_code)

    # Demo regression: same model, swap all tasks to degraded v2 prompts
    if args.demo_regression:
        console.print("[bold yellow]DEMO MODE: Comparing v1 (good) vs v2 (degraded) prompts across all tasks[/]")
        exit_code = run_full_eval(
            baseline_provider="anthropic",
            baseline_model=None,
            candidate_provider="anthropic",
            candidate_model=None,
            deal_copy_baseline_prompt="v1.txt",
            deal_copy_candidate_prompt="v2.txt",
            insurance_baseline_prompt="v1.txt",
            insurance_candidate_prompt="v2.txt",
            credit_baseline_prompt="v1.txt",
            credit_candidate_prompt="v1.txt",
            max_cases=args.max_cases,
            use_judge=True,
        )
    else:
        exit_code = run_full_eval(
            baseline_provider=args.baseline_provider,
            baseline_model=args.baseline_model,
            candidate_provider=args.candidate_provider,
            candidate_model=args.candidate_model,
            deal_copy_baseline_prompt="v1.txt",
            deal_copy_candidate_prompt=args.deal_copy_prompt,
            insurance_baseline_prompt="v1.txt",
            insurance_candidate_prompt=args.insurance_prompt,
            credit_baseline_prompt="v1.txt",
            credit_candidate_prompt=args.credit_prompt,
            max_cases=5 if args.quick else args.max_cases,
            use_judge=not args.quick,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
