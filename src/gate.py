from __future__ import annotations

"""
GO / NO-GO Gate

Takes the statistical comparison results across all eval tasks and makes a
final deployment decision: GO, NO-GO, or INCONCLUSIVE.

Rules (configurable via env):
  NO-GO  if ANY task shows statistically significant regression (p < alpha)
         AND absolute score drop > regression_threshold
  GO     if all tasks show NO_CHANGE or IMPROVED
  INCONCLUSIVE  if some tasks improved, some regressed, none statistically significant

The gate is what the GitHub Action checks. It exits with code 1 on NO-GO.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

from src.statistical import ComparisonResult, McNemarResult


Decision = Literal["GO", "NO-GO", "INCONCLUSIVE"]

SIGNIFICANCE_THRESHOLD = float(os.getenv("SIGNIFICANCE_THRESHOLD", "0.05"))
REGRESSION_THRESHOLD = float(os.getenv("REGRESSION_THRESHOLD", "0.05"))
# If the drop exceeds this, flag NO-GO even without statistical significance.
# An 18% quality drop is never acceptable regardless of p-value.
PRACTICAL_THRESHOLD = float(os.getenv("PRACTICAL_THRESHOLD", "0.10"))


@dataclass
class TaskVerdicts:
    task_name: str
    scorer_name: str
    result: ComparisonResult | McNemarResult
    baseline_version: str
    candidate_version: str


@dataclass
class GateDecision:
    decision: Decision
    regressions: list[TaskVerdicts] = field(default_factory=list)
    improvements: list[TaskVerdicts] = field(default_factory=list)
    inconclusive: list[TaskVerdicts] = field(default_factory=list)
    summary: str = ""
    failure_reasons: list[dict] = field(default_factory=list)  # structured per-task reasons
    exit_code: int = 0   # 0 = GO, 1 = NO-GO, 2 = INCONCLUSIVE


def _format_regression(tv: TaskVerdicts) -> str:
    r = tv.result
    confidence = (1 - r.p_value) * 100
    if isinstance(r, McNemarResult):
        b = r.baseline_correct
        c = r.candidate_correct
        n = b + r.discordant_b  # total cases baseline saw
        return (
            f"  - [{tv.task_name} / {tv.scorer_name}]: "
            f"baseline correct on {b}/{n} cases, candidate correct on {c}/{n} cases "
            f"(dropped {b - c} cases, confidence={confidence:.1f}%, p={r.p_value:.4f})\n"
            f"    Reason: candidate got wrong answers on cases that baseline passed"
        )
    else:
        b_mean = getattr(r, "baseline_mean", "?")
        c_mean = getattr(r, "candidate_mean", "?")
        diff = getattr(r, "absolute_diff", 0)
        pct = getattr(r, "relative_diff_pct", 0)
        ci_low = getattr(r, "ci_low", "?")
        ci_high = getattr(r, "ci_high", "?")
        return (
            f"  - [{tv.task_name} / {tv.scorer_name}]: "
            f"{b_mean:.4f} → {c_mean:.4f} "
            f"(drop of {abs(diff):.4f} = {abs(pct):.1f}%, confidence={confidence:.1f}%, p={r.p_value:.4f})\n"
            f"    95% CI of drop: [{ci_low:.4f}, {ci_high:.4f}]\n"
            f"    Reason: candidate scored lower than baseline on average across all cases"
        )


def evaluate(
    verdicts: list[TaskVerdicts],
    alpha: float = SIGNIFICANCE_THRESHOLD,
    regression_threshold: float = REGRESSION_THRESHOLD,
    practical_threshold: float = PRACTICAL_THRESHOLD,
) -> GateDecision:
    """
    Aggregate all task verdicts into a single deployment decision.

    Two ways to trigger NO-GO:
    1. Statistically significant regression: p < alpha AND drop > regression_threshold
    2. Practically significant regression: drop > practical_threshold (regardless of p-value)
       — catches large regressions that are real but noisy across cases

    A regression in ANY task type blocks deployment (conservative by default).
    """
    regressions = []
    improvements = []
    inconclusive_list = []

    for tv in verdicts:
        r = tv.result
        verdict = r.verdict

        significant = getattr(r, "significant", False)

        abs_diff = 0.0
        if isinstance(r, ComparisonResult):
            abs_diff = r.absolute_diff

        # Statistically confirmed regression
        stat_regression = (
            verdict == "REGRESSED"
            and significant
            and (isinstance(r, McNemarResult) or abs_diff < -regression_threshold)
        )
        # Practically significant regression — too large to ignore even without stats
        practical_regression = (
            isinstance(r, ComparisonResult)
            and abs_diff < -practical_threshold
        )

        if stat_regression or practical_regression:
            regressions.append(tv)
        elif verdict == "IMPROVED" and significant:
            improvements.append(tv)
        else:
            inconclusive_list.append(tv)

    if regressions:
        reg_details = "\n".join(
            _format_regression(tv) for tv in regressions
        )
        # Explain whether it was stats-driven, practically-driven, or both
        stat_count = sum(
            1 for tv in regressions
            if getattr(tv.result, "significant", False)
        )
        practical_count = len(regressions) - stat_count
        if stat_count and practical_count:
            trigger = f"{stat_count} statistically significant + {practical_count} practically significant (drop > {practical_threshold*100:.0f}%)"
        elif practical_count:
            trigger = f"drop exceeded {practical_threshold*100:.0f}% practical threshold (too large to ignore even without statistical significance)"
        else:
            trigger = "statistically significant regression detected"

        summary = (
            f"NO-GO: {len(regressions)} regression(s) detected — {trigger}.\n"
            f"Regressed tasks:\n{reg_details}\n"
            f"Candidate version blocked from deployment."
        )

        # Build structured per-task failure reasons for JSON storage
        failure_reasons = []
        for tv in regressions:
            r = tv.result
            is_practical = (
                isinstance(r, ComparisonResult) and r.absolute_diff < -practical_threshold
                and not getattr(r, "significant", False)
            )
            if isinstance(r, McNemarResult):
                reason = (
                    f"Candidate correctly classified {r.candidate_correct} cases, "
                    f"baseline correctly classified {r.baseline_correct} cases — "
                    f"candidate got wrong answers on {r.baseline_correct - r.candidate_correct} "
                    f"cases that baseline passed (p={r.p_value:.4f})"
                )
            else:
                drop_pct = abs(r.absolute_diff / r.baseline_mean * 100) if r.baseline_mean else 0
                reason = (
                    f"Score dropped from {r.baseline_mean:.4f} to {r.candidate_mean:.4f} "
                    f"(drop of {abs(r.absolute_diff):.4f} = {drop_pct:.1f}%). "
                    f"{'Triggered by practical threshold — drop too large to ignore.' if is_practical else f'Statistically significant at p={r.p_value:.4f}.'}"
                )
            failure_reasons.append({
                "task": tv.task_name,
                "scorer": tv.scorer_name,
                "baseline_version": tv.baseline_version,
                "candidate_version": tv.candidate_version,
                "reason": reason,
                "trigger": "practical_threshold" if is_practical else "statistical",
                "p_value": getattr(r, "p_value", None),
                "confidence_pct": round((1 - getattr(r, "p_value", 1.0)) * 100, 1),
                "score_drop": round(getattr(r, "absolute_diff", 0), 4),
            })

        return GateDecision(
            decision="NO-GO",
            regressions=regressions,
            improvements=improvements,
            inconclusive=inconclusive_list,
            summary=summary,
            failure_reasons=failure_reasons,
            exit_code=1,
        )

    # No regressions → GO. Statistically confirmed improvements are a bonus, not required.
    improvement_details = ""
    if improvements:
        improvement_details = "\n" + "\n".join(
            f"  - [{tv.task_name} / {tv.scorer_name}]: "
            f"{getattr(tv.result, 'baseline_mean', '?'):.3f} → {getattr(tv.result, 'candidate_mean', '?'):.3f}"
            f" (p={tv.result.p_value:.4f})"
            for tv in improvements
        )

    if inconclusive_list and not improvements:
        no_change_tasks = ", ".join(f"{tv.task_name}/{tv.scorer_name}" for tv in inconclusive_list)
        summary = (
            f"GO: No regressions detected across {len(verdicts)} task(s). "
            f"Candidate is safe to deploy.\n"
            f"Tasks with no significant change (within noise): {no_change_tasks}"
        )
    else:
        summary = (
            f"GO: All tasks passed. "
            f"{len(improvements)} improvement(s) statistically confirmed.{improvement_details}"
        )

    return GateDecision(
        decision="GO",
        regressions=[],
        improvements=improvements,
        inconclusive=inconclusive_list,
        summary=summary,
        exit_code=0,
    )
