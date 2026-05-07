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
    exit_code: int = 0   # 0 = GO, 1 = NO-GO, 2 = INCONCLUSIVE


def evaluate(
    verdicts: list[TaskVerdicts],
    alpha: float = SIGNIFICANCE_THRESHOLD,
    regression_threshold: float = REGRESSION_THRESHOLD,
) -> GateDecision:
    """
    Aggregate all task verdicts into a single deployment decision.

    A regression in ANY task type blocks deployment (conservative by default).
    This can be changed to require ALL tasks to regress for stricter review.
    """
    regressions = []
    improvements = []
    inconclusive_list = []

    for tv in verdicts:
        r = tv.result
        verdict = r.verdict

        # Get p_value from either type
        p_value = getattr(r, "p_value", 1.0)
        significant = getattr(r, "significant", False)

        # For ComparisonResult, also check absolute diff
        abs_diff = 0.0
        if isinstance(r, ComparisonResult):
            abs_diff = r.absolute_diff

        if verdict == "REGRESSED" and significant and (
            isinstance(r, McNemarResult) or abs_diff < -regression_threshold
        ):
            regressions.append(tv)
        elif verdict == "IMPROVED" and significant:
            improvements.append(tv)
        else:
            inconclusive_list.append(tv)

    if regressions:
        reg_details = "\n".join(
            f"  - [{tv.task_name} / {tv.scorer_name}]: "
            f"{tv.result.baseline_correct if hasattr(tv.result, 'baseline_correct') else getattr(tv.result, 'baseline_mean', '?')}"
            f" → {tv.result.candidate_correct if hasattr(tv.result, 'candidate_correct') else getattr(tv.result, 'candidate_mean', '?')}"
            f" (p={tv.result.p_value:.4f})"
            for tv in regressions
        )
        summary = (
            f"NO-GO: {len(regressions)} regression(s) detected.\n"
            f"Regressed tasks:\n{reg_details}\n"
            f"Candidate version blocked from deployment."
        )
        return GateDecision(
            decision="NO-GO",
            regressions=regressions,
            improvements=improvements,
            inconclusive=inconclusive_list,
            summary=summary,
            exit_code=1,
        )

    if not improvements and inconclusive_list:
        summary = (
            f"INCONCLUSIVE: No statistically significant changes detected in {len(inconclusive_list)} task(s). "
            f"Differences may be noise. Run more test cases or reduce temperature for a cleaner signal."
        )
        return GateDecision(
            decision="INCONCLUSIVE",
            regressions=[],
            improvements=improvements,
            inconclusive=inconclusive_list,
            summary=summary,
            exit_code=2,
        )

    improvement_details = "\n".join(
        f"  - [{tv.task_name} / {tv.scorer_name}]: "
        f"{getattr(tv.result, 'baseline_mean', '?'):.3f} → {getattr(tv.result, 'candidate_mean', '?'):.3f}"
        f" (p={tv.result.p_value:.4f})"
        for tv in improvements
    )
    summary = (
        f"GO: All tasks passed. "
        f"{len(improvements)} improvement(s) confirmed.\n"
        f"{improvement_details}"
    )
    return GateDecision(
        decision="GO",
        regressions=[],
        improvements=improvements,
        inconclusive=inconclusive_list,
        summary=summary,
        exit_code=0,
    )
