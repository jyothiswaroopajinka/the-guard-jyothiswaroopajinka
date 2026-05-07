"""
Statistical Comparison Engine

Compares two sets of scores (e.g. Claude Sonnet v1 prompt vs v2 prompt)
using rigorous statistical tests.

Why this matters: "Average went up 2%" is not a conclusion. We need to know
if the difference is real (statistically significant) or just noise from the
random seed / temperature of the LLM.

Methods:
  - Paired bootstrap test: Resample score differences with replacement,
    compute p-value as fraction of bootstrap samples where diff ≤ 0.
  - McNemar's test: For binary pass/fail outcomes (correct/incorrect).
  - Cohen's d: Effect size — tells us if the difference is practically meaningful.
"""

import numpy as np
from dataclasses import dataclass
from scipy.stats import chi2


@dataclass
class ComparisonResult:
    baseline_mean: float
    candidate_mean: float
    absolute_diff: float       # candidate - baseline
    relative_diff_pct: float   # % change
    p_value: float             # from bootstrap test
    significant: bool          # p < threshold
    ci_low: float              # 95% CI on the difference
    ci_high: float
    cohens_d: float            # effect size
    n_samples: int
    verdict: str               # "IMPROVED", "REGRESSED", "NO_CHANGE"
    recommendation: str


@dataclass
class McNemarResult:
    p_value: float
    significant: bool
    baseline_correct: int
    candidate_correct: int
    discordant_b: int          # baseline wrong, candidate right
    discordant_c: int          # baseline right, candidate wrong
    verdict: str


def paired_bootstrap_test(
    baseline_scores: list[float],
    candidate_scores: list[float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    regression_threshold: float = 0.0,
) -> ComparisonResult:
    """
    Paired bootstrap significance test.

    baseline_scores[i] and candidate_scores[i] are scores for the SAME test case.
    This pairing removes per-case difficulty variance.

    p_value = fraction of bootstrap samples where (candidate - baseline) <= regression_threshold
    If we're testing for regression: small p_value means the drop IS real.
    If we're testing for improvement: small p_value means the gain IS real.
    """
    assert len(baseline_scores) == len(candidate_scores), "Score arrays must be same length"
    n = len(baseline_scores)

    baseline = np.array(baseline_scores)
    candidate = np.array(candidate_scores)
    diffs = candidate - baseline
    observed_diff = np.mean(diffs)

    rng = np.random.default_rng(42)
    bootstrap_diffs = np.array([
        np.mean(rng.choice(diffs, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    # Two-sided p-value: P(|bootstrap_diff| >= |observed_diff|)
    p_value = float(np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff)))
    p_value = max(p_value, 1 / n_bootstrap)  # Prevent exact 0

    ci_low = float(np.percentile(bootstrap_diffs, 2.5))
    ci_high = float(np.percentile(bootstrap_diffs, 97.5))

    # Cohen's d for effect size
    pooled_std = float(np.std(np.concatenate([baseline, candidate])) + 1e-9)
    cohens_d = float(observed_diff / pooled_std)

    baseline_mean = float(np.mean(baseline))
    candidate_mean = float(np.mean(candidate))
    relative_diff = ((candidate_mean - baseline_mean) / (baseline_mean + 1e-9)) * 100

    significant = p_value < alpha

    if not significant:
        verdict = "NO_CHANGE"
        recommendation = (
            f"Difference of {observed_diff:+.4f} ({relative_diff:+.1f}%) is not statistically "
            f"significant (p={p_value:.4f}). Cannot distinguish from noise. INCONCLUSIVE."
        )
    elif observed_diff < -abs(regression_threshold):
        verdict = "REGRESSED"
        recommendation = (
            f"REGRESSION DETECTED: score dropped {observed_diff:+.4f} ({relative_diff:+.1f}%) "
            f"with p={p_value:.4f} (significant). CI=[{ci_low:+.4f}, {ci_high:+.4f}]. "
            f"Effect size (Cohen's d)={cohens_d:.3f}. Block this deployment."
        )
    else:
        verdict = "IMPROVED"
        recommendation = (
            f"Improvement of {observed_diff:+.4f} ({relative_diff:+.1f}%) is statistically "
            f"significant (p={p_value:.4f}). CI=[{ci_low:+.4f}, {ci_high:+.4f}]. Safe to deploy."
        )

    return ComparisonResult(
        baseline_mean=round(baseline_mean, 4),
        candidate_mean=round(candidate_mean, 4),
        absolute_diff=round(float(observed_diff), 4),
        relative_diff_pct=round(relative_diff, 2),
        p_value=round(p_value, 6),
        significant=significant,
        ci_low=round(ci_low, 4),
        ci_high=round(ci_high, 4),
        cohens_d=round(cohens_d, 4),
        n_samples=n,
        verdict=verdict,
        recommendation=recommendation,
    )


def mcnemar_test(
    baseline_correct: list[bool],
    candidate_correct: list[bool],
    alpha: float = 0.05,
) -> McNemarResult:
    """
    McNemar's test for paired binary outcomes.
    Used for intent classification: did the model get the case right or wrong?

    The test only cares about discordant pairs:
      b = baseline wrong, candidate right  (improvement)
      c = baseline right, candidate wrong  (regression)
    """
    assert len(baseline_correct) == len(candidate_correct)

    b = sum(1 for bas, cand in zip(baseline_correct, candidate_correct) if not bas and cand)
    c = sum(1 for bas, cand in zip(baseline_correct, candidate_correct) if bas and not cand)

    if b + c == 0:
        # No discordant pairs — cannot distinguish the two models
        return McNemarResult(
            p_value=1.0,
            significant=False,
            baseline_correct=sum(baseline_correct),
            candidate_correct=sum(candidate_correct),
            discordant_b=0,
            discordant_c=0,
            verdict="NO_CHANGE",
        )

    # McNemar's chi-squared statistic with continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - chi2.cdf(chi2_stat, df=1))

    significant = p_value < alpha
    if not significant:
        verdict = "NO_CHANGE"
    elif b > c:
        verdict = "IMPROVED"
    else:
        verdict = "REGRESSED"

    return McNemarResult(
        p_value=round(p_value, 6),
        significant=significant,
        baseline_correct=sum(baseline_correct),
        candidate_correct=sum(candidate_correct),
        discordant_b=b,
        discordant_c=c,
        verdict=verdict,
    )
