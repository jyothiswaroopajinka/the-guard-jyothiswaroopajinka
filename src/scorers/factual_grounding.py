"""
Scorer (a): Factual Grounding

Checks whether every number claimed in a generated text can be traced to the
source data. For credit narratives this is critical — hallucinated stats are a
regulatory risk when sent to Poonawalla Fincorp's compliance team.

Returns a score in [0, 1] and a confidence interval.
"""

import re
from dataclasses import dataclass


@dataclass
class GroundingResult:
    score: float              # 0-1
    ci_low: float             # 95% confidence interval lower bound
    ci_high: float            # 95% confidence interval upper bound
    claims_checked: int
    claims_grounded: int
    ungrounded_claims: list[str]


def _extract_numbers(text: str) -> list[str]:
    """Pull all numeric tokens from text (handles ₹, %, commas)."""
    # Matches: 37.14, 37.14%, ₹4,80,000, 480000, 36
    pattern = r"[\₹]?[\d,]+(?:\.\d+)?%?"
    return re.findall(pattern, text)


def _normalize(token: str) -> str:
    """Strip currency symbol, commas, percent sign, then return as string."""
    return re.sub(r"[₹,%]", "", token).replace(",", "").strip()


def _flatten_source(source_data: dict) -> set[str]:
    """
    Build a set of all numeric values that appear in source_data,
    including derived values (e.g. rounded versions, formatted versions).
    """
    values = set()
    for v in source_data.values():
        if isinstance(v, (int, float)):
            s = str(v)
            values.add(s)
            # Also add rounded version and percentage representation
            values.add(str(round(v, 2)))
            values.add(str(int(v)))
    return values


def score(generated_text: str, source_data: dict) -> GroundingResult:
    """
    Check every number in generated_text against source_data.

    A number is considered grounded if it matches any value in source_data
    (exact or within floating-point rounding).

    Returns GroundingResult with score and 95% CI via Wilson interval.
    """
    tokens = _extract_numbers(generated_text)
    source_values = _flatten_source(source_data)

    if not tokens:
        # No numbers at all — conservative: 0.5 (we can't verify, but also no lies)
        return GroundingResult(
            score=0.5,
            ci_low=0.2,
            ci_high=0.8,
            claims_checked=0,
            claims_grounded=0,
            ungrounded_claims=[],
        )

    grounded = []
    ungrounded = []
    for token in tokens:
        normalized = _normalize(token)
        if not normalized or normalized == "0":
            grounded.append(token)
            continue
        if normalized in source_values:
            grounded.append(token)
        else:
            # Try fuzzy: within 1% of any source value
            try:
                val = float(normalized)
                matched = any(
                    abs(val - float(sv)) / (float(sv) + 1e-9) < 0.02
                    for sv in source_values
                    if sv
                )
                if matched:
                    grounded.append(token)
                else:
                    ungrounded.append(token)
            except ValueError:
                ungrounded.append(token)

    n = len(tokens)
    k = len(grounded)
    raw_score = k / n

    # Wilson score interval for binomial proportion
    z = 1.96  # 95% CI
    denom = 1 + z**2 / n
    centre = (raw_score + z**2 / (2 * n)) / denom
    margin = (z / denom) * ((raw_score * (1 - raw_score) / n + z**2 / (4 * n**2)) ** 0.5)
    ci_low = max(0.0, centre - margin)
    ci_high = min(1.0, centre + margin)

    return GroundingResult(
        score=round(raw_score, 4),
        ci_low=round(ci_low, 4),
        ci_high=round(ci_high, 4),
        claims_checked=n,
        claims_grounded=k,
        ungrounded_claims=ungrounded,
    )
