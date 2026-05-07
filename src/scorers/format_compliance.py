"""
Scorer (c): Format Compliance

Checks whether generated deal copy fits the channel-specific constraints.
This is deterministic — no LLM needed here (a good example of knowing when NOT
to use an LLM).

Channel rules:
  email    — no hard limit, but check for required fields
  whatsapp — ≤160 characters
  push     — ≤50 characters (title)
  glance   — ≤80 characters
"""

from dataclasses import dataclass

CHANNEL_LIMITS = {
    "email": None,       # No hard char limit
    "whatsapp": 160,
    "push": 50,
    "glance": 80,
}

REQUIRED_FIELDS_BY_CHANNEL = {
    "email": [],    # Checked separately via factual grounding
    "whatsapp": [],
    "push": [],
    "glance": [],
}


@dataclass
class FormatResult:
    score: float
    ci_low: float
    ci_high: float
    channel: str
    char_count: int
    char_limit: int | None
    within_limit: bool
    violations: list[str]


def score(generated_text: str, channel: str) -> FormatResult:
    """
    Score format compliance for a given channel.

    Score breakdown:
    - 1.0 if within character limit (or no limit)
    - 0.0 if over character limit
    - Partial deductions for being within 10% of limit (nearly over)

    Returns FormatResult with tight CI since this is deterministic.
    """
    channel = channel.lower()
    char_count = len(generated_text.strip())
    limit = CHANNEL_LIMITS.get(channel)
    violations = []

    if limit is None:
        within_limit = True
        raw_score = 1.0
    elif char_count <= limit:
        within_limit = True
        # Small deduction for being unnecessarily short on WhatsApp (< 50% of limit)
        if channel == "whatsapp" and char_count < limit * 0.5:
            raw_score = 0.85  # Too brief, may be missing info
            violations.append(f"WhatsApp message very short ({char_count} chars) — may lose persuasiveness")
        else:
            raw_score = 1.0
    else:
        within_limit = False
        overshoot = (char_count - limit) / limit
        # Linear penalty: 10% over → 0.5, 50%+ over → 0.0
        raw_score = max(0.0, 1.0 - overshoot * 5)
        violations.append(
            f"{channel} limit is {limit} chars, got {char_count} ({char_count - limit} over)"
        )

    # Deterministic scorer — CI is tight (±0.05)
    ci_low = max(0.0, raw_score - 0.05)
    ci_high = min(1.0, raw_score + 0.05)

    return FormatResult(
        score=round(raw_score, 4),
        ci_low=round(ci_low, 4),
        ci_high=round(ci_high, 4),
        channel=channel,
        char_count=char_count,
        char_limit=limit,
        within_limit=within_limit,
        violations=violations,
    )
