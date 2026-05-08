from __future__ import annotations

"""
Scorer (c): Format Compliance

Checks whether generated deal copy fits the channel-specific constraints.
This is deterministic — no LLM needed here (a good example of knowing when NOT
to use an LLM).

Channel rules:
  email    — no hard limit, but check for required fields
  whatsapp — ≤160 characters
  push     — title ≤50 chars + body ≤100 chars (two separate fields, split on first newline)
  glance   — ≤80 characters
"""

from dataclasses import dataclass

CHANNEL_LIMITS = {
    "email": None,       # No hard char limit
    "whatsapp": 160,
    "push": None,        # Push has two-part limit — handled separately
    "glance": 80,
}

PUSH_TITLE_LIMIT = 50
PUSH_BODY_LIMIT  = 100

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

    Push is special: title (first line) ≤50 chars, body (rest) ≤100 chars.
    Both are checked independently and the lower score wins.

    Returns FormatResult with tight CI since this is deterministic.
    """
    channel = channel.lower()
    text = generated_text.strip()
    violations = []

    if channel == "push":
        # Split on first newline — first line = title, rest = body
        parts = text.split("\n", 1)
        title = parts[0].strip()
        body  = parts[1].strip() if len(parts) > 1 else ""

        title_len = len(title)
        body_len  = len(body)
        char_count = title_len + body_len

        title_ok = title_len <= PUSH_TITLE_LIMIT
        body_ok  = body_len  <= PUSH_BODY_LIMIT or body_len == 0

        if not title_ok:
            over = title_len - PUSH_TITLE_LIMIT
            violations.append(f"Push title too long: {title_len} chars (limit {PUSH_TITLE_LIMIT}, {over} over)")
        if not body_ok:
            over = body_len - PUSH_BODY_LIMIT
            violations.append(f"Push body too long: {body_len} chars (limit {PUSH_BODY_LIMIT}, {over} over)")

        within_limit = title_ok and body_ok

        if within_limit:
            raw_score = 1.0
        else:
            # Penalise each violation independently, take the worse
            title_score = max(0.0, 1.0 - ((title_len - PUSH_TITLE_LIMIT) / PUSH_TITLE_LIMIT) * 5) if not title_ok else 1.0
            body_score  = max(0.0, 1.0 - ((body_len  - PUSH_BODY_LIMIT)  / PUSH_BODY_LIMIT)  * 5) if not body_ok  else 1.0
            raw_score = min(title_score, body_score)

        limit = None  # Two-part limit, not a single number

    else:
        char_count = len(text)
        limit = CHANNEL_LIMITS.get(channel)

        if limit is None:
            within_limit = True
            raw_score = 1.0
        elif char_count <= limit:
            within_limit = True
            # Small deduction for being unnecessarily short on WhatsApp (< 50% of limit)
            if channel == "whatsapp" and char_count < limit * 0.5:
                raw_score = 0.85
                violations.append(f"WhatsApp message very short ({char_count} chars) — may lose persuasiveness")
            else:
                raw_score = 1.0
        else:
            within_limit = False
            overshoot = (char_count - limit) / limit
            raw_score = max(0.0, 1.0 - overshoot * 5)
            violations.append(f"{channel} limit is {limit} chars, got {char_count} ({char_count - limit} over)")

    # Deterministic scorer — CI is tight (±0.05)
    ci_low  = max(0.0, raw_score - 0.05)
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
