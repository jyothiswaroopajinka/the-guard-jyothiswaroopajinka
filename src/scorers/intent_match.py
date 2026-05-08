"""
Scorer (b): Intent Match

For insurance classification tasks. Checks if the predicted label matches the
ground-truth label, and penalizes overconfident wrong predictions (calibration).

Returns a score in [0, 1] with confidence interval.
"""

import json
import re
from dataclasses import dataclass


@dataclass
class IntentResult:
    score: float
    ci_low: float
    ci_high: float
    predicted_label: str
    ground_truth: str
    correct: bool
    predicted_confidence: float
    calibration_penalty: float


def _parse_llm_json(text: str) -> dict:
    """Extract JSON from LLM output, handling markdown code blocks and extra text."""
    text = text.strip()
    # Strip ```json ... ``` if present
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return json.loads(match.group(1))
    # Try parsing the whole thing as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the first { ... } block in the text (handles models that add preamble text)
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        return json.loads(match.group(0))
    raise json.JSONDecodeError("No JSON found", text, 0)


def score(
    llm_output: str,
    ground_truth_label: str,
    confidence_threshold: float = 0.7,
) -> IntentResult:
    """
    Parse the LLM's JSON output and compare against ground truth.

    Scoring:
    - Correct prediction: 1.0 base
    - Wrong prediction with high confidence (>0.8): penalized heavily (0.0)
    - Wrong prediction with appropriate uncertainty (<0.6): partial credit (0.3)
    - Confidence calibration: subtract penalty if confidence >> 1.0 on wrong answer

    Returns score with Wilson 95% CI (treated as single Bernoulli trial).
    """
    try:
        parsed = _parse_llm_json(llm_output)
        predicted = parsed.get("label", "").strip()
        confidence = float(parsed.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError, AttributeError):
        # If output isn't valid JSON, score as 0
        return IntentResult(
            score=0.0,
            ci_low=0.0,
            ci_high=0.31,  # Wilson CI for 0/1 (Laplace smoothed)
            predicted_label="<parse_error>",
            ground_truth=ground_truth_label,
            correct=False,
            predicted_confidence=0.0,
            calibration_penalty=0.0,
        )

    correct = predicted == ground_truth_label

    if correct:
        # Reward confident correct predictions, slight penalty for very low confidence on easy cases
        base = 1.0
        calibration_penalty = max(0.0, confidence_threshold - confidence) * 0.3
        raw_score = base - calibration_penalty
    else:
        # Penalize overconfident wrong answers more severely
        if confidence >= 0.8:
            raw_score = 0.0
            calibration_penalty = confidence
        elif confidence >= 0.6:
            raw_score = 0.2
            calibration_penalty = confidence * 0.5
        else:
            raw_score = 0.3  # At least uncertain, which is honest
            calibration_penalty = 0.0

    # Wilson interval for a single binary trial (n=1 is rough — aggregate across cases)
    # Here we return the per-case score; aggregation happens in statistical.py
    ci_half = 0.31 if raw_score in (0.0, 1.0) else 0.2
    ci_low = max(0.0, raw_score - ci_half)
    ci_high = min(1.0, raw_score + ci_half)

    return IntentResult(
        score=round(raw_score, 4),
        ci_low=round(ci_low, 4),
        ci_high=round(ci_high, 4),
        predicted_label=predicted,
        ground_truth=ground_truth_label,
        correct=correct,
        predicted_confidence=round(confidence, 4),
        calibration_penalty=round(calibration_penalty, 4),
    )
