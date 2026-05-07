from __future__ import annotations

"""
Eval Task 2: Insurance Intent Classification

For each deal object, ask the LLM to classify which insurance product to recommend.
Score using: intent_match (primary) + confidence calibration.

Uses OpenAI GPT-4o-mini by default (cheap, fast — right tool for classification).
Claude Sonnet as baseline for comparison.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.providers import call_model, estimate_cost_usd
from src.scorers import intent_match
from src.versioning import load_prompt

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "insurance_cases.json"


@dataclass
class InsuranceResult:
    case_id: str
    deal_object: dict
    ground_truth: str
    predicted_label: str
    predicted_confidence: float
    intent_score: float     # from intent_match scorer
    correct: bool
    is_edge_case: bool
    latency_ms: float
    cost_usd: float
    provider: str
    model: str
    raw_output: str
    errors: list[str] = field(default_factory=list)


def run_single(
    case: dict,
    provider: str = "openai",
    model: str | None = None,
    prompt_file: str = "v1.txt",
) -> InsuranceResult:
    errors = []
    system_prompt = load_prompt("insurance", prompt_file)
    user_prompt = (
        f"Classify the insurance intent for this deal object:\n"
        f"{json.dumps(case['deal_object'], indent=2)}"
    )

    t0 = time.time()
    try:
        raw_output, usage = call_model(
            prompt=user_prompt,
            system=system_prompt,
            provider=provider,
            model=model,
            max_tokens=200,
            temperature=0.1,  # Low temperature for classification consistency
        )
        cost = estimate_cost_usd(usage)
        used_model = usage["model"]
    except Exception as e:
        errors.append(f"Classification failed: {e}")
        raw_output = ""
        cost = 0.0
        used_model = model or "unknown"
    latency_ms = (time.time() - t0) * 1000

    result = intent_match.score(
        llm_output=raw_output,
        ground_truth_label=case["ground_truth_label"],
        confidence_threshold=case.get("confidence_threshold", 0.7),
    )

    return InsuranceResult(
        case_id=case["id"],
        deal_object=case["deal_object"],
        ground_truth=case["ground_truth_label"],
        predicted_label=result.predicted_label,
        predicted_confidence=result.predicted_confidence,
        intent_score=result.score,
        correct=result.correct,
        is_edge_case=case.get("edge_case", False),
        latency_ms=round(latency_ms, 1),
        cost_usd=round(cost, 6),
        provider=provider,
        model=used_model,
        raw_output=raw_output,
        errors=errors,
    )


def run_all(
    provider: str = "openai",
    model: str | None = None,
    prompt_file: str = "v1.txt",
    max_cases: int | None = None,
) -> list[InsuranceResult]:
    cases = json.loads(DATA_PATH.read_text())
    if max_cases:
        cases = cases[:max_cases]

    results = []
    for case in cases:
        result = run_single(case, provider=provider, model=model, prompt_file=prompt_file)
        results.append(result)

    return results
