"""
Eval Task 1: Deal Copy Quality

For each test case, we:
1. Call the LLM (provider + model under test) to generate deal copy
2. Score with 3 scorers: format_compliance, llm_judge, semantic_similarity
   (factual_grounding is also applied)
3. Return per-case scores for statistical aggregation

This module generates — it does NOT do comparison. Comparison happens in run_eval.py.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.providers import call_model, estimate_cost_usd
from src.scorers import factual_grounding, format_compliance, llm_judge, semantic_similarity
from src.versioning import load_prompt

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "deal_copy_cases.json"


@dataclass
class DealCopyResult:
    case_id: str
    channel: str
    merchant: str
    generated_text: str
    reference_text: str
    source_data: dict
    format_score: float
    grounding_score: float
    judge_score: float
    similarity_score: float
    composite_score: float     # weighted average of all four
    latency_ms: float
    cost_usd: float
    provider: str
    model: str
    prompt_file: str
    errors: list[str] = field(default_factory=list)


def _composite(format_s: float, grounding_s: float, judge_s: float, sim_s: float) -> float:
    """
    Weighted composite score.
    Format compliance is hard constraint — weight it highest.
    Judge covers persuasiveness so similarity is supplementary.
    """
    return (
        format_s    * 0.25 +
        grounding_s * 0.30 +
        judge_s     * 0.35 +
        sim_s       * 0.10
    )


def run_single(
    case: dict,
    provider: str = "anthropic",
    model: str | None = None,
    prompt_file: str = "v1.txt",
    use_judge: bool = True,
) -> DealCopyResult:
    """Run a single deal copy eval case."""
    errors = []
    system_prompt = load_prompt("deal_copy", prompt_file)

    user_prompt = (
        f"Channel: {case['channel']}\n"
        f"Merchant: {case['merchant']}\n"
        f"Category: {case['category']}\n"
        f"Source data: {json.dumps(case['source_data'])}\n\n"
        f"Generate the deal copy now."
    )

    t0 = time.time()
    try:
        generated_text, usage = call_model(
            prompt=user_prompt,
            system=system_prompt,
            provider=provider,
            model=model,
            max_tokens=256,
            temperature=0.3,
        )
        cost = estimate_cost_usd(usage)
        used_model = usage["model"]
    except Exception as e:
        errors.append(f"Generation failed: {e}")
        generated_text = ""
        cost = 0.0
        used_model = model or "unknown"
    latency_ms = (time.time() - t0) * 1000

    # Score 1: Format compliance (deterministic, no LLM)
    fmt = format_compliance.score(generated_text, case["channel"])

    # Score 2: Factual grounding (regex-based, no LLM)
    grnd = factual_grounding.score(generated_text, case["source_data"])

    # Score 3: LLM-as-judge (Opus judges the output)
    if use_judge and generated_text:
        try:
            jdg = llm_judge.score(
                generated_text=generated_text,
                reference_text=case["reference_output"],
                source_data=case["source_data"],
                channel=case["channel"],
            )
            judge_s = jdg.score
            cost += jdg.cost_usd
        except Exception as e:
            errors.append(f"Judge failed: {e}")
            judge_s = 0.0
    else:
        judge_s = 0.0

    # Score 4: Semantic similarity
    if generated_text:
        try:
            sim = semantic_similarity.score(generated_text, case["reference_output"])
            sim_s = sim.score
        except Exception as e:
            errors.append(f"Similarity failed: {e}")
            sim_s = 0.0
    else:
        sim_s = 0.0

    composite = _composite(fmt.score, grnd.score, judge_s, sim_s)

    return DealCopyResult(
        case_id=case["id"],
        channel=case["channel"],
        merchant=case["merchant"],
        generated_text=generated_text,
        reference_text=case["reference_output"],
        source_data=case["source_data"],
        format_score=fmt.score,
        grounding_score=grnd.score,
        judge_score=judge_s,
        similarity_score=sim_s,
        composite_score=round(composite, 4),
        latency_ms=round(latency_ms, 1),
        cost_usd=round(cost, 6),
        provider=provider,
        model=used_model,
        prompt_file=prompt_file,
        errors=errors,
    )


def run_all(
    provider: str = "anthropic",
    model: str | None = None,
    prompt_file: str = "v1.txt",
    max_cases: int | None = None,
    use_judge: bool = True,
) -> list[DealCopyResult]:
    """Run all deal copy eval cases. Returns list of results."""
    cases = json.loads(DATA_PATH.read_text())
    if max_cases:
        cases = cases[:max_cases]

    results = []
    for case in cases:
        result = run_single(case, provider=provider, model=model,
                            prompt_file=prompt_file, use_judge=use_judge)
        results.append(result)

    return results
