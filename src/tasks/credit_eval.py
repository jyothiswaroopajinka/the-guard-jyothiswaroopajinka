"""
Eval Task 3: Credit Narrative Faithfulness

For each user persona, generate a credit narrative and score it for:
1. Factual grounding (primary — regulatory risk if hallucinated)
2. Semantic similarity to reference narrative
3. LLM-as-judge on professional tone and completeness

Uses Claude Sonnet (capable model needed for narrative generation quality).
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.providers import call_model, estimate_cost_usd
from src.scorers import factual_grounding, semantic_similarity, llm_judge
from src.versioning import load_prompt

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "credit_cases.json"

CREDIT_JUDGE_SYSTEM = """You are a compliance reviewer for Poonawalla Fincorp.
Grade credit narratives for quality. Output ONLY valid JSON."""

CREDIT_JUDGE_PROMPT = """Grade this GrabOn credit narrative.

USER DATA: {persona}

GENERATED NARRATIVE:
{generated}

REFERENCE NARRATIVE:
{reference}

Grade 0-10 on:
1. factual_accuracy: Every number/stat traces to the user data? (10=perfect)
2. professional_tone: Suitable for banking compliance team? (10=excellent)
3. completeness: Covers key credit decision factors? (10=complete)

Output ONLY JSON (no markdown):
{{
  "factual_accuracy": <int>,
  "professional_tone": <int>,
  "completeness": <int>,
  "overall_reasoning": "<one sentence>"
}}"""


@dataclass
class CreditResult:
    case_id: str
    user_id: str
    persona: dict
    ground_truth_eligible: bool
    generated_narrative: str
    reference_narrative: str
    grounding_score: float
    similarity_score: float
    judge_score: float
    composite_score: float
    latency_ms: float
    cost_usd: float
    provider: str
    model: str
    prompt_file: str
    errors: list[str] = field(default_factory=list)


def run_single(
    case: dict,
    provider: str = "anthropic",
    model: str | None = None,
    prompt_file: str = "v1.txt",
    use_judge: bool = True,
) -> CreditResult:
    errors = []
    system_prompt = load_prompt("credit", prompt_file)
    persona = case["persona"]
    user_prompt = (
        f"Write a credit narrative for the following GrabOn user.\n\n"
        f"User data:\n{json.dumps(persona, indent=2)}\n\n"
        f"Eligible: {case['label']['eligible']}\n"
        f"Credit limit: ₹{case['label']['credit_limit_inr']:,}"
    )

    t0 = time.time()
    try:
        generated_text, usage = call_model(
            prompt=user_prompt,
            system=system_prompt,
            provider=provider,
            model=model,
            max_tokens=300,
            temperature=0.2,
        )
        cost = estimate_cost_usd(usage)
        used_model = usage["model"]
    except Exception as e:
        errors.append(f"Generation failed: {e}")
        generated_text = ""
        cost = 0.0
        used_model = model or "unknown"
    latency_ms = (time.time() - t0) * 1000

    # Score 1: Factual grounding — most important for credit narratives
    grnd = factual_grounding.score(generated_text, persona)

    # Score 2: Semantic similarity to reference
    if generated_text:
        try:
            sim = semantic_similarity.score(generated_text, case["reference_narrative"])
            sim_s = sim.score
        except Exception as e:
            errors.append(f"Similarity failed: {e}")
            sim_s = 0.0
    else:
        sim_s = 0.0

    # Score 3: LLM judge for professional tone (credit-specific prompt)
    judge_s = 0.0
    if use_judge and generated_text:
        try:
            import re
            judge_prompt = CREDIT_JUDGE_PROMPT.format(
                persona=json.dumps(persona),
                generated=generated_text,
                reference=case["reference_narrative"],
            )
            judge_text, judge_usage = call_model(
                prompt=judge_prompt,
                system=CREDIT_JUDGE_SYSTEM,
                provider="anthropic",
                model="claude-opus-4-7",
                max_tokens=200,
                temperature=0.1,
            )
            cost += estimate_cost_usd(judge_usage)

            text = judge_text.strip()
            m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
            parsed = json.loads(m.group(1) if m else text)
            fa = min(10, max(0, int(parsed["factual_accuracy"]))) / 10.0
            pt = min(10, max(0, int(parsed["professional_tone"]))) / 10.0
            co = min(10, max(0, int(parsed["completeness"]))) / 10.0
            # Factual accuracy weighted highest for compliance
            judge_s = fa * 0.5 + pt * 0.25 + co * 0.25
        except Exception as e:
            errors.append(f"Judge failed: {e}")

    # Composite: grounding is critical for credit (regulatory), weight highest
    composite = grnd.score * 0.45 + judge_s * 0.35 + sim_s * 0.20

    return CreditResult(
        case_id=case["id"],
        user_id=persona["user_id"],
        persona=persona,
        ground_truth_eligible=case["label"]["eligible"],
        generated_narrative=generated_text,
        reference_narrative=case["reference_narrative"],
        grounding_score=grnd.score,
        similarity_score=sim_s,
        judge_score=judge_s,
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
) -> list[CreditResult]:
    cases = json.loads(DATA_PATH.read_text())
    if max_cases:
        cases = cases[:max_cases]

    results = []
    for case in cases:
        result = run_single(case, provider=provider, model=model,
                            prompt_file=prompt_file, use_judge=use_judge)
        results.append(result)

    return results
