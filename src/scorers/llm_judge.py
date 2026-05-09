"""
Scorer (d): LLM-as-Judge

Uses Claude Sonnet to rate another model's deal copy on a 0-10 scale across
three dimensions: persuasiveness, factual accuracy, and channel fit.
Normalized to [0, 1].

Why Sonnet as judge? Strong reasoning for nuanced quality assessment without
the cost of Opus. Generation tasks use Haiku; judging uses Sonnet.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.providers import call_claude, CLAUDE_SONNET, CLAUDE_HAIKU, estimate_cost_usd  # noqa: F401

_JUDGES_DIR = Path(__file__).parent.parent.parent / "prompts" / "judges"


def _load_judge_prompt(filename: str) -> str:
    path = _JUDGES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Judge prompt not found: {path}")
    return path.read_text()


JUDGE_SYSTEM          = _load_judge_prompt("deal_copy_system.txt")
JUDGE_PROMPT_TEMPLATE = _load_judge_prompt("deal_copy_prompt.txt")


@dataclass
class JudgeResult:
    score: float           # 0-1, weighted average of all three dimensions
    ci_low: float
    ci_high: float
    persuasiveness: float  # 0-1
    factual_accuracy: float
    channel_fit: float
    reasoning: str
    cost_usd: float


def score(
    generated_text: str,
    reference_text: str,
    source_data: dict,
    channel: str,
    # judge_model: str = CLAUDE_SONNET,  # switch to Sonnet when budget allows
    judge_model: str = CLAUDE_HAIKU,
) -> JudgeResult:
    """
    Ask Claude Haiku to judge the generated copy.
    Returns scores normalized to [0, 1] with confidence interval.
    """
    prompt = (JUDGE_PROMPT_TEMPLATE
        .replace("{channel}", channel)
        .replace("{source_data}", json.dumps(source_data, ensure_ascii=False))
        .replace("{generated_text}", generated_text)
        .replace("{reference_text}", reference_text)
    )

    try:
        response_text, usage = call_claude(
            prompt=prompt,
            system=JUDGE_SYSTEM,
            model=judge_model,
            max_tokens=256,
            temperature=0.1,
        )
        cost = estimate_cost_usd(usage)
    except Exception as e:
        return JudgeResult(
            score=0.0, ci_low=0.0, ci_high=0.3,
            persuasiveness=0.0, factual_accuracy=0.0, channel_fit=0.0,
            reasoning=f"Judge call failed: {e}",
            cost_usd=0.0,
        )

    try:
        text = response_text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if match:
            text = match.group(1)
        parsed = json.loads(text)

        p = min(10, max(0, int(parsed["persuasiveness"]))) / 10.0
        f = min(10, max(0, int(parsed["factual_accuracy"]))) / 10.0
        c = min(10, max(0, int(parsed["channel_fit"]))) / 10.0
        reasoning = parsed.get("overall_reasoning", "")

        # Weighted average: factual accuracy weighted highest for compliance-critical outputs
        weights = {"persuasiveness": 0.3, "factual_accuracy": 0.4, "channel_fit": 0.3}
        overall = p * weights["persuasiveness"] + f * weights["factual_accuracy"] + c * weights["channel_fit"]

        # CI based on judge variance (Opus judge variability is ~±0.1 on 0-1 scale)
        ci_low = max(0.0, overall - 0.1)
        ci_high = min(1.0, overall + 0.1)

        return JudgeResult(
            score=round(overall, 4),
            ci_low=round(ci_low, 4),
            ci_high=round(ci_high, 4),
            persuasiveness=round(p, 4),
            factual_accuracy=round(f, 4),
            channel_fit=round(c, 4),
            reasoning=reasoning,
            cost_usd=round(cost, 6),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return JudgeResult(
            score=0.0, ci_low=0.0, ci_high=0.3,
            persuasiveness=0.0, factual_accuracy=0.0, channel_fit=0.0,
            reasoning=f"Parse error: {e} | Raw: {response_text[:200]}",
            cost_usd=cost,
        )
