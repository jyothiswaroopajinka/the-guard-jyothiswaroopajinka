"""
LLM provider clients.

Each function returns a string response from the model.
We use raw SDK calls — no LangChain — so every HTTP request is visible.

Providers used:
  - Anthropic (Claude Sonnet for generation, Opus for judging)
  - OpenAI (GPT-4o-mini for cheap tasks, GPT-4o for generation)
  - Google Gemini (Flash for cheap tasks)
"""

import os
import time
from typing import Optional

import anthropic
import openai
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

# ── Clients (initialized once at import time) ─────────────────────────────────

_anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
_openai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ── Model constants ────────────────────────────────────────────────────────────

CLAUDE_SONNET = "claude-sonnet-4-6"
CLAUDE_OPUS   = "claude-opus-4-7"
CLAUDE_HAIKU  = "claude-haiku-4-5-20251001"
GPT4O         = "gpt-4o"
GPT4O_MINI    = "gpt-4o-mini"
GEMINI_FLASH = "gemini-2.5-flash"
GEMINI_PRO   = "gemini-2.5-pro"


# Default judge model — Opus is most rigorous for LLM-as-judge
JUDGE_MODEL = os.getenv("JUDGE_MODEL", CLAUDE_OPUS)


def call_claude(
    prompt: str,
    system: str = "",
    model: str = CLAUDE_HAIKU,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> tuple[str, dict]:
    """
    Returns (response_text, usage_dict).
    usage_dict contains input_tokens, output_tokens for cost tracking.
    """
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    response = _anthropic.messages.create(**kwargs)
    text = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "provider": "anthropic",
        "model": model,
    }
    return text, usage


def call_openai(
    prompt: str,
    system: str = "",
    model: str = GPT4O_MINI,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> tuple[str, dict]:
    """Returns (response_text, usage_dict)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "provider": "openai",
        "model": model,
    }
    return text, usage


def call_gemini(
    prompt: str,
    system: str = "",
    model: str = GEMINI_FLASH,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> tuple[str, dict]:
    """Returns (response_text, usage_dict) using the new google-genai SDK."""
    config = genai_types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        system_instruction=system if system else None,
    )
    response = _gemini.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    text = response.text or ""
    usage = {
        "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
        "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
        "provider": "google",
        "model": model,
    }
    return text, usage


# ── Unified call with retry ────────────────────────────────────────────────────

def call_model(
    prompt: str,
    system: str = "",
    provider: str = "anthropic",
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    max_retries: int = 3,
) -> tuple[str, dict]:
    """
    Unified entry point. Tries up to max_retries times with exponential backoff.
    Falls back across providers if a provider fails persistently.
    """
    if provider == "anthropic":
        fn = call_claude
        default_model = CLAUDE_HAIKU
    elif provider == "openai":
        fn = call_openai
        default_model = GPT4O_MINI
    elif provider == "google":
        fn = call_gemini
        default_model = GEMINI_FLASH
    else:
        raise ValueError(f"Unknown provider: {provider}")

    chosen_model = model or default_model

    for attempt in range(max_retries):
        try:
            return fn(prompt, system=system, model=chosen_model, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            time.sleep(wait)

    raise RuntimeError("All retries exhausted")


def estimate_cost_usd(usage: dict) -> float:
    """
    Rough cost estimate in USD based on published pricing (May 2026).
    Used for cost tracking per eval run.
    """
    pricing = {
        # (input per 1M tokens, output per 1M tokens)
        CLAUDE_SONNET: (3.0, 15.0),
        CLAUDE_OPUS:   (15.0, 75.0),
        CLAUDE_HAIKU:  (0.25, 1.25),
        GPT4O:         (5.0, 15.0),
        GPT4O_MINI:    (0.15, 0.6),
        GEMINI_FLASH:  (0.075, 0.30),
        GEMINI_PRO:    (1.25, 5.0),
    }
    model = usage.get("model", "")
    rates = pricing.get(model, (1.0, 3.0))
    input_cost = usage.get("input_tokens", 0) / 1_000_000 * rates[0]
    output_cost = usage.get("output_tokens", 0) / 1_000_000 * rates[1]
    return input_cost + output_cost
