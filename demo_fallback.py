"""
demo_fallback.py — Live demo of The Guard's provider fallback chain.

Shows three scenarios:
  1. Primary (Anthropic) works fine — no fallback needed
  2. Primary fails (bad key) — falls back to Google Gemini
  3. Primary + secondary both fail — falls back to OpenAI

Run:
  python demo_fallback.py
"""

from __future__ import annotations

import os
from unittest.mock import patch

from rich.console import Console
from rich.panel import Panel

console = Console()

PROMPT = "What is 2 + 2? Reply with just the number."


def _show_result(scenario: str, text: str, usage: dict) -> None:
    provider = usage.get("fallback_to") or usage.get("provider", "?")
    fallback_from = usage.get("fallback_from")

    if fallback_from:
        status = f"[yellow]FALLBACK[/] {fallback_from} → {provider}"
    else:
        status = f"[green]PRIMARY[/]  {provider}"

    console.print(f"  {status}  |  response: [bold]{text.strip()}[/]  |  model: {usage['model']}")


def scenario_1_primary_works() -> None:
    """Normal path — Anthropic responds fine."""
    console.print(Panel(
        "[bold]Scenario 1:[/] Primary provider (Anthropic) works normally.\n"
        "Expected: Anthropic answers directly, no fallback.",
        title="[cyan]Scenario 1 — Primary OK[/]",
        border_style="cyan",
    ))
    from src.providers import call_model
    try:
        text, usage = call_model(PROMPT, provider="anthropic")
        _show_result("1", text, usage)
    except Exception as e:
        console.print(f"  [red]Error:[/] {e}")
    console.print()


def scenario_2_anthropic_fails() -> None:
    """Anthropic key is broken — should fall back to Google Gemini."""
    console.print(Panel(
        "[bold]Scenario 2:[/] Anthropic API key is bad (simulated).\n"
        "Expected: falls back to Google Gemini Flash.",
        title="[yellow]Scenario 2 — Anthropic Fails → Google Fallback[/]",
        border_style="yellow",
    ))
    from src import providers

    original_call_claude = providers.call_claude

    def broken_claude(*args, **kwargs):
        raise Exception("auth: invalid_api_key — simulated bad key")

    providers.call_claude = broken_claude
    try:
        from src.providers import call_model
        text, usage = call_model(PROMPT, provider="anthropic", enable_fallback=True)
        _show_result("2", text, usage)
    except Exception as e:
        console.print(f"  [red]All providers failed:[/] {e}")
    finally:
        providers.call_claude = original_call_claude
    console.print()


def scenario_3_anthropic_and_google_fail() -> None:
    """Both Anthropic and Google fail — should fall back to OpenAI."""
    console.print(Panel(
        "[bold]Scenario 3:[/] Anthropic + Google both fail (simulated).\n"
        "Expected: falls back to OpenAI GPT-4o-mini.",
        title="[red]Scenario 3 — Two Providers Fail → OpenAI Fallback[/]",
        border_style="red",
    ))
    from src import providers

    original_call_claude = providers.call_claude
    original_call_gemini = providers.call_gemini

    def broken_claude(*args, **kwargs):
        raise Exception("auth: invalid_api_key — simulated bad Anthropic key")

    def broken_gemini(*args, **kwargs):
        raise Exception("billing: quota_exceeded — simulated Gemini billing issue")

    providers.call_claude = broken_claude
    providers.call_gemini = broken_gemini
    try:
        from src.providers import call_model
        text, usage = call_model(PROMPT, provider="anthropic", enable_fallback=True)
        _show_result("3", text, usage)
    except Exception as e:
        console.print(f"  [red]All providers failed:[/] {e}")
    finally:
        providers.call_claude = original_call_claude
        providers.call_gemini = original_call_gemini
    console.print()


def scenario_4_all_fail() -> None:
    """All three providers fail — shows graceful error."""
    console.print(Panel(
        "[bold]Scenario 4:[/] All providers fail (simulated outage).\n"
        "Expected: raises an error — no silent failure.",
        title="[red]Scenario 4 — All Providers Down[/]",
        border_style="red",
    ))
    from src import providers

    original_claude = providers.call_claude
    original_gemini = providers.call_gemini
    original_openai = providers.call_openai

    def broken(*args, **kwargs):
        raise Exception("503: service unavailable — simulated outage")

    providers.call_claude = broken
    providers.call_gemini = broken
    providers.call_openai = broken
    try:
        from src.providers import call_model
        text, usage = call_model(PROMPT, provider="anthropic", enable_fallback=True)
        _show_result("4", text, usage)
    except Exception as e:
        console.print(f"  [bold red]Expected error caught:[/] {e}")
        console.print("  [green]✓ No silent failure — caller gets a clear exception.[/]")
    finally:
        providers.call_claude = original_claude
        providers.call_gemini = original_gemini
        providers.call_openai = original_openai
    console.print()


if __name__ == "__main__":
    console.print("\n[bold cyan]The Guard — Provider Fallback Chain Demo[/]\n")
    console.print("Fallback order: [green]Anthropic[/] → [blue]Google Gemini[/] → [magenta]OpenAI[/]\n")

    scenario_1_primary_works()
    scenario_2_anthropic_fails()
    scenario_3_anthropic_and_google_fail()
    scenario_4_all_fail()

    console.print("[bold green]Demo complete.[/] The Guard never silently uses a broken provider.\n")
