"""
Prompt Versioning

Tracks prompt versions using content hashing + git metadata.
When a regression is detected, we can diff the two prompt versions to show
exactly what changed and which commit introduced it.

Design: Each prompt file is hashed (SHA-256). The hash + git commit hash +
timestamp form the version identifier. Stored in a local JSON registry
(prompt_versions.json) that lives in git alongside the prompts.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False


REGISTRY_PATH = Path(__file__).parent.parent / "prompt_versions.json"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@dataclass
class PromptVersion:
    task: str           # "deal_copy", "insurance", "credit"
    filename: str       # "v1.txt", "v2.txt"
    content_hash: str   # SHA-256 of file content (first 12 chars)
    git_commit: str     # git commit hash (first 8 chars), "none" if no git
    timestamp: str      # ISO8601
    filepath: str


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _get_git_commit() -> str:
    if not GIT_AVAILABLE:
        return "none"
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha[:8]
    except Exception:
        return "none"


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _save_registry(reg: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


def register_all_prompts() -> list[PromptVersion]:
    """
    Scan all prompts/ subdirectories and register their current versions.
    Call this before running an eval so we know what prompt version was active.
    """
    registry = _load_registry()
    versions = []
    git_commit = _get_git_commit()
    now = datetime.now(timezone.utc).isoformat()

    for task_dir in sorted(PROMPTS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for prompt_file in sorted(task_dir.glob("*.txt")):
            content = prompt_file.read_text()
            content_hash = _sha256(content)[:12]
            key = f"{task}/{prompt_file.name}"

            pv = PromptVersion(
                task=task,
                filename=prompt_file.name,
                content_hash=content_hash,
                git_commit=git_commit,
                timestamp=now,
                filepath=str(prompt_file.relative_to(PROMPTS_DIR.parent)),
            )

            # Only update if content changed
            if key not in registry or registry[key]["content_hash"] != content_hash:
                registry[key] = asdict(pv)

            versions.append(pv)

    _save_registry(registry)
    return versions


def get_version(task: str, filename: str) -> PromptVersion | None:
    """Look up a specific prompt version from the registry."""
    registry = _load_registry()
    key = f"{task}/{filename}"
    entry = registry.get(key)
    if not entry:
        return None
    return PromptVersion(**entry)


def diff_prompts(task: str, file_a: str, file_b: str) -> str:
    """
    Return a unified diff of two prompt versions.
    Used in regression reports to highlight what changed.
    """
    import difflib
    path_a = PROMPTS_DIR / task / file_a
    path_b = PROMPTS_DIR / task / file_b

    if not path_a.exists():
        return f"File not found: {path_a}"
    if not path_b.exists():
        return f"File not found: {path_b}"

    lines_a = path_a.read_text().splitlines(keepends=True)
    lines_b = path_b.read_text().splitlines(keepends=True)

    diff = difflib.unified_diff(
        lines_a, lines_b,
        fromfile=f"{task}/{file_a}",
        tofile=f"{task}/{file_b}",
    )
    return "".join(diff) or "No differences found."


def load_prompt(task: str, filename: str) -> str:
    """Load prompt text from file."""
    path = PROMPTS_DIR / task / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text()
