# The Guard — GrabOn AI Eval Pipeline

**Assignment 03 | GrabOn AI Labs Agentic AI Engineer Challenge 2026**

A production eval framework that detects quality regressions in GrabOn's AI-generated outputs before they ship to 40M users or a bank's compliance team.

---

## What I Built and Why This Assignment

I chose Assignment 03 (Eval Engineering) because it's the hardest part of shipping AI in production that most engineers skip. Anyone can prompt an LLM and get output. The real question is: *how do you know if the output got worse after you changed something?*

GrabOn's AI produces three types of outputs where quality failures have real consequences:

1. **Deal copy** — sent to 40M subscribers across email, WhatsApp, push, and Glance. A hallucinated discount percentage destroys trust.
2. **Insurance intent classification** — determines which micro-policy a user sees at checkout. A wrong label means wrong product, wrong user.
3. **Credit narratives** — go to Poonawalla Fincorp's compliance team. A hallucinated stat is a regulatory violation.

The Guard catches regressions in all three *before deployment*, using statistical tests that distinguish real drops from noise.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    The Guard — Eval Agent Loop                   │
│                                                                  │
│  PLAN          ACT             OBSERVE          DECIDE           │
│  ─────         ────            ───────          ──────           │
│  Load cases    Call LLM        Score output     GO / NO-GO       │
│  Select model  (providers.py)  (5 scorers)      gate             │
│  Hash prompts  Generate text   Grounding        ↓                │
│  Register ver. Classify intent Format check     Exit code        │
│  ↓             ↓               Judge quality    0 / 1            │
│  30 cases ×    Baseline run    Semantic sim.    ↓                │
│  2 runs        Candidate run   ↓                CI blocks PR     │
│  (baseline +   Sequential,     Per-case scores  or passes it     │
│   candidate)   with retry +    aggregated       ↓                │
│                fallback        ↓                Save JSON +      │
│                                Paired bootstrap print report     │
│                                McNemar's test                    │
│                                p-values + CIs                    │
└─────────────────────────────────────────────────────────────────┘
```

```
the-guard/
│
├── run_eval.py                      ← Entry point (CLI + agent loop)
├── analyze_coverage.py              ← Dataset diversity report
├── requirements.txt
├── .env
│
├── src/                             ← All application code
│   ├── providers.py                 ← ACT: Anthropic / OpenAI / Google clients
│   │                                    smart retry + provider fallback chain
│   ├── versioning.py                ← PLAN: SHA-256 hash + git commit per prompt
│   ├── statistical.py               ← OBSERVE: paired bootstrap + McNemar's test
│   ├── gate.py                      ← DECIDE: GO / NO-GO + failure reasons
│   ├── dashboard.py                 ← DECIDE: terminal report + JSON persistence
│   │
│   ├── tasks/                       ← ACT: one runner per eval task
│   │   ├── deal_copy_eval.py
│   │   ├── insurance_eval.py
│   │   └── credit_eval.py
│   │
│   └── scorers/                     ← OBSERVE: 5 scoring functions
│       ├── factual_grounding.py     (a) regex — are all numbers traceable to source?
│       ├── intent_match.py          (b) deterministic — label match + calibration
│       ├── format_compliance.py     (c) deterministic — channel character limits
│       ├── llm_judge.py             (d) Claude Haiku grades output quality
│       └── semantic_similarity.py   (e) cosine similarity via sentence-transformers
│
├── prompts/                         ← Versioned prompt files
│   ├── deal_copy/
│   │   ├── v1.txt                   baseline prompt
│   │   └── v2.txt                   degraded prompt (for regression demo)
│   ├── insurance/
│   │   └── v1.txt
│   ├── credit/
│   │   └── v1.txt
│   └── judges/                      LLM-as-judge templates (loaded at runtime)
│       ├── deal_copy_system.txt
│       ├── deal_copy_prompt.txt
│       ├── credit_system.txt
│       └── credit_prompt.txt
│
├── data/                            ← Test datasets (static, versioned in git)
│   ├── deal_copy_cases.json         58 cases — 4 channels, 8 edge cases
│   ├── insurance_cases.json         46 cases — 5 labels, 16 edge cases
│   └── credit_cases.json            44 cases — eligible/ineligible, 4 edge cases
│
└── .github/
    └── workflows/
        └── eval-guard.yml           ← CI/CD: blocks PR on NO-GO
```

---

## Per-Module Design Decisions and Tradeoffs

### `src/providers.py` — Multi-LLM with smart retry
Three providers, chosen deliberately per task type. The `call_model()` function has two layers of resilience:
- **Smart retry**: Only retries transient errors (rate limit, 429, timeout, 500s). Never retries permanent failures (bad API key, billing not active) — those will fail on every attempt anyway.
- **Fallback chain**: If the primary provider fails after retries, automatically tries Anthropic → Google in sequence. The fallback tags the usage dict so callers know a fallback was used.

Tradeoff: fallback means you might compare Claude vs Gemini output within the same run if the primary fails mid-way. Acceptable for eval (every case logs which model actually ran), unacceptable for production generation.

### `src/scorers/` — Deterministic first, LLM last
The scoring philosophy: use the cheapest tool that answers the question accurately.
- Format compliance and factual grounding use **pure Python/regex** — no LLM, no network call, exact answer.
- Semantic similarity uses **sentence-transformers locally** — free, fast, no API cost.
- LLM-as-judge is last resort, only for things only an LLM can assess (persuasiveness, professional tone).

This makes the eval cheap to run in CI. Only 2 of the 5 scorers make API calls.

### `src/statistical.py` — Why paired bootstrap, not a t-test
A t-test assumes Gaussian-distributed scores. Our scores are bounded [0,1] and skewed. Paired bootstrap makes no distribution assumptions — it just resamples the observed differences 10,000 times. Pairing matters because each test case has a difficulty level; pairing removes that variance so we only measure what the model/prompt change contributed.

McNemar's test is used for insurance because the outcome is binary (correct/incorrect). Using a continuous test on binary data would be wrong.

### `src/gate.py` — Two-trigger NO-GO
A regression blocks deployment if EITHER:
1. Statistically significant: p < 0.05 AND absolute drop > 0.05
2. Practically significant: drop > 10% regardless of p-value (catches real regressions that are noisy across cases)

The practical threshold exists because with 30 cases, a 15% drop might not reach p < 0.05 due to variance. But a 15% drop is never acceptable. Conservative by design — false negatives (missing a regression) cost more than false positives (blocking a good deploy).

### `src/versioning.py` — Content hash, not filename
Prompt files are SHA-256 hashed at the start of every eval run. The hash + git commit + timestamp are stored in `prompt_versions.json`. This means if someone edits `v1.txt` without renaming it, the system detects the change. On regression, the runner automatically diffs the two prompt files and shows exactly what changed.

### `data/` — Edge cases are first-class citizens
Each dataset has 6-8 edge cases: borderline amounts (flight just under insurance threshold), refurbished electronics, OTC medicine without chronic condition flag. These test whether the model understands the *rule*, not just the common pattern. Edge cases are flagged in results so you can track model performance on hard cases separately from easy ones.

---

## Model Routing

The eval is **provider-agnostic**: all 3 tasks use whatever provider you pass via `--baseline-provider` and `--candidate-provider`. The default command compares:

- **Baseline**: Anthropic (Claude Haiku) for all tasks
- **Candidate**: Google (Gemini 2.5 Flash) for all tasks

This lets you answer "can we swap Haiku → Gemini everywhere?" in one run.

| Component | Model | Provider | Why |
|---|---|---|---|
| All generation tasks (baseline) | Claude Haiku | Anthropic | Default baseline — cheap, capable |
| All generation tasks (candidate) | Gemini 2.5 Flash | Google | Default candidate — 80% cheaper than Haiku, equivalent quality on eval |
| LLM-as-judge (deal copy + credit) | Claude Opus | Anthropic | Hardcoded — cheap judge, sufficient for structured 0-10 scoring |
| Semantic similarity | all-MiniLM-L6-v2 | Local | Free, no API call, no network latency |
| Format compliance | Python | None | Exact character counting — no LLM needed |
| Factual grounding | Python + regex | None | Number extraction — no LLM needed |

**Principle**: Use the cheapest tool that answers the question correctly. 3 of the 5 scorers make zero API calls.

---

## How to Run

### 1. Clone and install

```bash
git clone git@github.com:jyothiswaroopajinka/the-guard-jyothiswaroopajinka.git
cd the-guard-jyothiswaroopajinka
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys:
# ANTHROPIC_API_KEY  — console.anthropic.com
# OPENAI_API_KEY     — platform.openai.com
# GOOGLE_API_KEY     — aistudio.google.com (free, no card required)
```

### 3. Commands

```bash
# Quick smoke test (5 cases per task, no judge — ~2 min, ~$0.01)
python run_eval.py --quick

# Full eval: compare Claude Haiku (baseline) vs Gemini Flash (candidate)
python run_eval.py --baseline-provider anthropic --candidate-provider google

# Demo regression detection: good prompt v1 vs degraded prompt v2
python run_eval.py --demo-regression
# Expected output: NO-GO — deal_copy regresses significantly

# Compare specific models
python run_eval.py \
  --baseline-provider anthropic \
  --candidate-provider openai \
  --baseline-prompt v1.txt \
  --candidate-prompt v1.txt

# View history of all past runs
python run_eval.py --history

# Diff two prompt versions
python run_eval.py --diff deal_copy v1.txt v2.txt

# Analyze test case diversity
python analyze_coverage.py
```

### Environment variables (optional overrides)

| Variable | Default | Effect |
|---|---|---|
| `SIGNIFICANCE_THRESHOLD` | 0.05 | p-value cutoff for statistical significance |
| `REGRESSION_THRESHOLD` | 0.05 | Minimum score drop to call it a regression |
| `PRACTICAL_THRESHOLD` | 0.10 | Block if drop exceeds this regardless of p-value |

---

## Eval Results (Actual Numbers)

**Run: Claude Haiku (baseline) vs Gemini 2.5 Flash (candidate) — 30 cases per task**

| Task | Scorer | Baseline | Candidate | Δ | p-value | Decision |
|---|---|---|---|---|---|---|
| Deal Copy | composite | 0.8285 | 0.8511 | +0.0226 | 0.473 | NO_CHANGE |
| Insurance | intent_score | 0.9720 | 0.9965 | +0.0245 | 0.476 | NO_CHANGE |
| Insurance | mcnemar | 29/30 correct | 30/30 correct | +1 case | 1.000 | NO_CHANGE |
| Credit | composite | 0.8489 | 0.8623 | +0.0134 | 0.502 | NO_CHANGE |

**Gate: GO — No regressions detected. Gemini Flash is safe to replace Haiku (and is 80% cheaper).**

**Run: v1 prompt vs v2 degraded prompt — regression demo**

| Task | Scorer | Baseline | Candidate | Δ | Decision |
|---|---|---|---|---|---|
| Deal Copy | composite | 0.83 | ~0.55 | -0.28 | REGRESSED |

**Gate: NO-GO — deal_copy dropped 28%, exceeding the 10% practical threshold. PR blocked.**

Raw eval output files (JSON) are in `eval_results/`. Each file contains per-case scores, p-values, cost, and latency.

---

## Cost Data

**Per full eval run (30 cases per task, both baseline and candidate):**

| Component | Calls | Model | Cost |
|---|---|---|---|
| Deal copy generation × 2 runs | 60 | Claude Haiku | ~$0.004 |
| Deal copy judge × 2 runs | 60 | Claude Haiku | ~$0.006 |
| Insurance classification × 2 runs | 60 | Gemini Flash | ~$0.001 |
| Credit generation × 2 runs | 60 | Claude Haiku | ~$0.006 |
| Credit judge × 2 runs | 60 | Claude Haiku | ~$0.008 |
| Semantic similarity | 120 | Local | $0.000 |
| Format / grounding scoring | 120 | Python | $0.000 |
| **Total per full run** | | | **~$0.05** |

**Quick mode (5 cases, no judge — used in CI):** ~$0.01 per run.

**One generation call cost:** ~$0.0003 (Haiku), ~$0.00005 (Gemini Flash).

**Total development cost estimate:** ~$10 across all runs during development.

---

## What Broke First

**Bug 1: Gemini 2.5 Flash returning 6-token truncated JSON**

Insurance classification was failing with `<parse_error>` on 93% of Gemini cases. The raw output was `'{\n  "label": "travel'` — truncated mid-sentence. I assumed it was a prompt format issue and spent time tweaking the prompt. Then I added `repr()` logging of the raw output and saw the token count: 6 output tokens for a response that should be 50-60 tokens.

Root cause: Gemini 2.5 Flash uses internal "thinking tokens" that are deducted from `max_output_tokens`. With `max_tokens=200`, thinking consumed ~194 tokens, leaving only 6 for actual output.

Fix: `thinking_config=genai_types.ThinkingConfig(thinking_budget=0)` in `call_gemini()` disables thinking for simple classification tasks. Also increased `max_tokens` to 512 as a safety buffer.

**Bug 2: Judge prompts broke when moved to text files**

The LLM judge prompts were originally hardcoded strings. When I moved them to `.txt` files for cleanliness, the judge started returning garbled output. The text files contained JSON examples with `{` and `}` characters, which Python's `.format()` tried to interpret as variable placeholders.

Fix: Switched from `.format()` to explicit `.replace("{variable}", value)` calls. Only substitutes the named variables, ignores all other braces.

**Bug 3: Gemini looked like a bad model but it was actually an API error**

After fixing the truncation issue, re-ran the eval and insurance showed Gemini at 0% accuracy — 28 out of 30 cases predicted `<parse_error>`. My first assumption was that Gemini genuinely couldn't do classification. I spent time comparing prompt formats between Anthropic and Google, thinking the instruction style was wrong for Gemini.

Then I checked the raw eval JSON and saw that all 28 failures had `"errors": ["Classification failed: ..."]` — meaning the API call itself had failed silently, the model never ran, and the score defaulted to 0. The dashboard was showing these zeros as if they were real predictions.

Root cause: The API errors were being caught and swallowed — the case scored 0 and the eval continued, but nothing in the terminal output made it obvious that 28 calls had failed. The zeros looked like wrong predictions, not missing predictions.

Fix: Added a prominent API failure panel to the dashboard that shows up front when any cases failed due to API errors — with the error type (billing, rate limit, auth, etc.) clearly labeled so you don't mistake infrastructure failures for model quality issues.

**Bug 4: History tracking silently never worked**

The `--history` flag always showed an empty table despite multiple eval runs. The `update_history()` function was defined and worked correctly when called — but was never called anywhere. It was wired up in my mental model but not in the code.

Fix: Added the call at the end of `save_run()`. History now persists correctly across runs.

---

## What I Would Change with 2 More Weeks

1. **100+ test cases per task** — 30 cases gives reasonable power but high p-values on small differences. At 100 cases, a 5% improvement becomes statistically confirmable.

2. **Shadow testing mode** — Run baseline and candidate in parallel on every incoming real request, log both outputs, compute score drift on production traffic. Catches distribution shift that hand-crafted test cases miss.

3. **Telugu/Hindi eval cases** — Current tasks are English only. GrabOn operates in multiple languages; 10-15 localized cases per task would test localization quality.

4. **Historical trend dashboard** — A Streamlit app that plots accuracy per model per task over time, with git commits annotated. Currently `--history` is a terminal table; a visual makes regressions obvious at a glance.

5. **Hard budget kill switch** — If a run is consuming more tokens than expected (runaway retries), halt mid-run rather than letting costs accumulate. Currently the eval runs to completion regardless of cost.

---

## CI/CD Integration

The GitHub Action at `.github/workflows/eval-guard.yml` triggers on every PR touching `prompts/**`, `src/providers.py`, `data/**`, or task/scorer code.

**What it does:**
1. Runs eval in `--quick` mode (5 cases, no judge) — ~$0.01, keeps CI fast
2. Exits with code 1 on NO-GO → GitHub blocks the PR from merging
3. Uploads eval result JSON as a build artifact (retained 30 days)
4. Posts a GO/NO-GO comment on the PR with the gate summary

**Setup:** `Settings → Secrets → Actions` → Add `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`

---

## Failure Recovery

`src/providers.py` distinguishes error types before retrying:

- **Permanent failures** (bad API key, billing, 401): fail immediately, no retry. Retrying would waste time and money.
- **Transient failures** (rate limit, 429, timeout, 500): retry with exponential backoff (1s, 2s, 4s).
- **After max retries**: fall back to the next provider in the chain (Anthropic → Google).

If an individual case fails, the error is logged in `result.errors` and the case scores 0. The eval continues to the next case — a partial outage shows up as lower accuracy, not a crashed run.

---

## Prompt Versioning

Every eval run calls `versioning.register_all_prompts()` first. This scans all `.txt` files in `prompts/`, SHA-256 hashes each one, and records hash + git commit + timestamp in `prompt_versions.json`. If someone edits a prompt without renaming it, the hash changes and the system detects it. On NO-GO with a prompt change, the runner automatically diffs the two prompt files.

```bash
python run_eval.py --diff deal_copy v1.txt v2.txt
```
