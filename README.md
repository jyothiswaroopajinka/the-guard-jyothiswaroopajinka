# The Guard — GrabOn AI Eval Pipeline

**Assignment 03 | GrabOn AI Labs Agentic AI Engineer Challenge**

A production eval framework that detects quality regressions in GrabOn's AI-generated outputs before they ship.

---

## What This Is

GrabOn's AI produces three types of outputs that real people and partners read:

1. **Deal copy** — sent to 40M subscribers across email, WhatsApp, push, and Glance
2. **Insurance intent classification** — determines which micro-policy a user sees at checkout
3. **Credit narratives** — go to Poonawalla Fincorp's compliance team; hallucinated stats are a regulatory risk

Every time someone changes a prompt, swaps a model, or updates a tool endpoint, quality can silently degrade. The Guard catches this **before deployment** using:

- Statistical tests (paired bootstrap + McNemar's) that distinguish real regressions from noise
- 5 scoring functions per task (factual grounding, intent match, format compliance, LLM-as-judge, semantic similarity)
- A GO/NO-GO/INCONCLUSIVE gate that CI/CD checks on every PR
- Prompt versioning integrated with git so you can trace regressions to a specific commit

---

## Architecture

```
run_eval.py                     ← Entry point (CLI)
│
├── src/versioning.py           ← Register + hash all prompt files before running
│
├── src/tasks/
│   ├── deal_copy_eval.py       ← Task 1: Generate deal copy, score it
│   ├── insurance_eval.py       ← Task 2: Classify insurance intent, score it
│   └── credit_eval.py          ← Task 3: Generate credit narrative, score it
│
├── src/scorers/
│   ├── factual_grounding.py    ← Scorer (a): Does text cite real numbers? (regex, no LLM)
│   ├── intent_match.py         ← Scorer (b): Does prediction match label? (deterministic)
│   ├── format_compliance.py    ← Scorer (c): Does text fit channel limits? (deterministic)
│   ├── llm_judge.py            ← Scorer (d): Claude Opus grades another model's output
│   └── semantic_similarity.py  ← Scorer (e): Cosine similarity via sentence-transformers (local)
│
├── src/statistical.py          ← Paired bootstrap test + McNemar's test + p-values
├── src/gate.py                 ← GO / NO-GO / INCONCLUSIVE decision
├── src/providers.py            ← Raw API clients: Anthropic, OpenAI, Google Gemini
├── src/dashboard.py            ← Terminal report + JSON result persistence
│
├── prompts/
│   ├── deal_copy/v1.txt        ← Good prompt (baseline)
│   ├── deal_copy/v2.txt        ← Degraded prompt (demo regression)
│   ├── insurance/v1.txt
│   └── credit/v1.txt
│
├── data/
│   ├── deal_copy_cases.json    ← 30 test cases with source data + reference outputs
│   ├── insurance_cases.json    ← 30 test cases with ground-truth labels
│   └── credit_cases.json       ← 30 test cases with user personas + reference narratives
│
└── .github/workflows/
    └── eval-guard.yml          ← GitHub Action: runs eval on every PR, blocks on NO-GO
```

**Data flow for one eval run:**
```
[Load 30 cases] → [Call LLM to generate output] → [Score with 4-5 scorers]
      ↓
[Repeat for baseline model/prompt AND candidate model/prompt]
      ↓
[Paired bootstrap test per scorer per task]
      ↓
[GO / NO-GO / INCONCLUSIVE gate]
      ↓
[Save JSON + print terminal report + exit with code 0/1/2]
```

---

## Model Routing Rationale

| Task | Model Used | Why |
|---|---|---|
| Insurance classification | GPT-4o-mini | Simple classification, low latency needed, $0.15/1M tokens |
| Deal copy generation | Claude Sonnet | Best instruction-following, respects format constraints |
| Credit narrative generation | Claude Sonnet | Needs factual precision, strong instruction following |
| LLM-as-judge | Claude Opus | Most capable evaluator — we only use it to grade, not generate |
| Semantic similarity | sentence-transformers (local) | Free, fast, no API call — perfect for embedding comparison |
| Format compliance | Deterministic Python | No LLM needed — character counting is exact |
| Factual grounding | Deterministic Python (regex) | No LLM needed — number extraction is exact |

**The principle**: Use the cheapest model that can do the job correctly. Never use Opus to check if a number is in a string — that's what regex is for.

---

## Quick Start

### 1. Setup

```bash
git clone <your-repo>
cd the-guard-jyothiswaroopa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

**API keys needed** (all have free tiers):
- `ANTHROPIC_API_KEY` — console.anthropic.com
- `OPENAI_API_KEY` — platform.openai.com (GPT-4o-mini costs ~$0 for this)
- `GOOGLE_API_KEY` — aistudio.google.com (free, no card required)

### 2. Run a quick eval (5 cases per task, no judge)

```bash
python run_eval.py --quick
```

### 3. Run full eval (all 30 cases, all scorers)

```bash
python run_eval.py
```

### 4. Demo the regression detection (compare good prompt vs bad prompt)

```bash
python run_eval.py --demo-regression
# Should produce NO-GO because v2 prompt allows hallucination
```

### 5. Compare two models directly

```bash
python run_eval.py \
  --baseline-provider anthropic --baseline-model claude-sonnet-4-6 --baseline-prompt v1.txt \
  --candidate-provider openai --candidate-model gpt-4o-mini --candidate-prompt v1.txt
```

### 6. Show history of past runs

```bash
python run_eval.py --history
```

### 7. Diff two prompt versions

```bash
python run_eval.py --diff deal_copy v1.txt v2.txt
```

---

## Eval Tasks

### Task 1: Deal Copy Quality (30 cases)

Tests deal copy generation across 4 channels. Each case has source data (merchant, discount %, coupon code, expiry, min order) and a reference output.

**Scorers used:**
- Format compliance (deterministic): Does WhatsApp copy fit 160 chars? Push fit 50?
- Factual grounding (regex): Are all numbers traceable to source data?
- LLM-as-judge (Claude Opus): Persuasiveness, factual accuracy, channel fit (0-10 each)
- Semantic similarity (sentence-transformers): Cosine distance from reference

### Task 2: Insurance Intent Classification (30 cases)

Tests classification of deal objects into 5 categories: travel_insurance, device_protection, health_insurance, loan_protection, no_insurance.

**Scorers used:**
- Intent match (deterministic): Predicted label vs ground truth
- Confidence calibration: Penalizes overconfident wrong predictions
- McNemar's test for statistical comparison of binary correct/incorrect

### Task 3: Credit Narrative Faithfulness (30 cases)

Tests narrative generation for mock user personas. Each narrative goes to a bank's compliance team — any hallucinated stat is a regulatory risk.

**Scorers used:**
- Factual grounding (regex, weighted 45%): Most critical — are all numbers real?
- LLM-as-judge (Claude Opus, weighted 35%): Factual accuracy + professional tone + completeness
- Semantic similarity (weighted 20%): How close is it to the reference narrative?

---

## Statistical Testing

The eval uses **paired bootstrap testing** — the gold standard for comparing NLP systems.

**Why paired?** Each test case has a difficulty level. Pairing removes per-case variance — we only measure the difference the model/prompt change makes.

**Why bootstrap?** Doesn't assume Gaussian distribution of scores. Works for small n (30 cases).

```python
# The test in plain English:
# 1. Compute score difference for each case: diff[i] = candidate[i] - baseline[i]
# 2. Resample diffs with replacement 10,000 times
# 3. p-value = fraction of samples where |mean_diff| >= |observed_mean_diff|
# 4. If p < 0.05 AND absolute drop > 0.05: call it a regression
```

**McNemar's test** is used for classification (insurance task) because the outcome is binary (correct/incorrect), not continuous.

---

## GO/NO-GO Gate

```
REGRESSION in ANY task + statistically significant (p < 0.05)
+ absolute drop > 0.05 → NO-GO (exit code 1, PR blocked)

All tasks NO_CHANGE or IMPROVED → GO (exit code 0, PR passes)

Some improved, some regressed, none significant → INCONCLUSIVE (exit code 2, warning)
```

The gate is conservative by design: any regression in any task blocks deployment. This matches GrabOn's risk profile — a credit narrative sent to Poonawalla with hallucinated numbers is worse than blocking a deploy.

---

## Prompt Versioning

Each prompt file is SHA-256 hashed. On every eval run, all prompts in `prompts/` are registered in `prompt_versions.json` with their hash + git commit + timestamp.

When a regression is detected and `--baseline-prompt != --candidate-prompt`, the runner automatically prints the unified diff showing exactly what changed.

```bash
# Manually diff any two prompt versions:
python run_eval.py --diff deal_copy v1.txt v2.txt
```

---

## CI/CD Integration

The GitHub Action at `.github/workflows/eval-guard.yml` runs on every PR that touches:
- `prompts/**` (prompt changes)
- `src/providers.py` (model config changes)
- `data/**` (test data changes)
- Task or scorer code

**The action:**
1. Runs eval in quick mode (5 cases per task, no judge) to save API cost in CI
2. Exits with code 1 (fail) on NO-GO → PR is blocked
3. Uploads eval results as a GitHub artifact (retained 30 days)
4. Posts a comment on the PR with the GO/NO-GO verdict

**To add secrets to your repo:**
`Settings > Secrets > Actions` → Add `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`

---

## Eval Results

Results are saved to `eval_results/run_{timestamp}.json` (excluded from git via .gitignore).

Each result file contains:
- Raw scores per case per scorer
- Statistical comparison results with p-values and CIs
- Gate decision and summary
- Total cost in USD

The history dashboard (`python run_eval.py --history`) shows the last 20 runs with score deltas — this is how you answer "When did quality drop, and which commit caused it?"

---

## Cost Data

| Operation | Model | Tokens | Est. Cost |
|---|---|---|---|
| Deal copy generation (30 cases) | Claude Sonnet | ~60K tokens | ~$0.18 |
| Insurance classification (30 cases) | GPT-4o-mini | ~30K tokens | ~$0.005 |
| Credit narrative generation (30 cases) | Claude Sonnet | ~90K tokens | ~$0.27 |
| LLM-as-judge deal copy (30 cases) | Claude Opus | ~90K tokens | ~$2.25 |
| LLM-as-judge credit (30 cases) | Claude Opus | ~90K tokens | ~$2.25 |
| **Full eval run (baseline + candidate)** | — | — | **~$10** |
| **Quick eval run (5 cases, no judge)** | — | — | **~$0.05** |
| Semantic similarity | sentence-transformers | local | **$0** |
| Format compliance | deterministic | no LLM | **$0** |

Opus is expensive — but it's only used as a judge (30 calls per task), not for generation. In CI, `--quick` mode skips the judge entirely, costing ~$0.05 per run.

**My total development cost estimate:** ~$15 across all test runs.

---

## Tradeoffs and Design Decisions

**Why raw API calls instead of LangChain?**
Every HTTP request is visible. During the deep-dive, I can explain exactly what happens between a function call and the API response. No hidden retry logic, no unexpected context injection.

**Why sentence-transformers for embeddings instead of OpenAI embeddings?**
Free, local, fast. The quality (all-MiniLM-L6-v2) is sufficient for catching semantic drift. OpenAI embeddings cost extra per call and add a network hop.

**Why conservative gate (block on any regression)?**
GrabOn's outputs go to 40M subscribers and a bank's compliance team. False negatives (missing a regression) cost more than false positives (blocking a good deploy). Engineers can always override with a manual review.

**Why Wilson interval for per-case CI?**
It handles edge cases (0/1 proportions) better than normal approximation. The Wald interval breaks at 0% or 100% accuracy.

---

## What Broke First

The trickiest bug: the factual grounding scorer initially failed to handle Indian number formatting (₹4,80,000 with the Indian comma convention). The regex `[\d,]+` matched ₹4, separately from 80,000. Fixed by treating the full token including Indian-style commas as a unit and normalizing before comparison.

---

## What I Would Change With 2 More Weeks

1. **Larger test dataset (100+ cases)** — 30 cases gives decent statistical power but more is better, especially for edge cases in insurance classification.
2. **Historical trend dashboard** — A Streamlit app that plots accuracy over time per model per task, making it easy to visually spot when a regression started.
3. **Telugu/Hindi test cases** — The credit and insurance tasks only test English. Adding 10 Telugu cases would test localization quality, which is a real GrabOn concern.
4. **Shadow testing mode** — Run both models on every incoming request (not just eval cases) and log the comparison. After 500+ real requests, auto-update the routing recommendation.
5. **Tighter confidence intervals** — Current CI uses empirical rules. Proper Bayesian credible intervals via MCMC would be more rigorous for small n.

---

## Live vs Mocked Calls

All API calls are live in this implementation. No mocking.

- **Anthropic** (Claude Sonnet + Claude Opus): Live
- **OpenAI** (GPT-4o-mini): Live
- **Google Gemini**: Live (used when `--baseline-provider google`)
- **sentence-transformers**: Local (not a network call)

If you hit rate limits during testing, the providers.py `call_model()` function retries with exponential backoff (max 3 attempts). After that it raises an error logged in the result's `errors` field — the eval continues on the next case.
