# Multi-Agent Code Review System

A code review system that compares a single generalist LLM reviewer against a team of specialized agents (Style, Logic, Test Coverage, Critic), evaluated against real human review comments on merged pull requests from [`fastapi/typer`](https://github.com/fastapi/typer).

**Result:** the best specialized configuration scored **4.27–4.50/5** on an LLM-judge evaluation against human ground truth, versus **4.07/5** for a single generalist model — while also surfacing regression risks and edge cases the single-agent baseline missed entirely.

---

## Try it live

streamlit run streamlit_app.py

## The problem

A single LLM given a full diff and asked to "review this code" tends to produce a review that touches lightly on everything — naming, tests, logic — without going deep on any one of them. On a real bug-fix PR ([#1821](https://github.com/fastapi/typer/pull/1821)), the single-agent baseline explicitly wrote *"I'm not entirely sure why this change was made"* about the exact line that fixed the bug, while the human reviewer identified the root cause in one sentence.

This project tests whether splitting that single agent into specialists — each with a narrow prompt and a single responsibility — closes that gap, and whether the choice of *model per role* matters as much as the *specialization* itself.

## Architecture

```
                    ┌──────────────┐
   diff  ─────────► │ Style Agent  │──┐
                    └──────────────┘  │
                    ┌──────────────┐  │      ┌──────────────┐
                ┌──►│ Logic Agent  │──┼─────►│ Critic Agent │──► final review
                │   └──────────────┘  │      └──────────────┘
                │   ┌──────────────┐  │
                └──►│ Test Agent   │──┘
                    └──────────────┘
```

Orchestrated as a [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`: each agent is a node, a shared `ReviewState` carries the diff and accumulating reviews, and the Critic synthesizes the three specialist outputs into one final review. Execution is sequential rather than parallel fan-out — see [Lessons learned](#lessons-learned) for why.

## Three configurations tested

| Config | Setup | Purpose |
|---|---|---|
| **A** | Single generalist agent, one prompt | Baseline |
| **B** | 4 specialized agents, same model (`gpt-oss-120b`) for every role | Isolates the effect of prompt specialization alone |
| **C** | 4 specialized agents, model chosen per role (see below) | Tests whether matching model strengths to role improves results further |

**Config C model assignment**, based on empirical testing in Config A/B:

| Role | Model | Why |
|---|---|---|
| Logic | `qwen/qwen3.6-27b` | Chain-of-thought reasoning consistently caught causal bug→fix links that non-reasoning models missed |
| Style, Test | `llama-3.1-8b-instant` | Narrow-scope tasks; a small fast model is sufficient |
| Critic | `openai/gpt-oss-120b` | Strongest synthesis and independent edge-case detection in testing |

## Methodology

15 merged PRs were pulled from `fastapi/typer` via the GitHub API, filtered to exclude bot/CI/dependency-bump noise. PRs were split into two evaluation categories:

- **Category A** (PRs with real inline review comments from maintainers): the AI review is scored against what a human reviewer actually said.
- **Category B** (PRs merged without review comments — common on this repo): the AI review is scored against the PR's own title and description.

Scoring uses an LLM-judge (`llama-3.1-8b-instant`, chosen because it wasn't otherwise under quota pressure) on a 1–5 scale, with a one-sentence justification per PR. Full per-PR results and reasoning are in `data/evaluation_report.md` and `data/evaluation_report_config_c.md`.

## Results

| | Config A (single-agent) | Config B (same model, 4 roles) | Config C (model per role) |
|---|---|---|---|
| Avg. LLM-judge score | not formally scored | **4.07/5** | **4.27/5** (full 15/15) |
| Sample size | — | 15/15 PRs | 15/15 PRs |

On PR #1821, Config B's Critic independently flagged a regression risk (loss of the ability to distinguish "no values supplied" from "explicit empty list") that neither the single-agent baseline nor the human reviewer mentioned — and proposed a concrete API fix (`--allow-empty` sentinel).

### Where the system struggled

- **PR #1773** (CI/workflow diff): scored 1/5. The specialized agents drilled into permission/security details of the GitHub Actions changes and lost sight of the PR's actual one-line purpose ("bring branch up to date with master"). Dense, low-level diffs are a known weak spot.
- The LLM-judge itself sometimes penalizes reviews that are *more* thorough than the PR title implies — a limitation of automated judging, not necessarily of the reviews themselves.

## Guardrails

Built after hitting each failure mode in practice, not preemptively:

- **Shared rate limiter** (sliding 60s token window) across all agent calls, after naive parallel execution repeatedly blew through Groq's per-minute quota.
- **Anti-hallucination guardrail on the Critic**: if an upstream agent's output is empty or truncated, the Critic returns `[INCOMPLETE REVIEW]` naming which agent failed, instead of fabricating a plausible-sounding synthesis. This was caught in testing — an early version of the Critic once inverted the actual meaning of a code change when given a truncated input.
- **Reasoning/answer separation**: reasoning models (Qwen) emit a `<think>...</think>` block before their answer; this is stripped before the text reaches the Critic, so raw chain-of-thought never pollutes downstream synthesis.
- **Confidence flagging**: reviews with a high density of hedging language ("might", "unclear", "possibly") are prepended with a visible low-confidence banner rather than presented as flat assertions.
- **Persistent file logging**: every run writes a timestamped log to `logs/`, not just the console.
- **Resume support**: both the multi-agent and LangGraph runners skip PRs that already have a valid saved result, so a run interrupted by a rate limit or network error can be resumed without re-consuming quota on already-completed work.

## Lessons learned

- **Token budgets need generous headroom, not minimums.** `openai/gpt-oss-120b` is non-deterministic enough that the same prompt produced a complete response on one run and a truncated one on the next, at identical `max_tokens`.
- **Parallel agent execution is a liability on a shared rate-limited quota.** Firing Style/Logic/Test concurrently creates request bursts that exceed Groq's tokens-per-minute cap almost immediately. Sequential execution, paced by a shared rate limiter, was more reliable — even though it's slower.
- **A reasoning model needs room to finish reasoning.** Qwen initially appeared weaker than GPT-OSS on causal analysis; the actual issue was `max_tokens` cutting off its `<think>` block before it reached a conclusion. Once given sufficient budget, it identified root causes other models missed.
- **Free-tier daily quotas (not just per-minute) are a real constraint.** Runs were repeatedly interrupted by Groq's 200k-tokens/day cap on `gpt-oss-120b`. This shaped the guardrail/resume design as much as any architectural decision.

## Project structure

```
src/
├── fetch_prs.py              # GitHub data collection, noise filtering, A/B classification
├── baseline_multi_model.py   # Config A: single-agent, multi-model comparison
├── compare_models.py         # Side-by-side comparison viewer (human vs. model reviews)
├── multi_agent_baseline.py   # Config B/C: sequential 4-agent orchestration (pre-LangGraph)
├── graph.py                  # Config B/C: LangGraph StateGraph orchestration (current)
├── evaluate.py                # LLM-judge evaluation against ground truth
├── logging_config.py         # Persistent file + console logging
└── agents/
    ├── prompts.py             # The 4 specialized prompts
    ├── agents.py               # Agent callers, rate limiting, guardrails
    ├── model_configs.py        # Named model configs (B, C)
    ├── guardrails.py            # Confidence-flagging heuristic
    └── rate_limiter.py          # Shared sliding-window token rate limiter
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

`.env`:
```
GITHUB_TOKEN=your_github_token
GROQ_API_KEY=your_groq_key
AGENT_CONFIG=B               # or C
```

## Usage

```bash
# 1. Collect PR data
python src/fetch_prs.py

# 2. Run the single-agent baseline
python src/baseline_multi_model.py

# 3. Run the multi-agent system (LangGraph)
python src/graph.py

# 4. Evaluate against ground truth
python src/evaluate.py
```

Each script supports `--pr <number>` to run/re-run a single PR without reprocessing the full dataset.

## Limitations

- Evaluated on a single repository (`fastapi/typer`); results may not generalize to other codebases or languages.
- The LLM-judge scoring is itself an LLM, with its own biases (see [Where the system struggled](#where-the-system-struggled)).
- Free-tier API quotas constrained sample size and required manual resume/retry workflows — not representative of a production deployment.
- Config C's model assignment was chosen based on qualitative observation during Config A/B testing, not a systematic per-role model search.

## Possible next steps

- Systematic model search per role instead of hand-picked assignment
- Test on a second, different-language repository to check generalization
- Package as a GitHub App with webhook triggers (see architecture notes in project history)
- Streamlit demo: paste a diff, watch the 4 agents respond live