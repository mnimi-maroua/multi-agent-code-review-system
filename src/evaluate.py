"""
Step 5: Evaluate multi-agent reviews against ground truth.

Two evaluation paths, matching the hybrid methodology defined in Step 1:
    - Category A (PRs with real human review comments): compare the
      Critic's final_review against what a human reviewer actually said.
    - Category B (PRs with no review comments): compare final_review
      against the PR's own title/description -- does the review capture
      the problem the PR claims to solve?

This script uses a lightweight, free, LLM-judge approach: a single Groq
call per PR that scores the match and explains why, reusing the same
rate limiter as the agents so it doesn't blow through the shared quota.

Usage:
    python evaluate.py
    python evaluate.py --input data/multi_agent_results.json

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from agents.rate_limiter import RateLimiter, estimate_tokens

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

# Deliberately a lighter model: this is a judging task (score + short
# justification), not deep reasoning, and it's a model with quota that
# hasn't been heavily used by the agents/baseline runs.
JUDGE_MODEL = "llama-3.1-8b-instant"
JUDGE_MAX_TOKENS = 400
TOKENS_PER_MINUTE = 8000  # verify current limit at https://console.groq.com/docs/rate-limits

_rate_limiter = RateLimiter(tokens_per_minute=TOKENS_PER_MINUTE)


JUDGE_PROMPT_CATEGORY_A = """You are evaluating an AI-generated code review against what a real
human reviewer actually said on the same pull request.

Human reviewer comment(s):
{human_comments}

AI-generated review:
{ai_review}

Score how well the AI review captures the SAME key point(s) the human raised,
on a 1-5 scale:
    5 = AI review identifies the same core issue the human pointed out
    3 = AI review is in the right area but misses the specific point
    1 = AI review does not address what the human raised at all

Respond in exactly this format, nothing else:
SCORE: <1-5>
REASON: <one sentence>
"""

JUDGE_PROMPT_CATEGORY_B = """You are evaluating whether an AI-generated code review correctly
identifies the problem a pull request claims to solve, based on its title
and description (no human review comments are available for this PR).

PR title: {pr_title}
PR description: {pr_body}

AI-generated review:
{ai_review}

Score how well the AI review identifies the actual problem/change described
in the title and description, on a 1-5 scale:
    5 = AI review clearly identifies and engages with the PR's actual purpose
    3 = AI review is generic but not wrong / partially on-topic
    1 = AI review misunderstands or ignores what the PR is actually about

Respond in exactly this format, nothing else:
SCORE: <1-5>
REASON: <one sentence>
"""


@dataclass
class EvalResult:
    number: int
    title: str
    category: str
    score: int | None
    reason: str
    judge_error: str | None = None


def call_judge(prompt: str, max_retries: int = 3) -> str:
    estimated = estimate_tokens(prompt) + JUDGE_MAX_TOKENS
    _rate_limiter.reserve(estimated)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=JUDGE_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            wait = 5 * attempt
            logger.warning("Judge call failed (attempt %d/%d): %s -- retrying in %ds",
                            attempt, max_retries, last_error, wait)
            time.sleep(wait)
    raise RuntimeError(f"Judge failed after {max_retries} retries: {last_error}")


def parse_judge_output(raw: str) -> tuple[int | None, str]:
    """Parse the SCORE:/REASON: format. Returns (score or None, reason)."""
    score = None
    reason = raw.strip()
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            digits = "".join(c for c in line.split(":", 1)[1] if c.isdigit())
            if digits:
                score = int(digits[0])
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return score, reason


def evaluate_pr(pr: dict) -> EvalResult:
    final_review = pr.get("final_review", "")
    if not final_review or final_review.startswith("[ERROR]") or final_review.startswith("[INCOMPLETE"):
        return EvalResult(
            number=pr["number"], title=pr["title"], category=pr["category"],
            score=None, reason="No usable final_review to evaluate.",
            judge_error="missing_review",
        )

    if pr["category"] == "A" and pr.get("human_review_comments"):
        human_text = "\n".join(
            f"- [{c['author']}] {c['body']}" for c in pr["human_review_comments"]
        )
        prompt = JUDGE_PROMPT_CATEGORY_A.format(human_comments=human_text, ai_review=final_review)
    else:
        prompt = JUDGE_PROMPT_CATEGORY_B.format(
            pr_title=pr["title"],
            pr_body=(pr.get("body") or "(no description provided)")[:1500],
            ai_review=final_review,
        )

    try:
        raw = call_judge(prompt)
        score, reason = parse_judge_output(raw)
        return EvalResult(number=pr["number"], title=pr["title"], category=pr["category"],
                           score=score, reason=reason)
    except Exception as exc:  # noqa: BLE001
        return EvalResult(number=pr["number"], title=pr["title"], category=pr["category"],
                           score=None, reason="Judge call failed.", judge_error=str(exc))


def build_report(results: list[EvalResult]) -> str:
    scored = [r for r in results if r.score is not None]
    lines = ["# Evaluation Report\n"]

    if scored:
        avg = sum(r.score for r in scored) / len(scored)
        lines.append(f"**Average score:** {avg:.2f}/5 across {len(scored)}/{len(results)} evaluated PRs\n")

        for cat in ("A", "B"):
            cat_scored = [r for r in scored if r.category == cat]
            if cat_scored:
                cat_avg = sum(r.score for r in cat_scored) / len(cat_scored)
                label = "vs. human comments" if cat == "A" else "vs. PR description"
                lines.append(f"- Category {cat} ({label}): {cat_avg:.2f}/5 over {len(cat_scored)} PRs")
        lines.append("")

    lines.append("| PR | Category | Score | Reason |")
    lines.append("|---|---|---|---|")
    for r in results:
        score_display = str(r.score) if r.score is not None else "N/A"
        lines.append(f"| #{r.number} {r.title[:50]} | {r.category} | {score_display} | {r.reason} |")

    failed = [r for r in results if r.judge_error]
    if failed:
        lines.append("\n## PRs that could not be evaluated\n")
        for r in failed:
            lines.append(f"- #{r.number}: {r.judge_error}")

    return "\n".join(lines)


def run_evaluation(input_path: Path, output_json: Path, output_md: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        prs = json.load(f)

    logger.info("Evaluating %d PRs from %s", len(prs), input_path)

    results: list[EvalResult] = []
    for i, pr in enumerate(prs, start=1):
        logger.info("[%d/%d] PR #%s - %s", i, len(prs), pr["number"], pr["title"])
        result = evaluate_pr(pr)
        results.append(result)
        logger.info("  -> score=%s reason=%s", result.score, result.reason)
        time.sleep(1)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    report = build_report(results)
    output_md.write_text(report, encoding="utf-8")

    logger.info("Done. Results saved to %s and %s", output_json, output_md)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-agent reviews against ground truth.")
    parser.add_argument("--input", type=Path, default=Path("data/multi_agent_results.json"))
    parser.add_argument("--output-json", type=Path, default=Path("data/evaluation_results.json"))
    parser.add_argument("--output-md", type=Path, default=Path("data/evaluation_report.md"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.input, args.output_json, args.output_md)