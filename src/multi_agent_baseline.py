"""
Step 3: Multi-agent baseline (Config B -- same model, specialized prompts).

Runs Style, Logic, and Test Coverage agents in parallel on each PR diff,
then feeds their outputs to a Critic agent that merges them into one
final review. All four agents use the same model (GPT-OSS 120B), so
results are directly comparable to the single-agent baseline in
baseline_multi_model.py: any quality difference comes from prompt
specialization, not from a different model.

Usage:
    python multi_agent_baseline.py
    python multi_agent_baseline.py --pr 1821

Environment (.env):
    GROQ_API_KEY=...
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from agents.agents import run_style_agent, run_logic_agent, run_test_agent, run_critic_agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multi_agent_baseline")

MAX_DIFF_CHARS = 8000
REQUEST_DELAY_SECONDS = 1.0


def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run fetch_prs.py first.")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def review_with_agents(diff_text: str) -> dict:
    """
    Run Style, Logic, and Test sequentially, then Critic on their outputs.

    Sequential rather than parallel on purpose: the bottleneck here is
    Groq's shared tokens-per-minute quota, not compute time. Firing 3
    calls at once just creates a burst that blows through the quota
    immediately (see rate_limiter.py, which every call already goes
    through) without actually finishing any faster.
    """
    style_review = run_style_agent(diff_text)
    logic_review = run_logic_agent(diff_text)
    test_review = run_test_agent(diff_text)
    final_review = run_critic_agent(style_review, logic_review, test_review)

    return {
        "style_review": style_review,
        "logic_review": logic_review,
        "test_review": test_review,
        "final_review": final_review,
    }


def pr_has_valid_result(entry: dict) -> bool:
    """A PR result is usable if it has a final_review and no top-level error."""
    if "error" in entry:
        return False
    final_review = entry.get("final_review", "")
    return bool(final_review) and not final_review.startswith("[ERROR]")


def run_multi_agent_baseline(dataset_path: Path, output_path: Path) -> None:
    dataset = load_dataset(dataset_path)
    logger.info("Loaded %d PRs from %s", len(dataset), dataset_path)

    # Resume support: don't re-burn quota on PRs that already succeeded in
    # a previous run (e.g. one that got cut off by a daily token limit).
    existing_results: dict[int, dict] = {}
    if output_path.exists():
        for entry in load_dataset(output_path):
            if pr_has_valid_result(entry):
                existing_results[entry["number"]] = entry
        if existing_results:
            logger.info("Resuming: %d PR(s) already have valid results and will be skipped",
                        len(existing_results))

    results: list[dict] = []
    for i, pr in enumerate(dataset, start=1):
        if pr["number"] in existing_results:
            logger.info("[%d/%d] PR #%s - already done, skipping", i, len(dataset), pr["number"])
            results.append(existing_results[pr["number"]])
            continue

        logger.info("[%d/%d] PR #%s - %s", i, len(dataset), pr["number"], pr["title"])
        diff_text = pr["diff"][:MAX_DIFF_CHARS]

        try:
            agent_outputs = review_with_agents(diff_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PR #%s failed: %s", pr["number"], exc)
            agent_outputs = {"error": str(exc)}

        results.append({
            "number": pr["number"],
            "title": pr["title"],
            "category": pr["category"],
            "human_review_comments": pr["review_comments"],
            **agent_outputs,
        })

        # Save incrementally after every PR, not just at the end -- if the
        # daily quota runs out mid-run, everything done so far is preserved.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(REQUEST_DELAY_SECONDS)

    succeeded = sum(1 for r in results if pr_has_valid_result(r))
    logger.info("Done. %d/%d PRs saved to %s (%d succeeded, %d failed)",
                len(results), len(dataset), output_path, succeeded, len(results) - succeeded)


def run_single_pr(dataset_path: Path, output_path: Path, pr_number: int) -> None:
    dataset = load_dataset(dataset_path)
    pr = next((p for p in dataset if p["number"] == pr_number), None)
    if pr is None:
        raise SystemExit(f"PR #{pr_number} not found in {dataset_path}")

    diff_text = pr["diff"][:MAX_DIFF_CHARS]
    logger.info("Running 4 agents on PR #%s - %s", pr["number"], pr["title"])
    agent_outputs = review_with_agents(diff_text)

    existing = load_dataset(output_path) if output_path.exists() else []
    existing = [r for r in existing if r["number"] != pr_number]
    existing.append({
        "number": pr["number"],
        "title": pr["title"],
        "category": pr["category"],
        "human_review_comments": pr["review_comments"],
        **agent_outputs,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    logger.info("Updated PR #%s in %s", pr_number, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4-agent code review baseline.")
    parser.add_argument("--dataset", type=Path, default=Path("data/pr_dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("data/multi_agent_results.json"))
    parser.add_argument("--pr", type=int, default=None, help="Only re-run a single PR number")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.pr is not None:
        run_single_pr(args.dataset, args.output, args.pr)
    else:
        run_multi_agent_baseline(args.dataset, args.output)