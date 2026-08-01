"""
Step 4: Orchestrate the 4 review agents with LangGraph.

This replaces the plain sequential function calls in multi_agent_baseline.py
with a StateGraph: each agent is a node, the diff/reviews flow through a
shared state object, and edges define execution order.

Execution is kept SEQUENTIAL (Style -> Logic -> Test -> Critic), not
parallel fan-out, on purpose: Step 3 showed that firing multiple agents
at once creates request bursts that blow through Groq's shared
tokens-per-minute quota, even with a rate limiter in place. LangGraph
would happily run nodes concurrently, but that reintroduces the exact
problem the rate limiter was built to avoid. Sequential execution here
still demonstrates the graph/state/node/edge pattern -- switching to
parallel later is a one-line change (see comment near add_edge calls)
once running on a paid tier with real concurrency headroom.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from agents.agents import run_style_agent, run_logic_agent, run_test_agent, run_critic_agent
from logging_config import setup_logging

load_dotenv()

logger = logging.getLogger("graph")

MAX_DIFF_CHARS = 8000
REQUEST_DELAY_SECONDS = 1.0


class ReviewState(TypedDict):
    diff: str
    style_review: str
    logic_review: str
    test_review: str
    final_review: str


# ---------------------------------------------------------------------------
# Nodes -- each one reads from state and returns the keys it updates.
# LangGraph merges the returned dict into the running state automatically.
# ---------------------------------------------------------------------------

def style_node(state: ReviewState) -> dict:
    return {"style_review": run_style_agent(state["diff"])}


def logic_node(state: ReviewState) -> dict:
    return {"logic_review": run_logic_agent(state["diff"])}


def test_node(state: ReviewState) -> dict:
    return {"test_review": run_test_agent(state["diff"])}


def critic_node(state: ReviewState) -> dict:
    final = run_critic_agent(
        state.get("style_review", ""),
        state.get("logic_review", ""),
        state.get("test_review", ""),
    )
    return {"final_review": final}


def build_graph():
    graph = StateGraph(ReviewState)

    graph.add_node("style", style_node)
    graph.add_node("logic", logic_node)
    graph.add_node("test", test_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("style")

    # Sequential chain -- see module docstring for why this isn't fan-out
    # parallel (graph.add_edge(START, "style"); ...(START, "logic"); etc.)
    graph.add_edge("style", "logic")
    graph.add_edge("logic", "test")
    graph.add_edge("test", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


app = build_graph()


# ---------------------------------------------------------------------------
# Same dataset/resume/incremental-save pattern as multi_agent_baseline.py,
# reused here so this file is drop-in comparable to the Step 3 results.
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run fetch_prs.py first.")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pr_has_valid_result(entry: dict) -> bool:
    if "error" in entry:
        return False
    final_review = entry.get("final_review", "")
    return bool(final_review) and not final_review.startswith("[ERROR]") \
        and not final_review.startswith("[INCOMPLETE")


def review_with_graph(diff_text: str) -> dict:
    result = app.invoke({"diff": diff_text, "style_review": "", "logic_review": "",
                          "test_review": "", "final_review": ""})
    return {
        "style_review": result["style_review"],
        "logic_review": result["logic_review"],
        "test_review": result["test_review"],
        "final_review": result["final_review"],
    }


def run_graph_baseline(dataset_path: Path, output_path: Path) -> None:
    dataset = load_dataset(dataset_path)
    logger.info("Loaded %d PRs from %s", len(dataset), dataset_path)

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
            agent_outputs = review_with_graph(diff_text)
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
    logger.info("Running graph on PR #%s - %s", pr["number"], pr["title"])

    try:
        agent_outputs = review_with_graph(diff_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PR #%s failed: %s", pr["number"], exc)
        agent_outputs = {"error": str(exc)}

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
    parser = argparse.ArgumentParser(description="Run the LangGraph-orchestrated code review.")
    parser.add_argument("--dataset", type=Path, default=Path("data/pr_dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("data/graph_results.json"))
    parser.add_argument("--pr", type=int, default=None, help="Only run a single PR number")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging("graph")
    args = parse_args()
    if args.pr is not None:
        run_single_pr(args.dataset, args.output, args.pr)
    else:
        run_graph_baseline(args.dataset, args.output)