"""
Side-by-side comparison of model reviews vs real human review comments
for a given PR. Reasoning models are shown with their final answer first,
and their raw chain-of-thought available separately (collapsed by default
in the text output, but included so nothing is lost).

Usage:
    python compare_models.py 1821
    python compare_models.py 1821 --save
    python compare_models.py 1821 --save --show-reasoning
"""

import argparse
import json
from pathlib import Path


def format_comparison(pr_number: int, results_path: Path, show_reasoning: bool) -> str:
    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    pr = next((r for r in results if r["number"] == pr_number), None)
    if pr is None:
        return f"PR #{pr_number} not found in {results_path}"

    lines = []
    lines.append("=" * 70)
    lines.append(f"PR #{pr['number']} - {pr['title']}  (category {pr['category']})")
    lines.append("=" * 70)
    lines.append("")

    lines.append("--- HUMAN REVIEW COMMENTS ---")
    if pr["human_review_comments"]:
        for c in pr["human_review_comments"]:
            lines.append(f"  [{c['author']}] {c['body']}")
    else:
        lines.append("  (none -- Category B, no inline review comments on this PR)")

    for model_key, entry in pr["model_reviews"].items():
        lines.append("")
        lines.append(f"--- {model_key.upper()} ---")

        # Backward-compat: older results files stored a plain string instead
        # of the {"answer", "reasoning", "truncated"} structure.
        if isinstance(entry, str):
            lines.append(entry)
            continue

        if entry.get("truncated"):
            lines.append("[TRUNCATED -- model ran out of tokens before finishing its reasoning, "
                          "no final answer was produced. Raise MAX_TOKENS_BY_MODEL for this model.]")
        elif entry.get("answer"):
            lines.append(entry["answer"])
        else:
            lines.append("[EMPTY -- no answer text returned]")

        if entry.get("reasoning") and show_reasoning:
            lines.append("")
            lines.append(f"  [reasoning trace for {model_key}]")
            lines.append("  " + entry["reasoning"].replace("\n", "\n  "))
        elif entry.get("reasoning"):
            lines.append(f"  (reasoning trace available -- rerun with --show-reasoning to see it)")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model reviews against human comments for one PR.")
    parser.add_argument("pr_number", type=int, help="PR number to inspect")
    parser.add_argument("--results", type=Path, default=Path("data/baseline_results_multi.json"),
                         help="Path to the multi-model results JSON")
    parser.add_argument("--save", action="store_true",
                         help="Write the comparison to a text file instead of printing it")
    parser.add_argument("--output", type=Path, default=None,
                         help="Output file path when --save is used (default: comparison_<pr_number>.txt)")
    parser.add_argument("--show-reasoning", action="store_true",
                         help="Include the full <think> reasoning trace for reasoning models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    text = format_comparison(args.pr_number, args.results, args.show_reasoning)

    if args.save:
        output_path = args.output or Path(f"comparison_{args.pr_number}.txt")
        output_path.write_text(text, encoding="utf-8")
        print(f"Saved to {output_path}")
    else:
        print(text)