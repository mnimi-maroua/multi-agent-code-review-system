"""
Step 2: Multi-model single-agent baseline comparison.

Runs the SAME generic reviewer prompt through several models on the same
PR dataset, so review quality differences can be attributed to the model
itself rather than to prompt design.

Reasoning models (e.g. Qwen) emit their chain-of-thought wrapped in
<think>...</think> before the actual review. This script separates that
reasoning trace from the final answer, so you get both:
    - "reasoning": the raw chain-of-thought (useful for analysis)
    - "answer":    the clean, final review text (useful for display/reading)
    - "truncated": True if the model ran out of tokens mid-reasoning and
                    never produced a closing </think> tag / final answer



Before running, verify current model names/limits:
    https://console.groq.com/docs/models
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline_multi_model")


REVIEW_PROMPT = """You are an experienced code reviewer. Review this diff \
the way you would on a real pull request. Be specific and concrete, not generic.

Diff:
{diff}
"""

MAX_DIFF_CHARS = 8000        # keep prompts within a safe context window across models
MAX_RETRIES = 3
REQUEST_DELAY_SECONDS = 1.0  # spacing between calls to stay under rate limits
DEFAULT_MAX_TOKENS = 3000

# Reasoning models (chain-of-thought before the final answer) need a larger
# token budget, or their response gets cut off mid-reasoning before they
# ever state a conclusion. Override per model here as needed.
MAX_TOKENS_BY_MODEL: dict[str, int] = {
    "qwen/qwen3.6-27b": 6000,
}

THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def max_tokens(self) -> int:
        return MAX_TOKENS_BY_MODEL.get(self.model, DEFAULT_MAX_TOKENS)


# ---------------------------------------------------------------------------
# Models under comparison. Chosen to isolate two variables independently:
#   - model SIZE   (llama-3.3-70b vs llama-3.1-8b, same family)
#   - model FAMILY (Llama vs GPT-OSS vs Qwen, comparable size class)
# ---------------------------------------------------------------------------
MODELS_TO_RUN: list[ModelConfig] = [
    ModelConfig("groq", "llama-3.3-70b-versatile"),
    ModelConfig("groq", "openai/gpt-oss-120b"),
    ModelConfig("groq", "llama-3.1-8b-instant"),
    ModelConfig("groq", "qwen/qwen3.6-27b"),
]


@dataclass
class CallResult:
    text: str | None
    ok: bool
    error: str | None = None


def split_reasoning_and_answer(raw_text: str) -> dict:
    """
    Separate a <think>...</think> reasoning trace from the final answer.

    Returns a dict with:
        - "answer":    text after </think>, or the full text if no <think> tag
                        was ever opened (non-reasoning model)
        - "reasoning": the content inside <think>...</think>, or None
        - "truncated": True if a <think> tag was opened but never closed
                        (model ran out of tokens before finishing)
    """
    if raw_text is None:
        return {"answer": "", "reasoning": None, "truncated": False}

    match = THINK_BLOCK_RE.search(raw_text)
    if match:
        reasoning = match.group(1).strip()
        answer = raw_text[match.end():].strip()
        return {"answer": answer, "reasoning": reasoning, "truncated": False}

    if THINK_OPEN_RE.search(raw_text):
        # <think> was opened but the response was cut off before </think>
        # -- the whole thing is reasoning, and there is no final answer yet.
        reasoning = THINK_OPEN_RE.sub("", raw_text, count=1).strip()
        return {"answer": "", "reasoning": reasoning, "truncated": True}

    # No <think> tag at all -- a plain, non-reasoning response.
    return {"answer": raw_text.strip(), "reasoning": None, "truncated": False}


# ---------------------------------------------------------------------------
# Provider callers -- each returns the raw review text or raises on failure.
# Each caller accepts max_tokens explicitly, so per-model overrides
# (see MAX_TOKENS_BY_MODEL) actually take effect at call time.
# Add a new provider by writing a `call_<provider>` function with this
# signature and registering it in PROVIDER_CALLERS below.
# ---------------------------------------------------------------------------

def call_groq(model: str, diff_text: str, max_tokens: int) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": REVIEW_PROMPT.format(diff=diff_text)}],
    )
    return response.choices[0].message.content


PROVIDER_CALLERS: dict[str, Callable[[str, str, int], str]] = {
    "groq": call_groq,
}


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------

def check_required_env(models: list[ModelConfig]) -> None:
    """Fail fast if a required API key is missing, instead of failing per-call."""
    required_keys = {
        "groq": "GROQ_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    providers_used = {m.provider for m in models}
    missing = [
        required_keys[p] for p in providers_used
        if p in required_keys and not os.getenv(required_keys[p])
    ]
    if missing:
        raise EnvironmentError(f"Missing required environment variable(s): {', '.join(missing)}")


def call_with_retry(config: ModelConfig, diff_text: str, max_retries: int = MAX_RETRIES) -> CallResult:
    """Call a model with exponential backoff. Never raises -- returns a CallResult instead."""
    caller = PROVIDER_CALLERS.get(config.provider)
    if caller is None:
        return CallResult(text=None, ok=False, error=f"Unknown provider '{config.provider}'")

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            text = caller(config.model, diff_text, config.max_tokens)
            return CallResult(text=text, ok=True)
        except Exception as exc:  # noqa: BLE001 -- intentionally broad, we log and retry
            last_error = str(exc)
            wait = 2 ** attempt
            logger.warning("[%s] attempt %d/%d failed (%s); retrying in %ds",
                            config.key, attempt, max_retries, last_error, wait)
            time.sleep(wait)

    return CallResult(text=None, ok=False, error=last_error)


def build_model_review_entry(config: ModelConfig, result: CallResult) -> dict:
    """Turn a CallResult into the structured entry saved in the results JSON."""
    if not result.ok:
        return {"answer": f"[ERROR] {result.error}", "reasoning": None, "truncated": False}
    return split_reasoning_and_answer(result.text)


def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run fetch_prs.py first, "
            f"or pass the correct path with --dataset."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_baseline(
    dataset_path: Path,
    output_path: Path,
    models: list[ModelConfig] = None,
) -> dict:
    """Run every configured model over every PR in the dataset. Returns a run summary."""
    models = models or MODELS_TO_RUN
    check_required_env(models)

    dataset = load_dataset(dataset_path)
    logger.info("Loaded %d PRs from %s", len(dataset), dataset_path)
    logger.info("Models under comparison: %s",
                ", ".join(f"{m.key} (max_tokens={m.max_tokens})" for m in models))

    results: list[dict] = []
    failures: list[str] = []
    truncated_count = 0

    for i, pr in enumerate(dataset, start=1):
        logger.info("[%d/%d] PR #%s - %s", i, len(dataset), pr["number"], pr["title"])
        diff_text = pr["diff"][:MAX_DIFF_CHARS]

        model_reviews: dict[str, dict] = {}
        for config in models:
            result = call_with_retry(config, diff_text)
            entry = build_model_review_entry(config, result)
            model_reviews[config.key] = entry

            if not result.ok:
                failures.append(f"PR #{pr['number']} / {config.key}: {result.error}")
            if entry["truncated"]:
                truncated_count += 1
                logger.warning("[%s] reasoning truncated on PR #%s -- consider raising max_tokens",
                                config.key, pr["number"])

            time.sleep(REQUEST_DELAY_SECONDS)

        results.append({
            "number": pr["number"],
            "title": pr["title"],
            "category": pr["category"],
            "human_review_comments": pr["review_comments"],
            "model_reviews": model_reviews,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = {
        "prs_processed": len(results),
        "models_compared": [m.key for m in models],
        "failed_calls": len(failures),
        "truncated_reasoning": truncated_count,
        "output_path": str(output_path),
    }

    logger.info("Done. %d PRs x %d models saved to %s", len(results), len(models), output_path)
    if failures:
        logger.warning("%d call(s) failed after retries:", len(failures))
        for line in failures:
            logger.warning("  - %s", line)
    else:
        logger.info("No failed calls.")
    if truncated_count:
        logger.warning("%d response(s) had truncated reasoning (no final answer produced).", truncated_count)

    return summary


def run_single_pr(dataset_path: Path, output_path: Path, pr_number: int, models: list[ModelConfig] = None) -> None:
    """Re-run just one PR and merge the result into an existing results file."""
    models = models or MODELS_TO_RUN
    check_required_env(models)

    full_dataset = load_dataset(dataset_path)
    pr = next((p for p in full_dataset if p["number"] == pr_number), None)
    if pr is None:
        raise SystemExit(f"PR #{pr_number} not found in {dataset_path}")

    diff_text = pr["diff"][:MAX_DIFF_CHARS]
    model_reviews: dict[str, dict] = {}
    for config in models:
        logger.info("-> %s (max_tokens=%d)", config.key, config.max_tokens)
        result = call_with_retry(config, diff_text)
        entry = build_model_review_entry(config, result)
        model_reviews[config.key] = entry
        if entry["truncated"]:
            logger.warning("[%s] reasoning truncated -- consider raising max_tokens further", config.key)
        time.sleep(REQUEST_DELAY_SECONDS)

    existing = load_dataset(output_path) if output_path.exists() else []
    existing = [r for r in existing if r["number"] != pr_number]
    existing.append({
        "number": pr["number"],
        "title": pr["title"],
        "category": pr["category"],
        "human_review_comments": pr["review_comments"],
        "model_reviews": model_reviews,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    logger.info("Updated PR #%s in %s", pr_number, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-model code review baseline.")
    parser.add_argument("--dataset", type=Path, default=Path("data/pr_dataset.json"),
                         help="Path to the PR dataset produced by fetch_prs.py")
    parser.add_argument("--output", type=Path, default=Path("data/baseline_results_multi.json"),
                         help="Where to save the multi-model results")
    parser.add_argument("--pr", type=int, default=None,
                         help="Only re-run a single PR number (useful after tuning max_tokens for one model)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.pr is not None:
        run_single_pr(args.dataset, args.output, args.pr)
    else:
        run_baseline(dataset_path=args.dataset, output_path=args.output)