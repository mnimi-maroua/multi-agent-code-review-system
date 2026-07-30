"""
Automates the collection of PR data for evaluating the multi-agent system.

This script replaces all the manual commands:
1. Fetch closed PRs from a repo
2. Filter to keep only real merged contributions (excludes bots/CI/releases)
3. Download the diff for each PR
4. Fetch review comments (if any)
5. Classify each PR as Category A (has human comments) or B (diff + description only)
6. Save everything cleanly to JSON

Usage:
    python fetch_prs.py
"""

import os
import re
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN is missing from the .env file")

HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"}

# Config: change these values depending on the repo you want to analyze
OWNER = "fastapi"
REPO = "typer"
MAX_PRS = 100  # how many closed PRs to fetch before filtering

# Regex to exclude automated/noise PRs (bots, CI, releases, minor docs)
NOISE_PATTERN = re.compile(r"⬆|👷|🔖|🔒️|📝|dependabot", re.IGNORECASE)


def get_closed_merged_prs(owner: str, repo: str, max_prs: int = 100) -> list[dict]:
    """Fetch closed PRs and filter to keep only real merged contributions."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    params = {"state": "closed", "per_page": min(max_prs, 100)}
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    prs = resp.json()

    filtered = [
        pr for pr in prs
        if pr.get("merged_at") is not None
        and not NOISE_PATTERN.search(pr["title"])
        and len(pr["title"].strip()) > 5  # discard suspicious/empty titles
    ]
    return filtered


def get_diff(diff_url: str) -> str:
    """Download the raw diff for a PR."""
    resp = requests.get(diff_url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text


def get_review_comments(owner: str, repo: str, pr_number: int) -> list[dict]:
    """Fetch inline review comments on a PR."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def build_dataset(owner: str, repo: str, max_prs: int = 100) -> list[dict]:
    """Build the full dataset: diff + comments + A/B classification for each PR."""
    prs = get_closed_merged_prs(owner, repo, max_prs)
    print(f"{len(prs)} relevant PRs found after filtering on {owner}/{repo}")

    dataset = []
    for pr in prs:
        number = pr["number"]
        title = pr["title"]
        print(f"  -> processing PR #{number}: {title}")

        diff = get_diff(pr["diff_url"])
        comments = get_review_comments(owner, repo, number)

        category = "A" if len(comments) > 0 else "B"

        dataset.append({
            "number": number,
            "title": title,
            "merged_at": pr["merged_at"],
            "html_url": pr["html_url"],
            "body": pr.get("body", ""),
            "diff": diff,
            "review_comments": [
                {"body": c["body"], "path": c["path"], "line": c.get("line"), "author": c["user"]["login"]}
                for c in comments
            ],
            "category": category,
        })

        time.sleep(0.5)  # avoid hitting GitHub's rate limit too hard

    return dataset


if __name__ == "__main__":
    data = build_dataset(OWNER, REPO, MAX_PRS)

    output_path = "pr_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    cat_a = sum(1 for d in data if d["category"] == "A")
    cat_b = sum(1 for d in data if d["category"] == "B")
    print(f"\nDone. {len(data)} PRs saved to {output_path}")
    print(f"  Category A (with review comments): {cat_a}")
    print(f"  Category B (diff + description only): {cat_b}")