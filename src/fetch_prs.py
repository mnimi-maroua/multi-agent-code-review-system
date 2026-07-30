"""
Automatise la collecte de données PR pour l'évaluation du système multi-agent.

Ce script remplace toutes les commandes PowerShell manuelles qu'on a faites :
1. Récupère les PRs fermées d'un repo
2. Filtre pour ne garder que les vraies contributions mergées (exclut bots/CI/releases)
3. Télécharge le diff de chaque PR
4. Récupère les review comments (s'il y en a)
5. Classe chaque PR en Catégorie A (avec commentaires humains) ou B (diff + description seulement)
6. Sauvegarde tout proprement en JSON

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
    raise ValueError("GITHUB_TOKEN manquant dans le fichier .env")

HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"}

# Config : change ces valeurs selon le repo que tu veux analyser
OWNER = "tiangolo"
REPO = "typer"
MAX_PRS = 100  # combien de PRs fermées récupérer pour filtrer dedans

# Regex pour exclure les PRs automatiques/bruit (bots, CI, releases, docs mineures)
NOISE_PATTERN = re.compile(r"⬆|👷|🔖|🔒️|📝|dependabot", re.IGNORECASE)


def get_closed_merged_prs(owner: str, repo: str, max_prs: int = 100) -> list[dict]:
    """Récupère les PRs fermées et filtre pour ne garder que les vraies contributions mergées."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    params = {"state": "closed", "per_page": min(max_prs, 100)}
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    prs = resp.json()

    filtered = [
        pr for pr in prs
        if pr.get("merged_at") is not None
        and not NOISE_PATTERN.search(pr["title"])
        and len(pr["title"].strip()) > 5  # écarte les titres suspects/vides
    ]
    return filtered


def get_diff(diff_url: str) -> str:
    """Télécharge le diff brut d'une PR."""
    resp = requests.get(diff_url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text


def get_review_comments(owner: str, repo: str, pr_number: int) -> list[dict]:
    """Récupère les commentaires de review inline sur une PR."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def build_dataset(owner: str, repo: str, max_prs: int = 100) -> list[dict]:
    """Construit le dataset complet : diff + comments + classification A/B pour chaque PR."""
    prs = get_closed_merged_prs(owner, repo, max_prs)
    print(f"{len(prs)} PRs pertinentes trouvées après filtrage sur {owner}/{repo}")

    dataset = []
    for pr in prs:
        number = pr["number"]
        title = pr["title"]
        print(f"  -> traitement PR #{number}: {title}")

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

        time.sleep(0.5)  # évite de taper trop fort sur le rate limit GitHub

    return dataset


if __name__ == "__main__":
    data = build_dataset(OWNER, REPO, MAX_PRS)

    output_path = "pr_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    cat_a = sum(1 for d in data if d["category"] == "A")
    cat_b = sum(1 for d in data if d["category"] == "B")
    print(f"\nTerminé. {len(data)} PRs sauvegardées dans {output_path}")
    print(f"  Catégorie A (avec review comments): {cat_a}")
    print(f"  Catégorie B (diff + description seulement): {cat_b}")