"""
Step 6: Confidence flagging guardrail.

An LLM review that is full of hedging language ("might", "possibly",
"it's unclear whether", "I'm not sure") is effectively the model telling
you it isn't confident -- but that signal gets lost if the review is
just presented as flat, authoritative text. This scans the final review
for hedge density and prepends a visible flag when it's high, so a human
reading it knows to double-check rather than trust it at face value.

This mirrors the "confidence threshold" guardrail from the original
101 guide (Step 6): flag instead of assert when the agent isn't sure.
"""

import re

HEDGE_PATTERNS = [
    r"\bmight\b", r"\bmay\b", r"\bpossibly\b", r"\bperhaps\b",
    r"\bnot (?:entirely |completely |fully )?sure\b",
    r"\bunclear\b", r"\bit'?s (?:hard|difficult) to (?:tell|say|know)\b",
    r"\bI (?:don'?t|do not) (?:know|understand) why\b",
    r"\bcould be\b", r"\bseems? to\b", r"\bappears? to\b",
    r"\bassuming\b", r"\bpresumably\b",
]

HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)

# Hedges are normal in a handful of sentences -- flag only when they're
# dense enough to suggest the review as a whole is more guesswork than
# grounded analysis.
HEDGE_DENSITY_THRESHOLD = 0.012  # hedge matches per word


def compute_hedge_density(text: str) -> float:
    word_count = max(1, len(text.split()))
    hedge_count = len(HEDGE_RE.findall(text))
    return hedge_count / word_count


def apply_confidence_flag(review_text: str) -> str:
    """Prepend a low-confidence banner if the review reads as more speculative than grounded."""
    if not review_text or review_text.startswith("[ERROR]") or review_text.startswith("[INCOMPLETE"):
        return review_text

    density = compute_hedge_density(review_text)
    if density >= HEDGE_DENSITY_THRESHOLD:
        banner = (
            "⚠️ LOW CONFIDENCE: this review contains a high density of "
            "hedging language (\"might\", \"unclear\", \"possibly\", ...). "
            "Treat its conclusions as a starting point for human review, "
            "not as settled findings.\n\n"
        )
        return banner + review_text

    return review_text