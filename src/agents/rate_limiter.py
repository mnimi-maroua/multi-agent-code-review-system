"""
Sliding-window rate limiter for Groq's tokens-per-minute (TPM) cap.

Free-tier Groq models are capped at a fixed number of tokens per rolling
60-second window (e.g. 8000 TPM for openai/gpt-oss-120b at time of writing).
Running Style/Logic/Test in parallel with generous max_tokens budgets
blows through this window almost immediately, causing repeated 429s.

Instead of guessing fixed sleep() durations, this tracks how many tokens
have been "reserved" in the last 60 seconds and blocks just long enough
to stay under the cap before each call.

This is intentionally conservative: it counts max_tokens (the requested
completion budget) plus a rough estimate of prompt tokens, even though
the model may use fewer tokens than requested. Better to wait a bit too
long occasionally than to keep hitting 429s.
"""

import threading
import time
from collections import deque


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token for English text)."""
    return max(1, len(text) // 4)


class RateLimiter:
    def __init__(self, tokens_per_minute: int, window_seconds: int = 60):
        self.tokens_per_minute = tokens_per_minute
        self.window_seconds = window_seconds
        self._log: deque[tuple[float, int]] = deque()  # (timestamp, tokens_reserved)
        self._lock = threading.Lock()

    def _prune(self, now: float) -> int:
        """Drop entries older than the window and return tokens still counted."""
        while self._log and now - self._log[0][0] > self.window_seconds:
            self._log.popleft()
        return sum(tokens for _, tokens in self._log)

    def reserve(self, tokens: int) -> None:
        """Block until reserving `tokens` keeps the rolling total under the cap."""
        with self._lock:
            while True:
                now = time.time()
                used = self._prune(now)

                if used + tokens <= self.tokens_per_minute:
                    self._log.append((now, tokens))
                    return

                # Not enough room yet -- wait until the oldest entry ages out
                oldest_ts = self._log[0][0]
                wait = self.window_seconds - (now - oldest_ts) + 0.5  # small safety margin
                wait = max(wait, 1.0)
                time.sleep(wait)