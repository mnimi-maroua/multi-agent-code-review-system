"""
Specialized agent callers.

Config B: all four agents use the SAME model (GPT-OSS 120B via Groq), so
any quality difference vs. the single-agent baseline can be attributed to
prompt specialization alone, not to a different/better model.

All calls go through a shared RateLimiter to respect Groq's tokens-per-minute
cap, instead of relying on fixed sleep() guesses.
"""

import logging
import os
from groq import Groq

from agents.prompts import (
    STYLE_AGENT_PROMPT,
    LOGIC_AGENT_PROMPT,
    TEST_AGENT_PROMPT,
    CRITIC_AGENT_PROMPT,
)
from agents.rate_limiter import RateLimiter, estimate_tokens

logger = logging.getLogger("agents")

# Config B: one model for every role. Change per-role here for Config C.
MODEL_FOR_ROLE = {
    "style": "openai/gpt-oss-120b",
    "logic": "openai/gpt-oss-120b",
    "test": "openai/gpt-oss-120b",
    "critic": "openai/gpt-oss-120b",
}

MAX_TOKENS_FOR_ROLE = {
    "style": 2000,
    "logic": 4000,
    "test": 1200,
    "critic": 2500,
}

MIN_VALID_LENGTH = 20  # a real answer is never just a few characters

# Groq free tier: 8000 tokens/minute for openai/gpt-oss-120b at time of writing.
# Verify current limits at https://console.groq.com/docs/rate-limits before changing.
TOKENS_PER_MINUTE = 8000
_rate_limiter = RateLimiter(tokens_per_minute=TOKENS_PER_MINUTE)


def _call_groq(model: str, prompt: str, max_tokens: int, role: str) -> str:
    # Reserve budget for both the prompt tokens (input) and the requested
    # completion tokens (output) -- both count against the TPM cap.
    estimated_prompt_tokens = estimate_tokens(prompt)
    _rate_limiter.reserve(estimated_prompt_tokens + max_tokens)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = response.choices[0]
    text = choice.message.content or ""

    logger.info("[%s] finish_reason=%s, length=%d chars", role, choice.finish_reason, len(text))
    if len(text.strip()) < MIN_VALID_LENGTH:
        logger.warning("[%s] suspiciously short/empty response (finish_reason=%s): %r",
                        role, choice.finish_reason, text)

    return text


def run_style_agent(diff_text: str) -> str:
    prompt = STYLE_AGENT_PROMPT.format(diff=diff_text)
    return _call_groq(MODEL_FOR_ROLE["style"], prompt, MAX_TOKENS_FOR_ROLE["style"], "style")


def run_logic_agent(diff_text: str) -> str:
    prompt = LOGIC_AGENT_PROMPT.format(diff=diff_text)
    return _call_groq(MODEL_FOR_ROLE["logic"], prompt, MAX_TOKENS_FOR_ROLE["logic"], "logic")


def run_test_agent(diff_text: str) -> str:
    prompt = TEST_AGENT_PROMPT.format(diff=diff_text)
    return _call_groq(MODEL_FOR_ROLE["test"], prompt, MAX_TOKENS_FOR_ROLE["test"], "test")


def run_critic_agent(style_review: str, logic_review: str, test_review: str) -> str:
    # Guardrail: don't let the Critic silently fabricate a synthesis from
    # broken inputs. If an upstream agent failed, say so explicitly instead
    # of producing a confident-sounding but potentially wrong review.
    inputs = {"style": style_review, "logic": logic_review, "test": test_review}
    empty_inputs = [name for name, text in inputs.items() if len(text.strip()) < MIN_VALID_LENGTH]
    if empty_inputs:
        logger.warning("Critic received empty/invalid input(s) from: %s -- flagging instead of synthesizing",
                        empty_inputs)
        return (f"[INCOMPLETE REVIEW] The following agent(s) returned no usable output: "
                f"{', '.join(empty_inputs)}. Re-run this PR before trusting this review.")

    prompt = CRITIC_AGENT_PROMPT.format(
        style_review=style_review,
        logic_review=logic_review,
        test_review=test_review,
    )
    return _call_groq(MODEL_FOR_ROLE["critic"], prompt, MAX_TOKENS_FOR_ROLE["critic"], "critic")