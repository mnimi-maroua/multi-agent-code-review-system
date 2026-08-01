"""
Named agent configurations, kept side by side so both experiments stay
documented and reproducible.

Config B: same model for every role. Isolates the effect of prompt
    specialization alone (already run and evaluated: avg 4.07/5 on the
    LLM-judge, see data/evaluation_report.md).

Config C: a different model per role, chosen from what Step 2/3 testing
    already showed:
    - Logic needs real reasoning (Qwen caught the causal bug->fix link
      that GPT-OSS sometimes missed) -> qwen/qwen3.6-27b
    - Style/Test are narrow-scope tasks that don't need a large model
      -> llama-3.1-8b-instant
    - Critic showed the strongest synthesis + edge-case detection
      -> openai/gpt-oss-120b

Switch between them with the AGENT_CONFIG environment variable (see .env):
    AGENT_CONFIG=B   (default if unset)
    AGENT_CONFIG=C
"""

CONFIG_B = {
    "MODEL_FOR_ROLE": {
        "style": "openai/gpt-oss-120b",
        "logic": "openai/gpt-oss-120b",
        "test": "openai/gpt-oss-120b",
        "critic": "openai/gpt-oss-120b",
    },
    "MAX_TOKENS_FOR_ROLE": {
        "style": 2000,
        "logic": 4000,
        "test": 1800,
        "critic": 2500,
    },
}

CONFIG_C = {
    "MODEL_FOR_ROLE": {
        "style": "llama-3.1-8b-instant",
        "logic": "qwen/qwen3.6-27b",
        "test": "llama-3.1-8b-instant",
        "critic": "openai/gpt-oss-120b",
    },
    "MAX_TOKENS_FOR_ROLE": {
        "style": 1200,
        # Qwen is a reasoning model: it emits a <think>...</think> block
        # before its actual answer. Step 2 testing showed it needs ~6000
        # tokens or it gets cut off mid-reasoning with no final answer.
        "logic": 6000,
        "test": 1000,
        "critic": 2500,
    },
}

CONFIGS = {"B": CONFIG_B, "C": CONFIG_C}