"""
Cost calculator for common OpenAI and Anthropic models.
Prices in USD per 1M tokens (as of mid-2024 — update as needed).
"""

PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o":                     {"in": 5.00,   "out": 15.00},
    "gpt-4o-mini":                {"in": 0.15,   "out": 0.60},
    "gpt-4-turbo":                {"in": 10.00,  "out": 30.00},
    "gpt-4":                      {"in": 30.00,  "out": 60.00},
    "gpt-3.5-turbo":              {"in": 0.50,   "out": 1.50},
    "gpt-3.5-turbo-16k":          {"in": 3.00,   "out": 4.00},
    "o1":                         {"in": 15.00,  "out": 60.00},
    "o1-mini":                    {"in": 3.00,   "out": 12.00},
    "o3-mini":                    {"in": 1.10,   "out": 4.40},
    # Anthropic
    "claude-opus-4-5":            {"in": 15.00,  "out": 75.00},
    "claude-sonnet-4-5":          {"in": 3.00,   "out": 15.00},
    "claude-haiku-4-5":           {"in": 0.80,   "out": 4.00},
    "claude-3-opus-20240229":     {"in": 15.00,  "out": 75.00},
    "claude-3-sonnet-20240229":   {"in": 3.00,   "out": 15.00},
    "claude-3-haiku-20240307":    {"in": 0.25,   "out": 1.25},
    "claude-3-5-sonnet-20241022": {"in": 3.00,   "out": 15.00},
}

DEFAULT_COST = {"in": 5.00, "out": 15.00}


def model_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return cost in USD for given token counts."""
    p = PRICING.get(model, DEFAULT_COST)
    return (tokens_in * p["in"] + tokens_out * p["out"]) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)
