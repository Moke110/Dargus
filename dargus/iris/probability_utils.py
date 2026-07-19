from __future__ import annotations

import math
from typing import Any


def effect_to_probability(effect_size: float) -> float:
    """Map a normalized effect size to a probability of positive efficacy.

    Uses a logistic link so that an effect size of 0 maps to 0.5,
    positive effects map above 0.5 and negative effects below 0.5.
    """
    # Guard against overflow for very large magnitudes.
    if effect_size >= 20.0:
        return 1.0
    if effect_size <= -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-effect_size))


def probability_interval_from_effect(
    effect_size: float,
    ci_lower: float,
    ci_upper: float,
) -> tuple[float, float]:
    """Convert an effect-size point estimate and 95% CI to a probability interval."""
    return clip_interval(
        effect_to_probability(ci_lower),
        effect_to_probability(ci_upper),
    )


def clip_interval(low: float, up: float) -> tuple[float, float]:
    """Clip interval endpoints to [0, 1] and ensure low <= up."""
    low, up = min(low, up), max(low, up)
    return max(0.0, min(1.0, low)), max(0.0, min(1.0, up))


def normalize_prediction_entry(
    entry: dict[str, Any] | None,
    reasoning_mode: str = "unknown",
    confidence_level: str = "unknown",
) -> dict[str, Any]:
    """Normalize a prediction entry to the canonical v5 Iris output format.

    Accepts either canonical ``efficacy_low`` / ``efficacy_up`` keys or legacy
    ``normalized_effect_size`` / ``ci95_lower`` / ``ci95_upper`` keys.
    """
    if entry is None:
        entry = {}

    if "efficacy_low" in entry and "efficacy_up" in entry:
        low, up = clip_interval(entry["efficacy_low"], entry["efficacy_up"])
        return {
            "efficacy_low": low,
            "efficacy_up": up,
            "supporting_records": list(entry.get("supporting_records", [])),
            "reasoning_mode": entry.get("reasoning_mode", reasoning_mode),
            "confidence_level": entry.get("confidence_level", confidence_level),
        }

    effect = entry.get("normalized_effect_size", 0.0)
    ci_lower = entry.get("ci95_lower", effect)
    ci_upper = entry.get("ci95_upper", effect)
    low, up = probability_interval_from_effect(effect, ci_lower, ci_upper)
    return {
        "efficacy_low": low,
        "efficacy_up": up,
        "supporting_records": list(entry.get("supporting_records", [])),
        "reasoning_mode": entry.get("reasoning_mode", reasoning_mode),
        "confidence_level": entry.get("confidence_level", confidence_level),
    }
