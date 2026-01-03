"""Configuration loader for interpretation rules."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_config_cache: dict[str, Any] | None = None


def load_interpretation_rules() -> dict[str, Any]:
    """Loads interpretation rules from JSON configuration file."""
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    # Rules are in rules/ folder at project root (same level as src/)
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "rules" / "interpretation_rules.json"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _config_cache = json.load(f)
        logger.debug(f"Interpretation rules loaded from {config_path}")
        return _config_cache
    except FileNotFoundError:
        logger.error(f"Interpretation rules file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in interpretation rules: {e}")
        raise


def find_threshold_level(
    value: float,
    thresholds: list[dict[str, Any]],
    field_name: str = "max",
) -> dict[str, Any] | None:
    """Finds the appropriate threshold level for a given value."""
    for threshold in thresholds:
        max_val = threshold.get("max")
        min_val = threshold.get("min")

        if max_val is None and min_val is None:
            continue

        # Range check: min <= value < max
        if max_val is None:
            if min_val is not None and value >= min_val:
                return threshold
        elif min_val is None:
            if value < max_val:
                return threshold
        else:
            if min_val <= value < max_val:
                return threshold

    return None


def get_swing_rules() -> dict[str, Any]:
    """Gets swing interpretation rules."""
    rules = load_interpretation_rules()
    return rules["swing"]


def get_syncopation_rules() -> dict[str, Any]:
    """Gets syncopation interpretation rules."""
    rules = load_interpretation_rules()
    return rules["syncopation"]


def get_timing_rules() -> dict[str, Any]:
    """Gets timing interpretation rules."""
    rules = load_interpretation_rules()
    return rules["timing"]


def get_brightness_rules() -> dict[str, Any]:
    """Gets brightness interpretation rules."""
    rules = load_interpretation_rules()
    return rules["brightness"]


def get_harmonicity_rules() -> dict[str, Any]:
    """Gets harmonicity interpretation rules."""
    rules = load_interpretation_rules()
    return rules["harmonicity"]


def get_sound_character_rules() -> dict[str, Any]:
    """Gets sound character interpretation rules."""
    rules = load_interpretation_rules()
    return rules["sound_character"]


def get_tone_color_quality_rules() -> dict[str, Any]:
    """Gets tone color quality interpretation rules."""
    rules = load_interpretation_rules()
    return rules["tone_color_quality"]

