"""Interprets raw tone color features into musical categories and descriptors."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from config.interpretation_config import (
    eval_harmonicity_condition,
    find_threshold_level,
    get_brightness_rules,
    get_harmonicity_rules,
    get_sound_character_rules,
    get_tone_color_quality_rules,
)

logger = logging.getLogger(__name__)


class BrightnessLevel(Enum):
    """Classification of spectral brightness."""

    DARK = "dark"
    WARM = "warm"
    BALANCED = "balanced"
    BRIGHT = "bright"
    SHARP = "sharp"


class ToneColorQuality(Enum):
    """Classification of tone color quality (sweet, metallic, opaque, bright)."""

    SWEET = "sweet"
    METALLIC = "metallic"
    OPAQUE = "opaque"
    BRIGHT = "bright"
    RICH = "rich"
    THIN = "thin"


class HarmonicityLevel(Enum):
    """Classification of harmonic vs percussive content."""

    PURE_HARMONIC = "pure_harmonic"
    PREDOMINANTLY_HARMONIC = "predominantly_harmonic"
    MIXED = "mixed"
    PREDOMINANTLY_PERCUSSIVE = "predominantly_percussive"
    PURE_PERCUSSIVE = "pure_percussive"


class SoundCharacter(Enum):
    """Overall sound character classification."""

    CLEAN = "clean"
    CLEAN_WITH_TEXTURE = "clean_with_texture"
    DISTORTED = "distorted"
    NOISY = "noisy"
    PURE = "pure"


@dataclass
class BrightnessInterpretation:
    """Interpreted brightness characteristics."""

    level: BrightnessLevel
    brightness_value: float
    description: str


@dataclass
class ToneColorQualityInterpretation:
    """Interpreted tone color quality."""

    quality: ToneColorQuality
    description: str


@dataclass
class HarmonicityInterpretation:
    """Interpreted harmonic vs percussive characteristics."""

    level: HarmonicityLevel
    harmonic_ratio: float
    percussive_ratio: float
    description: str


@dataclass
class SoundCharacterInterpretation:
    """Interpreted sound character (clean vs distorted)."""

    character: SoundCharacter
    zero_crossing_rate: float
    description: str


@dataclass
class ToneColorInterpretation:
    """Complete tone color interpretation with all categories."""

    brightness: BrightnessInterpretation
    tone_color_quality: ToneColorQualityInterpretation
    harmonicity: HarmonicityInterpretation
    sound_character: SoundCharacterInterpretation
    summary: str


def interpret_brightness(
    brightness_value: float,
    spectral_density: float | None = None,
) -> BrightnessInterpretation:
    """Converts raw brightness metrics into musical interpretation."""
    rules = get_brightness_rules()
    threshold = find_threshold_level(brightness_value, rules["thresholds"])
    if threshold is None:
        threshold = rules["thresholds"][-1]

    level = BrightnessLevel(threshold["level"])
    description = threshold["description"]

    logger.debug(f"Brightness interpreted: {level.value} (value={brightness_value:.3f})")

    return BrightnessInterpretation(
        level=level,
        brightness_value=brightness_value,
        description=description,
    )


def interpret_tone_color_quality(
    brightness: BrightnessInterpretation,
    spectral_density: float,
    mfcc_stats: dict[str, float] | None = None,
) -> ToneColorQualityInterpretation:
    """Determines tone color quality based on brightness and spectral characteristics."""
    brightness_level = brightness.level
    mfcc_variance = mfcc_stats.get("mfcc_variance", 0.0) if mfcc_stats else 0.0

    # Sweet: warm, low variance (simple harmonic content)
    if brightness_level in [BrightnessLevel.DARK, BrightnessLevel.WARM] and mfcc_variance < 5000:
        quality = ToneColorQuality.SWEET
        description = "Sweet, warm tone color with simple harmonic content."
    # Metallic: bright, high variance (complex harmonics)
    elif brightness_level in [BrightnessLevel.BRIGHT, BrightnessLevel.SHARP] and mfcc_variance > 8000:
        quality = ToneColorQuality.METALLIC
        description = "Metallic, bright tone color with complex harmonic structure."
    # Opaque: low brightness, high density (muffled)
    elif brightness_level in [BrightnessLevel.DARK, BrightnessLevel.WARM] and spectral_density > 0.5:
        quality = ToneColorQuality.OPAQUE
        description = "Opaque, muffled tone color with dense spectral content."
    # Bright: high brightness, clear
    elif brightness_level in [BrightnessLevel.BRIGHT, BrightnessLevel.SHARP] and spectral_density < 0.4:
        quality = ToneColorQuality.BRIGHT
        description = "Bright, clear tone color with focused spectral energy."
    # Rich: balanced with high variance (complex)
    elif brightness_level == BrightnessLevel.BALANCED and mfcc_variance > 6000:
        quality = ToneColorQuality.RICH
        description = "Rich, complex tone color with full spectral content."
    # Thin: bright but low density
    elif brightness_level in [BrightnessLevel.BRIGHT, BrightnessLevel.SHARP] and spectral_density < 0.3:
        quality = ToneColorQuality.THIN
        description = "Thin, bright tone color with sparse spectral content."
    # Default: balanced
    else:
        quality = ToneColorQuality.BRIGHT
        description = "Clear, balanced tone color."

    logger.debug(f"Tone color quality interpreted: {quality.value}")

    return ToneColorQualityInterpretation(
        quality=quality,
        description=description,
    )


def interpret_harmonicity(
    harmonic_data: dict[str, float] | None,
) -> HarmonicityInterpretation:
    """Converts harmonic/percussive ratios into musical interpretation."""
    if harmonic_data is None:
        return HarmonicityInterpretation(
            level=HarmonicityLevel.MIXED,
            harmonic_ratio=0.5,
            percussive_ratio=0.5,
            description="Harmonic/percussive separation not available.",
        )

    rules = get_harmonicity_rules()
    harmonic_ratio = harmonic_data.get("harmonic_ratio", 0.5)
    percussive_ratio = harmonic_data.get("percussive_ratio", 0.5)

    # Evaluate rules in order (first match wins)
    for rule in rules["evaluation_order"]:
        if eval_harmonicity_condition(rule["condition"], harmonic_ratio, percussive_ratio):
            level = HarmonicityLevel(rule["level"])
            description = rule["description"]
            break
    else:
        # Fallback (should not happen if default rule exists)
        level = HarmonicityLevel.MIXED
        description = "Mixed harmonic and percussive content. Balanced instrumentation."

    logger.debug(
        f"Harmonicity interpreted: {level.value} "
        f"(harmonic={harmonic_ratio:.3f}, percussive={percussive_ratio:.3f})"
    )

    return HarmonicityInterpretation(
        level=level,
        harmonic_ratio=harmonic_ratio,
        percussive_ratio=percussive_ratio,
        description=description,
    )


def interpret_sound_character(
    zero_crossing_rate: float | None,
    harmonicity: HarmonicityInterpretation,
) -> SoundCharacterInterpretation:
    """Determines sound character (clean vs distorted/noisy)."""
    if zero_crossing_rate is None:
        return SoundCharacterInterpretation(
            character=SoundCharacter.CLEAN,
            zero_crossing_rate=0.0,
            description="Sound character analysis not available.",
        )

    rules = get_sound_character_rules()
    thresholds = rules["thresholds"]
    harmonicity_level_str = harmonicity.level.value

    # Find matching threshold considering conditions
    for threshold in thresholds:
        min_val = threshold.get("min")
        max_val = threshold.get("max")
        condition = threshold.get("condition")

        # Check value range
        matches_range = False
        if min_val is not None and max_val is None:
            matches_range = zero_crossing_rate >= min_val
        elif max_val is not None and min_val is None:
            matches_range = zero_crossing_rate < max_val
        elif min_val is not None and max_val is not None:
            matches_range = min_val <= zero_crossing_rate < max_val

        if matches_range:
            # Check condition if present
            if condition:
                if "harmonicity_level not in" in condition:
                    excluded_str = condition.split("[")[1].split("]")[0]
                    excluded = [s.strip() for s in excluded_str.split(",")]
                    if harmonicity_level_str not in excluded:
                        character = SoundCharacter(threshold["level"])
                        description = threshold["description"]
                        break
                elif "harmonicity_level in" in condition:
                    included_str = condition.split("[")[1].split("]")[0]
                    included = [s.strip() for s in included_str.split(",")]
                    if harmonicity_level_str in included:
                        character = SoundCharacter(threshold["level"])
                        description = threshold["description"]
                        break
                elif "harmonicity_level ==" in condition:
                    expected = condition.split("==")[1].strip()
                    if harmonicity_level_str == expected:
                        character = SoundCharacter(threshold["level"])
                        description = threshold["description"]
                        break
            else:
                character = SoundCharacter(threshold["level"])
                description = threshold["description"]
                break
    else:
        # Default fallback
        character = SoundCharacter.CLEAN
        description = "Clean, clear sound with minimal distortion."

    logger.debug(f"Sound character interpreted: {character.value} (zcr={zero_crossing_rate:.3f})")

    return SoundCharacterInterpretation(
        character=character,
        zero_crossing_rate=zero_crossing_rate,
        description=description,
    )


def generate_tone_color_summary(
    brightness: BrightnessInterpretation,
    tone_color_quality: ToneColorQualityInterpretation,
    harmonicity: HarmonicityInterpretation,
    sound_character: SoundCharacterInterpretation,
) -> str:
    """Generates a human-readable summary of the tone color analysis."""
    parts = []

    # Brightness description
    brightness_descriptions = {
        BrightnessLevel.DARK: "dark, warm tones",
        BrightnessLevel.WARM: "warm, mellow character",
        BrightnessLevel.BALANCED: "balanced spectral distribution",
        BrightnessLevel.BRIGHT: "bright, clear instrumentation",
        BrightnessLevel.SHARP: "sharp, piercing high frequencies",
    }
    parts.append(brightness_descriptions.get(brightness.level, "balanced tone color"))

    # Tone color quality
    quality_descriptions = {
        ToneColorQuality.SWEET: "sweet",
        ToneColorQuality.METALLIC: "metallic",
        ToneColorQuality.OPAQUE: "opaque",
        ToneColorQuality.BRIGHT: "bright",
        ToneColorQuality.RICH: "rich",
        ToneColorQuality.THIN: "thin",
    }
    parts.append(quality_descriptions.get(tone_color_quality.quality, "balanced"))

    # Harmonicity
    if harmonicity.level == HarmonicityLevel.PURE_HARMONIC:
        parts.append("purely harmonic, sustained content")
    elif harmonicity.level == HarmonicityLevel.PREDOMINANTLY_HARMONIC:
        parts.append("predominantly harmonic with melodic focus")
    elif harmonicity.level == HarmonicityLevel.PURE_PERCUSSIVE:
        parts.append("purely percussive, transient-heavy")
    elif harmonicity.level == HarmonicityLevel.PREDOMINANTLY_PERCUSSIVE:
        parts.append("predominantly percussive, rhythmic")
    else:
        parts.append("mixed harmonic and percussive elements")

    # Sound character
    if sound_character.character == SoundCharacter.DISTORTED:
        parts.append("with significant distortion")
    elif sound_character.character == SoundCharacter.NOISY:
        parts.append("with noticeable noise content")
    elif sound_character.character == SoundCharacter.CLEAN_WITH_TEXTURE:
        parts.append("with subtle texture")
    elif sound_character.character == SoundCharacter.PURE:
        parts.append("with pure, clean sound")
    else:
        parts.append("with clean sound")

    # Build summary with available parts
    if len(parts) >= 4:
        summary = (
            f"The tone color features {parts[0]} with a {parts[1]} quality, "
            f"characterized by {parts[2]}, {parts[3]}."
        )
    elif len(parts) >= 3:
        summary = (
            f"The tone color features {parts[0]} with a {parts[1]} quality, "
            f"characterized by {parts[2]}."
        )
    else:
        summary = f"The tone color features {parts[0]} with a {parts[1]} quality."

    # Add energy emphasis if bright
    if brightness.level in [BrightnessLevel.BRIGHT, BrightnessLevel.SHARP]:
        summary += " High energy in upper frequencies creates tension and brightness."

    return summary


def interpret_tone_color(
    brightness: float,
    spectral_density: float,
    harmonic_data: dict[str, float] | None = None,
    zero_crossing_rate: float | None = None,
    mfcc_stats: dict[str, float] | None = None,
) -> ToneColorInterpretation:
    """Main function to interpret complete tone color analysis data."""
    brightness_interp = interpret_brightness(brightness, spectral_density)
    tone_color_quality = interpret_tone_color_quality(
        brightness_interp, spectral_density, mfcc_stats
    )
    harmonicity = interpret_harmonicity(harmonic_data)
    sound_character = interpret_sound_character(zero_crossing_rate, harmonicity)

    summary = generate_tone_color_summary(
        brightness_interp, tone_color_quality, harmonicity, sound_character
    )

    logger.info(f"Tone color interpretation complete: quality={tone_color_quality.quality.value}")

    return ToneColorInterpretation(
        brightness=brightness_interp,
        tone_color_quality=tone_color_quality,
        harmonicity=harmonicity,
        sound_character=sound_character,
        summary=summary,
    )


def tone_color_interpretation_to_dict(interpretation: ToneColorInterpretation) -> dict[str, Any]:
    """Converts ToneColorInterpretation dataclass to dictionary for JSON serialization."""
    return {
        "brightness": {
            "level": interpretation.brightness.level.value,
            "value": interpretation.brightness.brightness_value,
            "description": interpretation.brightness.description,
        },
        "tone_color_quality": {
            "quality": interpretation.tone_color_quality.quality.value,
            "description": interpretation.tone_color_quality.description,
        },
        "harmonicity": {
            "level": interpretation.harmonicity.level.value,
            "harmonic_ratio": interpretation.harmonicity.harmonic_ratio,
            "percussive_ratio": interpretation.harmonicity.percussive_ratio,
            "description": interpretation.harmonicity.description,
        },
        "sound_character": {
            "character": interpretation.sound_character.character.value,
            "zero_crossing_rate": interpretation.sound_character.zero_crossing_rate,
            "description": interpretation.sound_character.description,
        },
        "summary": interpretation.summary,
    }
