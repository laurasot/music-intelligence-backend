"""Interprets raw rhythm analysis data into musical categories and descriptors."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from config.interpretation_config import (
    find_threshold_level,
    get_rhythmic_stability_rules,
    get_swing_rules,
    get_syncopation_rules,
    get_timing_rules,
)

logger = logging.getLogger(__name__)


class SwingLevel(Enum):
    """Classification of swing intensity."""

    STRAIGHT = "straight"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"


class SyncopationLevel(Enum):
    """Classification of syncopation intensity."""

    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class TimingFeel(Enum):
    """Classification of micro-timing characteristics."""

    MECHANICAL = "mechanical"
    TIGHT = "tight"
    NATURAL = "natural"
    LOOSE = "loose"
    SLOPPY = "sloppy"


class GrooveFeel(Enum):
    """Overall groove classification combining all rhythm factors."""

    ROBOTIC = "robotic"
    PRECISE = "precise"
    GROOVY = "groovy"
    LAID_BACK = "laid_back"
    DRIVING = "driving"
    CHAOTIC = "chaotic"


class RhythmicStabilityLevel(Enum):
    """Classification of rhythmic stability."""

    VERY_STABLE = "very_stable"
    STABLE = "stable"
    LOOSE = "loose"
    CHAOTIC = "chaotic"


@dataclass
class SwingInterpretation:
    """Interpreted swing characteristics."""

    level: SwingLevel
    ratio: float
    description: str


@dataclass
class SyncopationInterpretation:
    """Interpreted syncopation characteristics."""

    level: SyncopationLevel
    ratio: float
    count: int
    total: int
    description: str


@dataclass
class TimingInterpretation:
    """Interpreted micro-timing characteristics."""

    feel: TimingFeel
    mean_deviation_ms: float
    std_deviation_ms: float
    tendency: str
    description: str


@dataclass
class RhythmicStabilityInterpretation:
    """Interpreted rhythmic stability characteristics."""

    level: RhythmicStabilityLevel
    timing_variation_ms: float
    duration_variation_s: float
    syncopation_ratio: float
    description: str


@dataclass
class RhythmInterpretation:
    """Complete rhythm interpretation with all categories."""

    swing: SwingInterpretation
    syncopation: SyncopationInterpretation
    timing: TimingInterpretation
    groove_feel: GrooveFeel
    groove_description: str
    rhythmic_stability: RhythmicStabilityInterpretation
    summary: str


def interpret_swing(swing_data: dict[str, Any]) -> SwingInterpretation:
    """Converts raw swing metrics into musical interpretation."""
    rules = get_swing_rules()
    ratio = swing_data.get(rules["field"], 1.0)
    has_swing = swing_data.get(rules["has_swing_field"], False)

    # Check priority condition first (has_swing == false has priority)
    if not has_swing:
        level = SwingLevel(rules["priority_level"])
        description = rules["priority_description"]
    else:
        # Find threshold that matches the ratio
        threshold = find_threshold_level(ratio, rules["thresholds"])
        if threshold is None:
            threshold = rules["thresholds"][-1]
        level = SwingLevel(threshold["level"])
        description = threshold["description"]

    logger.debug(f"Swing interpreted: {level.value} (ratio={ratio:.3f})")

    return SwingInterpretation(
        level=level,
        ratio=ratio,
        description=description,
    )


def interpret_syncopation(syncopation_data: dict[str, Any]) -> SyncopationInterpretation:
    """Converts raw syncopation metrics into musical interpretation."""
    rules = get_syncopation_rules()
    ratio = syncopation_data.get(rules["field"], 0.0)
    count = syncopation_data.get("syncopated_onsets_count", 0)
    total = syncopation_data.get("total_onsets", 0)

    threshold = find_threshold_level(ratio, rules["thresholds"])
    if threshold is None:
        threshold = rules["thresholds"][-1]

    level = SyncopationLevel(threshold["level"])
    description = threshold["description"]

    logger.debug(f"Syncopation interpreted: {level.value} (ratio={ratio:.3f})")

    return SyncopationInterpretation(
        level=level,
        ratio=ratio,
        count=count,
        total=total,
        description=description,
    )


def interpret_micro_timing(micro_timing_data: dict[str, Any]) -> TimingInterpretation:
    """Converts raw micro-timing metrics into musical interpretation."""
    rules = get_timing_rules()
    mean_deviation = micro_timing_data.get("mean_deviation_ms", 0.0)
    std_deviation = micro_timing_data.get("std_deviation_ms", 0.0)

    # Determine timing tendency (early, on-beat, or late)
    tendency_rules = rules["tendency"]
    tendency_threshold = find_threshold_level(mean_deviation, tendency_rules["thresholds"])
    if tendency_threshold is None:
        tendency_threshold = tendency_rules["thresholds"][-1]

    tendency = tendency_threshold["tendency"]
    tendency_desc = tendency_threshold["description"]

    # Determine timing consistency
    feel_rules = rules["feel"]
    feel_threshold = find_threshold_level(std_deviation, feel_rules["thresholds"])
    if feel_threshold is None:
        feel_threshold = feel_rules["thresholds"][-1]

    feel = TimingFeel(feel_threshold["level"])
    description = feel_threshold["description_template"].format(tendency_desc=tendency_desc)

    logger.debug(
        f"Timing interpreted: {feel.value} (mean={mean_deviation:.1f}ms, std={std_deviation:.1f}ms)"
    )

    return TimingInterpretation(
        feel=feel,
        mean_deviation_ms=mean_deviation,
        std_deviation_ms=std_deviation,
        tendency=tendency,
        description=description,
    )


def determine_groove_feel(
    swing: SwingInterpretation,
    syncopation: SyncopationInterpretation,
    timing: TimingInterpretation,
) -> tuple[GrooveFeel, str]:
    """Determines overall groove feel from combined rhythm characteristics."""
    # Mechanical/Robotic: straight, minimal syncopation, very precise
    if (
        swing.level == SwingLevel.STRAIGHT
        and syncopation.level in [SyncopationLevel.MINIMAL, SyncopationLevel.LOW]
        and timing.feel == TimingFeel.MECHANICAL
    ):
        return (
            GrooveFeel.ROBOTIC,
            "Robotic, machine-like rhythm. Quantized and precise, typical of electronic music.",
        )

    # Precise: straight or light swing, tight timing
    if timing.feel in [TimingFeel.MECHANICAL, TimingFeel.TIGHT] and swing.level in [
        SwingLevel.STRAIGHT,
        SwingLevel.LIGHT,
    ]:
        if syncopation.level in [SyncopationLevel.MODERATE, SyncopationLevel.HIGH]:
            return (
                GrooveFeel.GROOVY,
                "Groovy and precise. Syncopated patterns with tight execution.",
            )
        return (
            GrooveFeel.PRECISE,
            "Precise and controlled rhythm. Tight timing with clear beat structure.",
        )

    # Laid back: behind the beat tendency
    if timing.tendency == "behind" and timing.feel in [TimingFeel.NATURAL, TimingFeel.LOOSE]:
        return (
            GrooveFeel.LAID_BACK,
            "Laid back groove. Notes placed slightly behind the beat for relaxed feel.",
        )

    # Driving: ahead of beat, moderate to high energy
    if timing.tendency == "ahead" and syncopation.level in [
        SyncopationLevel.LOW,
        SyncopationLevel.MODERATE,
    ]:
        return (
            GrooveFeel.DRIVING,
            "Driving, forward-pushing rhythm. Slightly ahead of the beat, creating urgency.",
        )

    # Groovy: swing or syncopation with natural timing
    if (
        swing.level in [SwingLevel.MODERATE, SwingLevel.HEAVY]
        or syncopation.level in [SyncopationLevel.HIGH, SyncopationLevel.EXTREME]
    ):
        return (
            GrooveFeel.GROOVY,
            "Groovy, danceable rhythm. Strong rhythmic interest from swing or syncopation.",
        )

    # Chaotic: sloppy timing with extreme syncopation
    if timing.feel == TimingFeel.SLOPPY or syncopation.level == SyncopationLevel.EXTREME:
        return (
            GrooveFeel.CHAOTIC,
            "Chaotic, unpredictable rhythm. High variation and off-beat emphasis.",
        )

    # Default: groovy (catch-all for interesting combinations)
    return (
        GrooveFeel.GROOVY,
        "Balanced groove with rhythmic character.",
    )


def generate_summary(
    swing: SwingInterpretation,
    syncopation: SyncopationInterpretation,
    timing: TimingInterpretation,
    groove: GrooveFeel,
) -> str:
    """Generates a human-readable summary of the rhythm analysis."""
    parts = []

    # Swing description
    if swing.level == SwingLevel.STRAIGHT:
        parts.append("straight eighth-note feel")
    elif swing.level == SwingLevel.LIGHT:
        parts.append("subtle swing")
    elif swing.level == SwingLevel.MODERATE:
        parts.append("jazzy swing feel")
    else:
        parts.append("heavy shuffle groove")

    # Syncopation description
    if syncopation.level == SyncopationLevel.MINIMAL:
        parts.append("on-beat rhythm")
    elif syncopation.level == SyncopationLevel.LOW:
        parts.append("mostly on-beat with occasional syncopation")
    elif syncopation.level == SyncopationLevel.MODERATE:
        parts.append(f"moderate syncopation ({syncopation.count} syncopated notes)")
    elif syncopation.level == SyncopationLevel.HIGH:
        parts.append("heavily syncopated pattern")
    else:
        parts.append("extremely syncopated, complex rhythm")

    # Timing description
    if timing.feel == TimingFeel.MECHANICAL:
        parts.append("machine-precise timing")
    elif timing.feel == TimingFeel.TIGHT:
        parts.append("tight, controlled timing")
    elif timing.feel == TimingFeel.NATURAL:
        parts.append("natural human feel")
    elif timing.feel == TimingFeel.LOOSE:
        parts.append("loose, relaxed timing")
    else:
        parts.append("free-form timing")

    summary = f"This track features a {parts[0]} with {parts[1]} and {parts[2]}."

    # Add groove conclusion
    groove_conclusions = {
        GrooveFeel.ROBOTIC: "The overall feel is mechanical and precise.",
        GrooveFeel.PRECISE: "The rhythm is controlled and well-defined.",
        GrooveFeel.GROOVY: "Creates an engaging, danceable groove.",
        GrooveFeel.LAID_BACK: "The laid-back timing gives it a relaxed, chill vibe.",
        GrooveFeel.DRIVING: "The forward push creates energy and momentum.",
        GrooveFeel.CHAOTIC: "The unpredictable rhythm creates tension and complexity.",
    }

    summary += f" {groove_conclusions.get(groove, '')}"

    return summary


def interpret_rhythmic_stability(
    rhythm_analysis: dict[str, Any],
    note_durations: dict[str, Any] | None = None,
) -> RhythmicStabilityInterpretation:
    """Interprets rhythmic stability from timing and duration variations."""
    rules = get_rhythmic_stability_rules()
    fields = rules["fields"]

    # Extract values using field paths
    timing_variation_ms = rhythm_analysis.get("micro_timing", {}).get("std_deviation_ms", 0.0)
    
    if note_durations:
        duration_variation_s = note_durations.get("std", 0.0)
    else:
        duration_variation_s = 0.0

    syncopation_ratio = rhythm_analysis.get("syncopation", {}).get("syncopation_ratio", 0.0)

    # Find matching threshold (evaluates both timing and duration constraints)
    thresholds = rules["thresholds"]
    matched_threshold = None

    for threshold in thresholds:
        max_timing = threshold.get("max_timing_ms")
        max_duration = threshold.get("max_duration_std")

        # Default threshold (chaotic) matches if no other does
        if max_timing is None and max_duration is None:
            matched_threshold = threshold
            break

        # Check if both constraints are met
        timing_ok = max_timing is None or timing_variation_ms < max_timing
        duration_ok = max_duration is None or duration_variation_s < max_duration

        if timing_ok and duration_ok:
            matched_threshold = threshold
            break

    if matched_threshold is None:
        matched_threshold = thresholds[-1]  # Default to last (chaotic)

    level = RhythmicStabilityLevel(matched_threshold["level"])
    description = matched_threshold["description"]

    logger.debug(
        f"Rhythmic stability interpreted: {level.value} "
        f"(timing={timing_variation_ms:.1f}ms, duration={duration_variation_s:.3f}s)"
    )

    return RhythmicStabilityInterpretation(
        level=level,
        timing_variation_ms=timing_variation_ms,
        duration_variation_s=duration_variation_s,
        syncopation_ratio=syncopation_ratio,
        description=description,
    )


def interpret_rhythm(
    rhythm_analysis: dict[str, Any],
    note_durations: dict[str, Any] | None = None,
) -> RhythmInterpretation:
    """Main function to interpret complete rhythm analysis data."""
    swing_data = rhythm_analysis.get("swing", {})
    syncopation_data = rhythm_analysis.get("syncopation", {})
    micro_timing_data = rhythm_analysis.get("micro_timing", {})

    swing = interpret_swing(swing_data)
    syncopation = interpret_syncopation(syncopation_data)
    timing = interpret_micro_timing(micro_timing_data)

    groove_feel, groove_description = determine_groove_feel(swing, syncopation, timing)
    rhythmic_stability = interpret_rhythmic_stability(rhythm_analysis, note_durations)
    summary = generate_summary(swing, syncopation, timing, groove_feel)

    logger.info(f"Rhythm interpretation complete: groove={groove_feel.value}, stability={rhythmic_stability.level.value}")

    return RhythmInterpretation(
        swing=swing,
        syncopation=syncopation,
        timing=timing,
        groove_feel=groove_feel,
        groove_description=groove_description,
        rhythmic_stability=rhythmic_stability,
        summary=summary,
    )


def rhythm_interpretation_to_dict(interpretation: RhythmInterpretation) -> dict[str, Any]:
    """Converts RhythmInterpretation dataclass to dictionary for JSON serialization."""
    return {
        "swing": {
            "level": interpretation.swing.level.value,
            "ratio": interpretation.swing.ratio,
            "description": interpretation.swing.description,
        },
        "syncopation": {
            "level": interpretation.syncopation.level.value,
            "ratio": interpretation.syncopation.ratio,
            "count": interpretation.syncopation.count,
            "total": interpretation.syncopation.total,
            "description": interpretation.syncopation.description,
        },
        "timing": {
            "feel": interpretation.timing.feel.value,
            "mean_deviation_ms": interpretation.timing.mean_deviation_ms,
            "std_deviation_ms": interpretation.timing.std_deviation_ms,
            "tendency": interpretation.timing.tendency,
            "description": interpretation.timing.description,
        },
        "groove": {
            "feel": interpretation.groove_feel.value,
            "description": interpretation.groove_description,
        },
        "rhythmic_stability": {
            "level": interpretation.rhythmic_stability.level.value,
            "timing_variation_ms": interpretation.rhythmic_stability.timing_variation_ms,
            "duration_variation_s": interpretation.rhythmic_stability.duration_variation_s,
            "syncopation_ratio": interpretation.rhythmic_stability.syncopation_ratio,
            "description": interpretation.rhythmic_stability.description,
        },
        "summary": interpretation.summary,
    }
