"""Interprets raw rhythm analysis data into musical categories and descriptors."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

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
class RhythmInterpretation:
    """Complete rhythm interpretation with all categories."""

    swing: SwingInterpretation
    syncopation: SyncopationInterpretation
    timing: TimingInterpretation
    groove_feel: GrooveFeel
    groove_description: str
    summary: str


def interpret_swing(swing_data: dict[str, Any]) -> SwingInterpretation:
    """Converts raw swing metrics into musical interpretation."""
    ratio = swing_data.get("swing_ratio", 1.0)
    has_swing = swing_data.get("has_swing", False)

    if not has_swing or ratio < 1.1:
        level = SwingLevel.STRAIGHT
        description = "No swing detected. Straight, even rhythmic subdivisions."
    elif ratio < 1.3:
        level = SwingLevel.LIGHT
        description = "Light swing feel. Subtle rhythmic lilt."
    elif ratio < 1.5:
        level = SwingLevel.MODERATE
        description = "Moderate swing. Classic jazz-like triplet feel."
    else:
        level = SwingLevel.HEAVY
        description = "Heavy swing. Pronounced shuffle or dotted rhythm feel."

    logger.debug(f"Swing interpreted: {level.value} (ratio={ratio:.3f})")

    return SwingInterpretation(
        level=level,
        ratio=ratio,
        description=description,
    )


def interpret_syncopation(syncopation_data: dict[str, Any]) -> SyncopationInterpretation:
    """Converts raw syncopation metrics into musical interpretation."""
    ratio = syncopation_data.get("syncopation_ratio", 0.0)
    count = syncopation_data.get("syncopated_onsets_count", 0)
    total = syncopation_data.get("total_onsets", 0)

    if ratio < 0.05:
        level = SyncopationLevel.MINIMAL
        description = "Almost no syncopation. Very straightforward, on-the-beat rhythm."
    elif ratio < 0.15:
        level = SyncopationLevel.LOW
        description = "Low syncopation. Mostly on-beat with occasional off-beat accents."
    elif ratio < 0.30:
        level = SyncopationLevel.MODERATE
        description = "Moderate syncopation. Rhythmic interest with off-beat emphasis."
    elif ratio < 0.45:
        level = SyncopationLevel.HIGH
        description = "High syncopation. Frequent off-beat accents create rhythmic tension."
    else:
        level = SyncopationLevel.EXTREME
        description = "Extreme syncopation. Heavily off-beat, complex rhythmic pattern."

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
    mean_deviation = micro_timing_data.get("mean_deviation_ms", 0.0)
    std_deviation = micro_timing_data.get("std_deviation_ms", 0.0)

    # Determine timing tendency (early, on-beat, or late)
    if mean_deviation < -15:
        tendency = "ahead"
        tendency_desc = "slightly ahead of the beat"
    elif mean_deviation > 15:
        tendency = "behind"
        tendency_desc = "slightly behind the beat (laid back)"
    else:
        tendency = "centered"
        tendency_desc = "centered on the beat"

    # Determine timing consistency
    if std_deviation < 10:
        feel = TimingFeel.MECHANICAL
        description = f"Extremely precise timing, {tendency_desc}. Likely programmed or quantized."
    elif std_deviation < 30:
        feel = TimingFeel.TIGHT
        description = f"Very tight timing, {tendency_desc}. Highly skilled or quantized performance."
    elif std_deviation < 80:
        feel = TimingFeel.NATURAL
        description = f"Natural human timing, {tendency_desc}. Organic feel with subtle variations."
    elif std_deviation < 150:
        feel = TimingFeel.LOOSE
        description = f"Loose timing, {tendency_desc}. Relaxed or intentionally imprecise."
    else:
        feel = TimingFeel.SLOPPY
        description = f"Very loose timing, {tendency_desc}. High variation in note placement."

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


def interpret_rhythm(rhythm_analysis: dict[str, Any]) -> RhythmInterpretation:
    """Main function to interpret complete rhythm analysis data."""
    swing_data = rhythm_analysis.get("swing", {})
    syncopation_data = rhythm_analysis.get("syncopation", {})
    micro_timing_data = rhythm_analysis.get("micro_timing", {})

    swing = interpret_swing(swing_data)
    syncopation = interpret_syncopation(syncopation_data)
    timing = interpret_micro_timing(micro_timing_data)

    groove_feel, groove_description = determine_groove_feel(swing, syncopation, timing)
    summary = generate_summary(swing, syncopation, timing, groove_feel)

    logger.info(f"Rhythm interpretation complete: groove={groove_feel.value}")

    return RhythmInterpretation(
        swing=swing,
        syncopation=syncopation,
        timing=timing,
        groove_feel=groove_feel,
        groove_description=groove_description,
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
        "summary": interpretation.summary,
    }
