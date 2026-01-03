"""Audio processing module: feature extraction, rhythm analysis, and tone color interpretation."""

from src.audio_processing.tone_color_interpreter import (
    BrightnessLevel,
    HarmonicityLevel,
    SoundCharacter,
    ToneColorInterpretation,
    ToneColorQuality,
    interpret_tone_color,
    tone_color_interpretation_to_dict,
)

__all__ = [
    "interpret_tone_color",
    "tone_color_interpretation_to_dict",
    "ToneColorInterpretation",
    "BrightnessLevel",
    "ToneColorQuality",
    "HarmonicityLevel",
    "SoundCharacter",
]
