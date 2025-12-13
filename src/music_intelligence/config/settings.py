"""Centralized project configuration."""

from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Configuration for audio processing."""

    sample_rate: int = 22050
    hop_length: int = 512
    n_fft: int = 2048


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""

    tempo_range: tuple[float, float] = (60.0, 200.0)
    onset_threshold: float = 0.1
    beat_track_trim: bool = True


DEFAULT_AUDIO_CONFIG = AudioConfig()
DEFAULT_FEATURE_CONFIG = FeatureExtractionConfig()

