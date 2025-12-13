"""Extraction of global features: energy, brightness, density, etc."""

import logging

import librosa
import numpy as np
import numpy.typing as npt

from music_intelligence.config.settings import DEFAULT_AUDIO_CONFIG

logger = logging.getLogger(__name__)


def extract_energy(
    audio: npt.NDArray[np.float32],
    frame_length: int | None = None,
    hop_length: int | None = None,
) -> float:
    """Calculates the average energy of an audio signal."""
    if frame_length is None:
        frame_length = DEFAULT_AUDIO_CONFIG.n_fft
    if hop_length is None:
        hop_length = DEFAULT_AUDIO_CONFIG.hop_length

    root_mean_square = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    energy = float(np.mean(root_mean_square))
    logger.debug(f"Energy calculated: {energy:.4f}")
    return energy


def extract_brightness(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    n_fft: int | None = None,
    hop_length: int | None = None,
) -> float:
    """Calculates the spectral brightness of an audio signal."""
    if n_fft is None:
        n_fft = DEFAULT_AUDIO_CONFIG.n_fft
    if hop_length is None:
        hop_length = DEFAULT_AUDIO_CONFIG.hop_length

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]

    max_freq = sample_rate / 2.0
    brightness = float(np.mean(spectral_centroid) / max_freq)

    logger.debug(f"Brightness calculated: {brightness:.4f}")
    return brightness


def extract_spectral_density(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    n_fft: int | None = None,
    hop_length: int | None = None,
) -> float:
    """Calculates the spectral density of an audio signal."""
    if n_fft is None:
        n_fft = DEFAULT_AUDIO_CONFIG.n_fft
    if hop_length is None:
        hop_length = DEFAULT_AUDIO_CONFIG.hop_length

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        roll_percent=0.95,
    )[0]

    max_freq = sample_rate / 2.0
    rolloff_normalized = np.mean(spectral_rolloff) / max_freq

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]

    bandwidth_normalized = np.mean(spectral_bandwidth) / max_freq

    density = float((rolloff_normalized + bandwidth_normalized) / 2.0)

    logger.debug(f"Spectral density calculated: {density:.4f}")
    return density


def extract_global_features(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
) -> dict[str, float]:
    """Extracts all global features from an audio signal."""
    energy = extract_energy(audio)
    brightness = extract_brightness(audio, sample_rate)
    spectral_density = extract_spectral_density(audio, sample_rate)

    return {
        "energy": energy,
        "brightness": brightness,
        "spectral_density": spectral_density,
    }

