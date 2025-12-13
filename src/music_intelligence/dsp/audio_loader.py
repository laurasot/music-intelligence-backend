"""Functions for audio loading and preprocessing."""

import logging
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt

from music_intelligence.config.settings import AudioConfig, DEFAULT_AUDIO_CONFIG

logger = logging.getLogger(__name__)


def load_audio(
    path: Path | str,
    sample_rate: int | None = None,
    config: AudioConfig | None = None,
) -> tuple[npt.NDArray[np.float32], int]:
    """Loads an audio file and converts it to mono."""
    if config is None:
        config = DEFAULT_AUDIO_CONFIG

    target_sample_rate = (
        sample_rate if sample_rate is not None else config.sample_rate
    )

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio, loaded_sample_rate = librosa.load(
            str(audio_path),
            sr=target_sample_rate,
            mono=True,
            res_type="kaiser_best",
        )
        audio = audio.astype(np.float32)

        logger.info(
            f"Audio loaded: {len(audio)} samples at {loaded_sample_rate} Hz "
            f"(duration: {len(audio) / loaded_sample_rate:.2f}s)"
        )

        return audio, loaded_sample_rate

    except Exception as e:
        raise ValueError(f"Error loading audio file {path}: {e}") from e


def normalize_audio(
    audio: npt.NDArray[np.float32],
    target_db: float = -3.0,
) -> npt.NDArray[np.float32]:
    """Normalizes audio to a target decibel level."""
    if audio.size == 0:
        logger.warning("Empty audio received for normalization")
        return audio

    root_mean_square = np.sqrt(np.mean(audio**2))
    if root_mean_square == 0:
        logger.warning("Silent audio, cannot normalize")
        return audio

    current_db = 20 * np.log10(root_mean_square + 1e-10)
    gain_linear = 10 ** ((target_db - current_db) / 20)

    normalized = audio * gain_linear

    max_val = np.max(np.abs(normalized))
    if max_val > 1.0:
        normalized = normalized / max_val
        logger.debug(f"Audio clipped during normalization (max: {max_val:.2f})")

    return normalized.astype(np.float32)

