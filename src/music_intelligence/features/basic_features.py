"""Extraction of basic features: tempo, beat, and time signature."""

import logging

import librosa
import numpy as np
import numpy.typing as npt

from music_intelligence.config.settings import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_FEATURE_CONFIG,
    FeatureExtractionConfig,
)

logger = logging.getLogger(__name__)


def extract_tempo(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    config: FeatureExtractionConfig | None = None,
) -> float:
    """Extracts tempo (BPM) from an audio signal."""
    if config is None:
        config = DEFAULT_FEATURE_CONFIG

    try:
        tempo, _ = librosa.beat.tempo(
            y=audio,
            sr=sample_rate,
            hop_length=DEFAULT_AUDIO_CONFIG.hop_length,
            start_bpm=120.0,
            std_bpm=1.0,
            ac_size=8.0,
            max_tempo=config.tempo_range[1],
        )

        tempo_value = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)

        logger.debug(f"Tempo extracted: {tempo_value:.2f} BPM")
        return tempo_value

    except Exception as e:
        logger.error(f"Error extracting tempo: {e}")
        raise


def extract_beat_times(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    tempo: float | None = None,
    config: FeatureExtractionConfig | None = None,
) -> npt.NDArray[np.float64]:
    """Extracts beat times from an audio signal."""
    if config is None:
        config = DEFAULT_FEATURE_CONFIG

    try:
        if tempo is None:
            tempo = extract_tempo(audio, sample_rate, config)

        tempo_bpm = np.array([tempo])

        beats = librosa.beat.beat_track(
            y=audio,
            sr=sample_rate,
            hop_length=DEFAULT_AUDIO_CONFIG.hop_length,
            start_bpm=tempo,
            std_bpm=1.0,
            trim=config.beat_track_trim,
            units="time",
        )[1]

        beat_times = np.array(beats, dtype=np.float64)

        logger.debug(f"Beats extracted: {len(beat_times)} beats")
        return beat_times

    except Exception as e:
        logger.error(f"Error extracting beats: {e}")
        raise


def estimate_time_signature(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    tempo: float | None = None,
    beat_times: npt.NDArray[np.float64] | None = None,
) -> tuple[int, int]:
    """Estimates the approximate time signature of an audio signal."""
    try:
        if tempo is None:
            tempo = extract_tempo(audio, sample_rate)

        if beat_times is None:
            beat_times = extract_beat_times(audio, sample_rate, tempo)

        if len(beat_times) < 4:
            logger.warning("Few beats detected, using default time signature 4/4")
            return (4, 4)

        beat_intervals = np.diff(beat_times)
        mean_interval = np.mean(beat_intervals)

        beats_per_measure_candidates = [2, 3, 4, 6, 8]
        measure_durations = [
            candidate * mean_interval for candidate in beats_per_measure_candidates
        ]

        onset_frames = librosa.onset.onset_detect(
            y=audio,
            sr=sample_rate,
            hop_length=DEFAULT_AUDIO_CONFIG.hop_length,
            units="time",
        )

        if len(onset_frames) < 2:
            logger.warning("Few onsets detected, using default time signature 4/4")
            return (4, 4)

        best_candidate = 4
        best_score = 0.0

        for candidate, measure_duration in zip(
            beats_per_measure_candidates, measure_durations
        ):
            measure_boundaries = np.arange(0, beat_times[-1], measure_duration)
            if len(measure_boundaries) < 2:
                continue

            score = 0.0
            for boundary in measure_boundaries[1:]:
                nearby_onsets = onset_frames[
                    np.abs(onset_frames - boundary) < mean_interval * 0.3
                ]
                if len(nearby_onsets) > 0:
                    score += 1.0

            score = score / len(measure_boundaries[1:])
            if score > best_score:
                best_score = score
                best_candidate = candidate

        time_signature = (best_candidate, 4)

        logger.debug(f"Time signature estimated: {time_signature[0]}/{time_signature[1]}")
        return time_signature

    except Exception as e:
        logger.warning(f"Error estimating time signature, using 4/4 as default: {e}")
        return (4, 4)


def extract_basic_features(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    config: FeatureExtractionConfig | None = None,
) -> dict[str, float | tuple[int, int] | npt.NDArray[np.float64]]:
    """Extracts all basic features (tempo, beat, time signature) from an audio signal."""
    if config is None:
        config = DEFAULT_FEATURE_CONFIG

    tempo = extract_tempo(audio, sample_rate, config)
    beat_times = extract_beat_times(audio, sample_rate, tempo, config)
    time_signature = estimate_time_signature(audio, sample_rate, tempo, beat_times)

    return {
        "tempo": tempo,
        "beat_times": beat_times,
        "time_signature": time_signature,
    }

