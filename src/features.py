import logging

import librosa
import numpy as np
import numpy.typing as npt

from config import HOP_LENGTH, N_FFT

logger = logging.getLogger(__name__)


def extract_tempo(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
) -> float:
    """Extracts tempo (BPM) from an audio signal."""
    try:
        tempo = librosa.beat.tempo(
            y=audio,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            start_bpm=120.0,
            ac_size=8.0,
            max_tempo=200.0,
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
) -> npt.NDArray[np.float64]:
    """Extracts beat times from an audio signal."""
    try:
        if tempo is None:
            tempo = extract_tempo(audio, sample_rate)

        _, beats = librosa.beat.beat_track(
            y=audio,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            start_bpm=tempo,
            tightness=100,
            trim=True,
            units="time",
        )

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
            hop_length=HOP_LENGTH,
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


def extract_energy(
    audio: npt.NDArray[np.float32],
) -> float:
    """Calculates the average energy of an audio signal."""
    root_mean_square = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    energy = float(np.mean(root_mean_square))
    logger.debug(f"Energy calculated: {energy:.4f}")
    return energy


def extract_brightness(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
) -> float:
    """Calculates the spectral brightness of an audio signal."""
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    max_freq = sample_rate / 2.0
    brightness = float(np.mean(spectral_centroid) / max_freq)
    logger.debug(f"Brightness calculated: {brightness:.4f}")
    return brightness


def extract_spectral_density(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
) -> float:
    """Calculates the spectral density of an audio signal."""
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        roll_percent=0.95,
    )[0]

    max_freq = sample_rate / 2.0
    rolloff_normalized = np.mean(spectral_rolloff) / max_freq

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    bandwidth_normalized = np.mean(spectral_bandwidth) / max_freq
    density = float((rolloff_normalized + bandwidth_normalized) / 2.0)
    logger.debug(f"Spectral density calculated: {density:.4f}")
    return density


def extract_all_features(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
) -> dict:
    """Extracts all features from an audio signal."""
    tempo = extract_tempo(audio, sample_rate)
    beat_times = extract_beat_times(audio, sample_rate, tempo)
    time_signature = estimate_time_signature(audio, sample_rate, tempo, beat_times)

    energy = extract_energy(audio)
    brightness = extract_brightness(audio, sample_rate)
    spectral_density = extract_spectral_density(audio, sample_rate)

    return {
        "tempo": tempo,
        "beat_times": beat_times,
        "time_signature": time_signature,
        "energy": energy,
        "brightness": brightness,
        "spectral_density": spectral_density,
    }

