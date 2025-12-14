import logging

import librosa
import numpy as np
import numpy.typing as npt

from audio_processing.config import HOP_LENGTH, N_FFT
from audio_processing.rhythm_analysis import analyze_rhythm

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


def extract_onset_times(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
) -> npt.NDArray[np.float64]:
    """Extracts onset times (attack points) from an audio signal."""
    try:
        onset_times = librosa.onset.onset_detect(
            y=audio,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            units="time",
        )

        onset_times_array = np.array(onset_times, dtype=np.float64)
        logger.debug(f"Onsets extracted: {len(onset_times_array)} onsets")
        return onset_times_array

    except Exception as e:
        logger.error(f"Error extracting onsets: {e}")
        raise


def extract_note_durations(
    onset_times: npt.NDArray[np.float64],
) -> dict:
    """Analyzes note/event durations based on onset intervals."""
    try:
        if len(onset_times) < 2:
            logger.warning("Not enough onsets to analyze durations")
            return {
                "durations": [],
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "percentile_25": 0.0,
                "percentile_75": 0.0,
                "long_notes_count": 0,
                "short_notes_count": 0,
                "long_to_short_ratio": 0.0,
            }

        durations = np.diff(onset_times)
        durations_array = durations.astype(np.float64)

        mean_duration = float(np.mean(durations_array))
        median_duration = float(np.median(durations_array))
        std_duration = float(np.std(durations_array))
        min_duration = float(np.min(durations_array))
        max_duration = float(np.max(durations_array))
        percentile_25 = float(np.percentile(durations_array, 25))
        percentile_75 = float(np.percentile(durations_array, 75))

        threshold = median_duration
        long_notes = durations_array > threshold
        short_notes = durations_array <= threshold
        long_notes_count = int(np.sum(long_notes))
        short_notes_count = int(np.sum(short_notes))

        long_to_short_ratio = (
            long_notes_count / short_notes_count if short_notes_count > 0 else 0.0
        )

        logger.debug(
            f"Note durations analyzed: mean={mean_duration:.3f}s, "
            f"median={median_duration:.3f}s, long/short ratio={long_to_short_ratio:.2f}"
        )

        return {
            "durations": durations_array.tolist(),
            "mean": mean_duration,
            "median": median_duration,
            "std": std_duration,
            "min": min_duration,
            "max": max_duration,
            "percentile_25": percentile_25,
            "percentile_75": percentile_75,
            "long_notes_count": long_notes_count,
            "short_notes_count": short_notes_count,
            "long_to_short_ratio": long_to_short_ratio,
            "threshold_seconds": threshold,
        }

    except Exception as e:
        logger.error(f"Error analyzing note durations: {e}")
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


def extract_mfccs(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    n_mfcc: int = 13,
) -> npt.NDArray[np.float32]:
    """Extracts MFCCs (Mel-frequency Cepstral Coefficients) from an audio signal."""
    try:
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        mfccs_array = mfccs.astype(np.float32)
        logger.debug(f"MFCCs extracted: shape {mfccs_array.shape}")
        return mfccs_array

    except Exception as e:
        logger.error(f"Error extracting MFCCs: {e}")
        raise


def extract_log_magnitude_spectrogram(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
) -> npt.NDArray[np.float32]:
    """Extracts log-magnitude spectrogram from an audio signal."""
    try:
        stft = librosa.stft(
            y=audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        magnitude = np.abs(stft)
        log_magnitude = librosa.power_to_db(magnitude**2, ref=np.max)
        log_magnitude_array = log_magnitude.astype(np.float32)
        logger.debug(f"Log-magnitude spectrogram extracted: shape {log_magnitude_array.shape}")
        return log_magnitude_array

    except Exception as e:
        logger.error(f"Error extracting log-magnitude spectrogram: {e}")
        raise


def extract_cqt(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    n_bins: int = 84,
) -> npt.NDArray[np.float32]:
    """Extracts CQT (Constant-Q Transform) from an audio signal."""
    try:
        cqt = librosa.cqt(
            y=audio,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            n_bins=n_bins,
        )
        cqt_magnitude = np.abs(cqt)
        cqt_log = librosa.power_to_db(cqt_magnitude**2, ref=np.max)
        cqt_array = cqt_log.astype(np.float32)
        logger.debug(f"CQT extracted: shape {cqt_array.shape}")
        return cqt_array

    except Exception as e:
        logger.error(f"Error extracting CQT: {e}")
        raise


def extract_all_features(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
) -> dict:
    """Extracts all features from an audio signal."""
    tempo = extract_tempo(audio, sample_rate)
    beat_times = extract_beat_times(audio, sample_rate, tempo)
    onset_times = extract_onset_times(audio, sample_rate)
    time_signature = estimate_time_signature(audio, sample_rate, tempo, beat_times)

    energy = extract_energy(audio)
    brightness = extract_brightness(audio, sample_rate)
    spectral_density = extract_spectral_density(audio, sample_rate)

    mfccs = extract_mfccs(audio, sample_rate)
    log_magnitude_spectrogram = extract_log_magnitude_spectrogram(audio, sample_rate)
    cqt = extract_cqt(audio, sample_rate)

    note_durations = extract_note_durations(onset_times)
    rhythm_analysis = analyze_rhythm(
        onset_times, beat_times, tempo, time_signature
    )

    return {
        "tempo": tempo,
        "beat_times": beat_times,
        "onset_times": onset_times,
        "time_signature": time_signature,
        "energy": energy,
        "brightness": brightness,
        "spectral_density": spectral_density,
        "mfccs": mfccs,
        "log_magnitude_spectrogram": log_magnitude_spectrogram,
        "cqt": cqt,
        "note_durations": note_durations,
        "rhythm_analysis": rhythm_analysis,
    }

