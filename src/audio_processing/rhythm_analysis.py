import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def calculate_swing(
    onset_times: npt.NDArray[np.float64],
    beat_times: npt.NDArray[np.float64],
    tempo: float,
) -> dict:
    """Analyzes swing pattern by measuring the ratio between consecutive inter-onset intervals."""
    try:
        if len(onset_times) < 3:
            logger.warning("Not enough onsets to calculate swing")
            return {
                "swing_ratio": 1.0,
                "has_swing": False,
                "mean_ratio": 1.0,
                "std_ratio": 0.0,
            }

        beat_interval = 60.0 / tempo
        beat_subdivision = beat_interval / 2.0

        swing_ratios = []
        for i in range(len(onset_times) - 1):
            interval = onset_times[i + 1] - onset_times[i]

            if interval < beat_subdivision * 0.6:
                first_part = interval
                if i + 2 < len(onset_times):
                    second_interval = onset_times[i + 2] - onset_times[i + 1]
                    if second_interval < beat_subdivision * 0.6:
                        total = first_part + second_interval
                        if total < beat_interval * 1.2 and total > beat_interval * 0.8:
                            if first_part > 0:
                                ratio = second_interval / first_part
                                if 0.3 < ratio < 3.0:
                                    swing_ratios.append(ratio)

        if len(swing_ratios) == 0:
            return {
                "swing_ratio": 1.0,
                "has_swing": False,
                "mean_ratio": 1.0,
                "std_ratio": 0.0,
            }

        mean_ratio = float(np.mean(swing_ratios))
        std_ratio = float(np.std(swing_ratios))
        has_swing = mean_ratio > 1.05 or mean_ratio < 0.95

        logger.debug(
            f"Swing calculated: ratio={mean_ratio:.3f}, has_swing={has_swing}"
        )

        return {
            "swing_ratio": mean_ratio,
            "has_swing": has_swing,
            "mean_ratio": mean_ratio,
            "std_ratio": std_ratio,
            "num_swing_events": len(swing_ratios),
        }

    except Exception as e:
        logger.error(f"Error calculating swing: {e}")
        raise


def calculate_syncopation(
    onset_times: npt.NDArray[np.float64],
    beat_times: npt.NDArray[np.float64],
    tempo: float,
    time_signature: tuple[int, int],
) -> dict:
    """Detects syncopation by identifying onsets that occur on weak beats or between beats."""
    try:
        if len(onset_times) == 0 or len(beat_times) < 2:
            logger.warning("Not enough data to calculate syncopation")
            return {
                "syncopation_score": 0.0,
                "syncopated_onsets_count": 0,
                "total_onsets": 0,
                "syncopation_ratio": 0.0,
            }

        beat_interval = np.mean(np.diff(beat_times))
        half_beat = beat_interval / 2.0
        quarter_beat = beat_interval / 4.0

        syncopated_count = 0
        total_analyzed = 0

        for onset in onset_times:
            distances_to_beats = np.abs(beat_times - onset)
            closest_beat_idx = np.argmin(distances_to_beats)
            distance_to_closest_beat = distances_to_beats[closest_beat_idx]

            if distance_to_closest_beat < quarter_beat:
                total_analyzed += 1
                continue

            if distance_to_closest_beat > half_beat:
                syncopated_count += 1
                total_analyzed += 1
            elif distance_to_closest_beat > quarter_beat:
                next_beat_idx = closest_beat_idx + 1
                prev_beat_idx = closest_beat_idx - 1

                is_between_beats = False
                if next_beat_idx < len(beat_times) and prev_beat_idx >= 0:
                    midpoint = (beat_times[closest_beat_idx] + beat_times[next_beat_idx]) / 2.0
                    distance_to_midpoint = abs(onset - midpoint)
                    if distance_to_midpoint < quarter_beat:
                        is_between_beats = True

                if is_between_beats:
                    syncopated_count += 1
                total_analyzed += 1

        syncopation_ratio = (
            syncopated_count / total_analyzed if total_analyzed > 0 else 0.0
        )
        syncopation_score = syncopation_ratio

        logger.debug(
            f"Syncopation calculated: score={syncopation_score:.3f}, "
            f"syncopated={syncopated_count}/{total_analyzed}"
        )

        return {
            "syncopation_score": float(syncopation_score),
            "syncopated_onsets_count": int(syncopated_count),
            "total_onsets": int(total_analyzed),
            "syncopation_ratio": float(syncopation_ratio),
        }

    except Exception as e:
        logger.error(f"Error calculating syncopation: {e}")
        raise


def calculate_micro_timing(
    onset_times: npt.NDArray[np.float64],
    beat_times: npt.NDArray[np.float64],
    tempo: float,
) -> dict:
    """Analyzes micro-timing deviations of onsets from the theoretical beat grid."""
    try:
        if len(onset_times) == 0 or len(beat_times) < 2:
            logger.warning("Not enough data to calculate micro-timing")
            return {
                "mean_deviation_ms": 0.0,
                "std_deviation_ms": 0.0,
                "max_early_ms": 0.0,
                "max_late_ms": 0.0,
                "deviations_ms": [],
            }

        beat_interval = np.mean(np.diff(beat_times))
        deviations_ms = []

        for onset in onset_times:
            distances_to_beats = np.abs(beat_times - onset)
            closest_beat_idx = np.argmin(distances_to_beats)
            distance_to_closest_beat = distances_to_beats[closest_beat_idx]

            if onset < beat_times[closest_beat_idx]:
                deviation_ms = -distance_to_closest_beat * 1000.0
            else:
                deviation_ms = distance_to_closest_beat * 1000.0

            if abs(deviation_ms) < beat_interval * 500.0:
                deviations_ms.append(deviation_ms)

        if len(deviations_ms) == 0:
            return {
                "mean_deviation_ms": 0.0,
                "std_deviation_ms": 0.0,
                "max_early_ms": 0.0,
                "max_late_ms": 0.0,
                "deviations_ms": [],
            }

        deviations_array = np.array(deviations_ms)
        mean_deviation = float(np.mean(deviations_array))
        std_deviation = float(np.std(deviations_array))
        max_early = float(np.min(deviations_array))
        max_late = float(np.max(deviations_array))

        logger.debug(
            f"Micro-timing calculated: mean={mean_deviation:.2f}ms, "
            f"std={std_deviation:.2f}ms"
        )

        return {
            "mean_deviation_ms": mean_deviation,
            "std_deviation_ms": std_deviation,
            "max_early_ms": max_early,
            "max_late_ms": max_late,
            "deviations_ms": deviations_array.tolist(),
        }

    except Exception as e:
        logger.error(f"Error calculating micro-timing: {e}")
        raise


def analyze_rhythm(
    onset_times: npt.NDArray[np.float64],
    beat_times: npt.NDArray[np.float64],
    tempo: float,
    time_signature: tuple[int, int],
) -> dict:
    """Performs complete rhythm analysis including swing, syncopation, and micro-timing."""
    try:
        swing = calculate_swing(onset_times, beat_times, tempo)
        syncopation = calculate_syncopation(
            onset_times, beat_times, tempo, time_signature
        )
        micro_timing = calculate_micro_timing(onset_times, beat_times, tempo)

        return {
            "swing": swing,
            "syncopation": syncopation,
            "micro_timing": micro_timing,
        }

    except Exception as e:
        logger.error(f"Error in rhythm analysis: {e}")
        raise
