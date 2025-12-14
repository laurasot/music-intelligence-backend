#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_processing.audio_loader import load_audio
from audio_processing.feature_extraction import extract_all_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_array_metadata(array: np.ndarray) -> dict:
    """Generates metadata for a numpy array (shape, statistics)."""
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "size": int(array.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extracts features from audio files"
    )
    parser.add_argument("audio_path", type=str, help="Path to the audio file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to the output JSON file (if not specified, prints to stdout)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    audio_path = Path(args.audio_path)

    try:
        logger.info(f"Loading audio: {audio_path}")
        audio, sample_rate = load_audio(audio_path)

        logger.info("Extracting features...")
        features = extract_all_features(audio, sample_rate)

        results = {
            "audio_file": str(audio_path),
            "sample_rate": sample_rate,
            "duration_seconds": len(audio) / sample_rate,
            "tempo": float(features["tempo"]),
            "time_signature": {
                "numerator": int(features["time_signature"][0]),
                "denominator": int(features["time_signature"][1]),
            },
            "num_beats": int(len(features["beat_times"])),
            "num_onsets": int(len(features["onset_times"])),
            "energy": features["energy"],
            "brightness": features["brightness"],
            "spectral_density": features["spectral_density"],
            "mfccs": get_array_metadata(features["mfccs"]),
            "log_magnitude_spectrogram": get_array_metadata(
                features["log_magnitude_spectrogram"]
            ),
            "cqt": get_array_metadata(features["cqt"]),
            "note_durations": {
                "durations_seconds": features["note_durations"]["durations"],
                "mean_seconds": features["note_durations"]["mean"],
                "median_seconds": features["note_durations"]["median"],
                "std_seconds": features["note_durations"]["std"],
                "min_seconds": features["note_durations"]["min"],
                "max_seconds": features["note_durations"]["max"],
                "percentile_25_seconds": features["note_durations"]["percentile_25"],
                "percentile_75_seconds": features["note_durations"]["percentile_75"],
                "long_notes_count": features["note_durations"]["long_notes_count"],
                "short_notes_count": features["note_durations"]["short_notes_count"],
                "long_to_short_ratio": features["note_durations"]["long_to_short_ratio"],
                "threshold_seconds": features["note_durations"]["threshold_seconds"],
            },
            "rhythm_analysis": {
                "swing": features["rhythm_analysis"]["swing"],
                "syncopation": features["rhythm_analysis"]["syncopation"],
                "micro_timing": {
                    "mean_deviation_ms": features["rhythm_analysis"]["micro_timing"]["mean_deviation_ms"],
                    "std_deviation_ms": features["rhythm_analysis"]["micro_timing"]["std_deviation_ms"],
                    "max_early_ms": features["rhythm_analysis"]["micro_timing"]["max_early_ms"],
                    "max_late_ms": features["rhythm_analysis"]["micro_timing"]["max_late_ms"],
                    "deviations_ms": features["rhythm_analysis"]["micro_timing"]["deviations_ms"],
                },
            },
        }

        output_json = json.dumps(results, indent=2, ensure_ascii=False)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_json, encoding="utf-8")
            logger.info(f"Results saved to: {output_path}")
        else:
            print(output_json)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Error processing audio: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
