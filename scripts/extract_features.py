#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Import modules directly to avoid __init__.py import issues
import importlib.util
src_path = Path(__file__).parent.parent / "src"

# Load config first (needed by audio_loader)
spec_config = importlib.util.spec_from_file_location(
    "config",
    src_path / "audio_processing" / "config.py"
)
config_module = importlib.util.module_from_spec(spec_config)
spec_config.loader.exec_module(config_module)

# Make config available as audio_processing.config
if "audio_processing" not in sys.modules:
    import types
    audio_processing_pkg = types.ModuleType("audio_processing")
    sys.modules["audio_processing"] = audio_processing_pkg
sys.modules["audio_processing.config"] = config_module

# Load rhythm_analysis (needed by feature_extraction)
spec_rhythm_analysis = importlib.util.spec_from_file_location(
    "rhythm_analysis",
    src_path / "audio_processing" / "rhythm_analysis.py"
)
rhythm_analysis_module = importlib.util.module_from_spec(spec_rhythm_analysis)
spec_rhythm_analysis.loader.exec_module(rhythm_analysis_module)
sys.modules["audio_processing.rhythm_analysis"] = rhythm_analysis_module

# Load audio_loader
spec_loader = importlib.util.spec_from_file_location(
    "audio_loader",
    src_path / "audio_processing" / "audio_loader.py"
)
audio_loader = importlib.util.module_from_spec(spec_loader)
spec_loader.loader.exec_module(audio_loader)
load_audio = audio_loader.load_audio

# Load feature_extraction
spec_features = importlib.util.spec_from_file_location(
    "feature_extraction",
    src_path / "audio_processing" / "feature_extraction.py"
)
feature_extraction = importlib.util.module_from_spec(spec_features)
spec_features.loader.exec_module(feature_extraction)
extract_all_features = feature_extraction.extract_all_features

# Load rhythm interpreter
spec_rhythm = importlib.util.spec_from_file_location(
    "rhythm_interpreter",
    src_path / "rhythmic_analysis" / "rhythm_interpreter.py"
)
rhythm_interpreter = importlib.util.module_from_spec(spec_rhythm)
spec_rhythm.loader.exec_module(rhythm_interpreter)

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
        help="Path to the output JSON file (default: <audio_name>_features.json)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print results to stdout instead of saving to file",
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
            "rhythm_interpretation": rhythm_interpreter.rhythm_interpretation_to_dict(
                rhythm_interpreter.interpret_rhythm(features["rhythm_analysis"])
            ),
        }

        output_json = json.dumps(results, indent=2, ensure_ascii=False)

        if args.stdout:
            print(output_json)
        else:
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = audio_path.with_name(f"{audio_path.stem}_features.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_json, encoding="utf-8")
            logger.info(f"Results saved to: {output_path}")

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
