# Music Intelligence Backend

Extracts musical features from audio files: tempo, beats, time signature, energy, brightness, and spectral density.

## Installation

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

## Usage

```bash
python scripts/extract_features.py audio_file.wav -o output.json
```

## Structure

```
src/
├── config.py      # Constants (sample rate, etc.)
├── audio.py       # Load audio files
└── features.py    # Extract all features

scripts/
└── extract_features.py  # Main script
```
