# Music Intelligence Backend

Professional music analysis system that extracts detailed information from audio files.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Usage

### Extract basic and global features

```bash
python scripts/extract_features.py <audio_path> -o output.json
```

The script extracts:
- **Basic features**: tempo (BPM), beats, approximate time signature
- **Global features**: energy, spectral brightness, spectral density

## Project Structure

```
src/music_intelligence/
├── config/          # Configuration and constants
├── dsp/             # Digital signal processing
├── features/        # Feature extraction
└── utils/           # General utilities
```

## Development

```bash
# Format code
black src/ scripts/

# Sort imports
isort src/ scripts/

# Linter
ruff check src/ scripts/

# Tests
pytest
```

