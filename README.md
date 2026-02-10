# carrie_vision

## Setup

Install dependencies (Python 3.10+ recommended):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify Install

Run a short ROI crop to confirm imports, image loading, and output writing:

```bash
python data/scripts/extract_center_roi.py --side 300 --max-images 1
```

You should see a summary and new outputs under `data/processed/roi`.

## Run ROI Extraction

Basic run (uses `data/raw` input and writes to `data/processed/roi`):

```bash
python data/scripts/extract_center_roi.py --side 300
```

## Command-Line Arguments

Required:

- `--side` (int): Side length in pixels for the centered square crop.

Optional paths:

- `--project-root` (path): Project root override.
- `--input-dir` (path): Input directory override.
- `--output-dir` (path): Output directory override.

Input selection:

- `--recursive`: Scan input directory recursively.
- `--dry-run`: Resolve paths and list inputs without processing.
- `--max-images` (int): Limit number of images processed (0 = all).