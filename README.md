# carrie_vision

## Setup

Install dependencies (Python 3.10+ recommended):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify Install

Run a short pipeline pass to confirm imports, image loading, and output writing:

```bash
python data/scripts/run_circle_pipeline.py --min-radius 200 --max-radius 500 --max-images 1
```

You should see a summary and new outputs under `data/processed/circles`.

## Run Pipeline

Basic run (uses `data/raw` input and writes to `data/processed/circles`):

```bash
python data/scripts/run_circle_pipeline.py --min-radius 200 --max-radius 500
```

## Command-Line Arguments

Required:

- `--min-radius` (int): Minimum circle radius in pixels.
- `--max-radius` (int): Maximum circle radius in pixels.

Optional paths:

- `--project-root` (path): Project root override.
- `--input-dir` (path): Input directory override.
- `--output-dir` (path): Output directory override.

Input selection:

- `--recursive`: Scan input directory recursively.
- `--dry-run`: Resolve paths and list inputs without processing.
- `--max-images` (int): Limit number of images processed (0 = all).

Hough parameters:

- `--dp` (float): Inverse ratio of accumulator resolution to image resolution.
- `--min-dist` (float): Minimum distance between detected centers (defaults to `min-radius`).
- `--param1` (float): Upper Canny threshold.
- `--param2` (float): Accumulator threshold.
- `--blur-ksize` (int): Gaussian blur kernel size.
- `--blur-sigma` (float): Gaussian blur sigma.

Cropping:

- `--center-crop-side` (int): Centered square crop side length (0 disables).
- `--roi X Y W H`: Explicit ROI (overrides center crop).
- `--save-crops`: Save cropped images when cropping is enabled.