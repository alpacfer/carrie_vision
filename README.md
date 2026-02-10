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
python data/scripts/extract_center_roi.py --side 1000 --max-images 1
```

You should see a summary and new outputs under `data/processed/roi`.

## Run ROI Extraction

Basic run (uses `data/raw` input and writes to `data/processed/roi`):

```bash
python data/scripts/extract_center_roi.py --side 1000
```

## Command-Line Arguments

Optional paths:

- `--project-root` (path): Project root override.
- `--input-dir` (path): Input directory override.
- `--output-dir` (path): Output directory override.

Input selection:

- `--side` (int): Side length in pixels for the centered square crop (default: 1000).
- `--max-images` (int): Limit number of images processed (0 = all).

## Analyze ROI Ring Coverage

Run ring analysis on ROI images and verify coverage against all raw inputs:

```bash
python data/scripts/analyze_roi_ring_metrics.py \
  --csv-path data/processed/ring_metrics.csv \
  --debug-dir data/processed/ring_debug
```

This computes:

- `colored_ratio = colored_pixels_in_ring / ring_pixels`
- `angular_coverage = lit_angle_bins / 360`

It also checks whether each file in `data/raw` has a corresponding `_roi` image.
