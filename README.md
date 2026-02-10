# carrie_vision

This project crops a centered square ROI from images and then analyzes LED ring coverage on those ROI images.

## Quick Start (3 Steps)

1. Install Python (3.10 or newer).
2. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Run a small test (process only 1 image):

```bash
python data/scripts/extract_center_roi.py --side 1000 --max-images 1
```

If you see a summary printed and a new file in `data/processed/roi`, the setup works.

## What This Does

- **Step A: ROI extraction**
  - Reads raw images in `data/raw`.
  - Crops a centered square from each image.
  - Saves the cropped ROI images to `data/processed/roi`.

- **Step B: Ring analysis**
  - Reads ROI images from `data/processed/roi`.
  - Measures how much of the LED ring is lit.
  - Writes a CSV summary and optional debug overlays.

## Run ROI Extraction (Step A)

### Basic run

```bash
python data/scripts/extract_center_roi.py --side 1000
```

### Examples

- Process only 3 images:

```bash
python data/scripts/extract_center_roi.py --side 1000 --max-images 3
```

- Use a different input folder:

```bash
python data/scripts/extract_center_roi.py --input-dir data/raw_alt --side 800
```

- Use a different output folder:

```bash
python data/scripts/extract_center_roi.py --output-dir data/processed/roi_alt --side 800
```

### Parameters (ROI Extraction)

- `--side` (int): Size of the square crop in pixels. Example: `--side 1000`.
- `--max-images` (int): Limit how many images to process. Use `0` for all images.
- `--project-root` (path): Manually set the project root (rarely needed).
- `--input-dir` (path): Override the raw image folder (default is `data/raw`).
- `--output-dir` (path): Override the output folder (default is `data/processed/roi`).

### Output (ROI Extraction)

- Cropped images saved to `data/processed/roi`.
- Each file name ends with `_roi`.

## Run Ring Analysis (Step B)

### Basic run

```bash
python data/scripts/analyze_roi_ring_metrics.py \
  --csv-path data/processed/ring_metrics.csv \
  --debug-dir data/processed/ring_debug
```

### Examples

- Use a different ROI folder:

```bash
python data/scripts/analyze_roi_ring_metrics.py \
  --roi-dir data/processed/roi_alt \
  --csv-path data/processed/ring_metrics_alt.csv
```

- Skip debug overlays (CSV only):

```bash
python data/scripts/analyze_roi_ring_metrics.py --csv-path data/processed/ring_metrics.csv
```

### Parameters (Ring Analysis)

- `--roi-dir` (path): Folder with ROI images (default is `data/processed/roi`).
- `--raw-dir` (path): Folder with raw images (default is `data/raw`).
- `--csv-path` (path): Where to write the CSV results.
- `--debug-dir` (path): Optional folder for debug images.
- `--min-coverage` (float): Minimum angular coverage to pass. Default is `0.95`.
- `--min-colored-ratio` (float): Minimum colored ratio to pass. Default is `0.85`.
- `--project-root` (path): Manually set the project root (rarely needed).

### Output (Ring Analysis)

- A CSV file with one row per image and pass/fail results.
- Optional debug images showing the detected ring and lit angles.
- A printed summary with counts for pass, fail, and missing ROI files.

## Understanding the Results

- `colored_ratio` is the fraction of ring pixels that are colored.
- `angular_coverage` is how much of the ring is lit across 360 degrees.
- An image **passes** if both values meet the minimum thresholds.

## Troubleshooting

- **No images found**: Make sure your raw images are in `data/raw` (or use `--input-dir`).
- **ROI missing**: Run the ROI extraction step before the ring analysis step.
- **Errors reading images**: Make sure files are valid image formats (JPG/PNG).
