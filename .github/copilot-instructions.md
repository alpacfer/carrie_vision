# Copilot Instructions

## Project Overview
- Single-purpose image pipeline for extracting a centered square ROI.
- Primary entry point: [data/scripts/extract_center_roi.py](data/scripts/extract_center_roi.py).
- Core helpers live in [src/buttonvision/image_helpers.py](src/buttonvision/image_helpers.py).
- Inputs default to [data/raw](data/raw); outputs go to [data/processed/roi](data/processed/roi).

## How Data Flows
- Script resolves `project_root`, collects image paths, loads images, then crops to a centered square.
- Outputs are the cropped images written to the output directory.

## Run and Verify
- Install dependencies: `python -m pip install -r requirements.txt`.
- Verify run: `python data/scripts/extract_center_roi.py --side 300 --max-images 1`.
- Full run: `python data/scripts/extract_center_roi.py --side 300`.

## Project Conventions
- `data/scripts/extract_center_roi.py` is designed to run directly and inserts `src/` into `sys.path`.
- ROI handling is shared via `crop_images_to_roi()`; use it rather than custom per-script cropping.
- Saving outputs is standardized via `save_images`.

## Dependencies and Integration
- Uses `opencv-python-headless` and `numpy`; no other runtime deps are required.
- No test suite is present; rely on the verify command and produced outputs.
