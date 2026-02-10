from __future__ import annotations

from pathlib import Path

from analyze_roi_ring_metrics_helpers import (
    analyze_roi_images,
    find_missing_roi_names,
    load_image_paths,
    parse_args,
    print_report,
    resolve_paths,
    write_csv,
)

# Script location used for project root discovery.
SCRIPT_DIR = Path(__file__).resolve().parent




def main() -> None:
    args = parse_args()

    # Step 1: Resolve project, ROI, and raw paths.
    project_root, roi_dir, raw_dir = resolve_paths(
        SCRIPT_DIR, args.project_root, args.roi_dir, args.raw_dir
    )

    # Step 2: Load ROI and raw image paths.
    roi_paths, raw_paths = load_image_paths(roi_dir, raw_dir)

    # Step 3: Check for missing ROI outputs.
    missing_roi = find_missing_roi_names(roi_paths, raw_paths)

    # Step 4: Analyze ROI images and collect metrics.
    rows, passed, failed = analyze_roi_images(
        roi_paths=roi_paths,
        min_coverage=args.min_coverage,
        min_colored_ratio=args.min_colored_ratio,
        debug_dir=args.debug_dir,
    )

    # Step 5: Print report summary and per-image metrics.
    print_report(project_root, raw_paths, roi_paths, missing_roi, rows, passed, failed)

    # Step 6: Persist CSV output if requested.
    if args.csv_path:
        write_csv(args.csv_path, rows)
        print(f"CSV written: {args.csv_path}")


if __name__ == "__main__":
    main()
