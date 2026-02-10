from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable when running this script directly
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FROM_SCRIPT = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_ROOT_FROM_SCRIPT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from carrievision.image_helpers import (
    HoughParams,
    crop_images_to_roi,
    detect_circles_hough_batch,
    find_project_root,
    list_image_paths,
    load_images,
    save_images,
    visualize_all_circles,
    write_detections_csv,
    write_detections_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect circles in images and save visualized output."
    )

    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--recursive", action="store_true", help="Scan input directory recursively.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths and list inputs without processing.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images processed (0 = all).")

    parser.add_argument("--min-radius", type=int, required=True)
    parser.add_argument("--max-radius", type=int, required=True)
    parser.add_argument("--dp", type=float, default=1.2)
    parser.add_argument("--min-dist", type=float, default=None)
    parser.add_argument("--param1", type=float, default=120.0)
    parser.add_argument("--param2", type=float, default=35.0)
    parser.add_argument("--blur-ksize", type=int, default=9)
    parser.add_argument("--blur-sigma", type=float, default=2.0)

    parser.add_argument(
        "--center-crop-side",
        type=int,
        default=0,
        help="If > 0, crop each image to centered square before detection.",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="Explicit ROI. Overrides center crop if provided.",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save cropped images when cropping is enabled.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve() if args.project_root else find_project_root()
    input_dir = args.input_dir.resolve() if args.input_dir else (project_root / "data" / "raw")
    output_root = args.output_dir.resolve() if args.output_dir else (
        project_root / "data" / "processed" / "circles"
    )

    image_paths = list_image_paths(input_dir, recursive=args.recursive)
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_dir}")
    if args.max_images and args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    images = load_images(image_paths)

    # Optional crop stage
    crop_enabled = (args.roi is not None) or (args.center_crop_side > 0)
    used_roi = None
    working_images = images

    if crop_enabled:
        roi_tuple = tuple(args.roi) if args.roi is not None else None
        side = args.center_crop_side if args.center_crop_side > 0 else min(images[0].shape[:2])

        working_images, used_roi = crop_images_to_roi(
            images=images,
            roi=roi_tuple,
            center_square_side=side,
        )

        if args.save_crops:
            crop_dir = output_root / "crops"
            save_images(working_images, image_paths, crop_dir, suffix="_crop")
            print(f"Saved crops to: {crop_dir}")

    params = HoughParams(
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        dp=args.dp,
        min_dist=args.min_dist,
        param1=args.param1,
        param2=args.param2,
        blur_ksize=args.blur_ksize,
        blur_sigma=args.blur_sigma,
    )

    all_circles = detect_circles_hough_batch(working_images, params=params)
    vis_images = visualize_all_circles(working_images, all_circles)

    overlays_dir = output_root / "overlays"
    save_images(vis_images, image_paths, overlays_dir, suffix="_circles")

    json_path = output_root / "detections.json"
    csv_path = output_root / "detections.csv"
    write_detections_json(image_paths, all_circles, json_path)
    write_detections_csv(image_paths, all_circles, csv_path)

    total_circles = sum(len(c) for c in all_circles)

    print(f"\nProject root: {project_root}")
    print(f"Input dir:    {input_dir}")
    print(f"Output dir:   {output_root}")
    if used_roi is not None:
        print(f"Used ROI:     {used_roi} (x, y, w, h)")
    print(f"Images:       {len(image_paths)}")
    print(f"Total circles:{total_circles}")
    print(f"Overlays:     {overlays_dir}")
    print(f"JSON report:  {json_path}")
    print(f"CSV report:   {csv_path}")

    print("\nPer-image counts:")
    for p, circles in zip(image_paths, all_circles):
        print(f"  {p.name}: {len(circles)}")

    if args.dry_run:
        print("\nDry-run complete.")
        return


if __name__ == "__main__":
    main()
