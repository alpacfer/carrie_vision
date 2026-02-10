from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

# Make src importable when running this script directly
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FROM_SCRIPT = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_ROOT_FROM_SCRIPT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from carrievision.image_helpers import (
    crop_images_to_roi,
    find_project_root,
    list_image_paths,
    load_images,
    save_images,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop each image to a centered square ROI and save the crops."
    )

    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--side", type=int, required=True, help="Side length in pixels for the centered square.")
    parser.add_argument("--recursive", action="store_true", help="Scan input directory recursively.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images processed (0 = all).")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths and list inputs without processing.")

    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    project_root = args.project_root or find_project_root(SCRIPT_DIR)
    input_dir = args.input_dir or (project_root / "data" / "raw")
    output_dir = args.output_dir or (project_root / "data" / "processed" / "roi")

    if args.side <= 0:
        raise ValueError("--side must be > 0")

    image_paths = list_image_paths(input_dir, recursive=args.recursive)
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_dir}")

    if args.max_images and args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    if args.dry_run:
        print(f"Project root: {project_root}")
        print(f"Input dir:    {input_dir}")
        print(f"Output dir:   {output_dir}")
        print(f"Images:       {len(image_paths)}")
        print("\nInputs:")
        for p in image_paths:
            print(f"  {p.name}")
        print("\nDry-run complete.")
        return

    images = load_images(image_paths)
    cropped, used_roi = crop_images_to_roi(images=images, center_square_side=args.side)
    save_images(cropped, image_paths, output_dir, suffix="_roi")

    print(f"\nProject root: {project_root}")
    print(f"Input dir:    {input_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Images:       {len(image_paths)}")
    print(f"Used ROI:     {used_roi} (x, y, w, h)")


if __name__ == "__main__":
    main()
