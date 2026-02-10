from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

# Make src importable when this helper is imported directly.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FROM_SCRIPT = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_ROOT_FROM_SCRIPT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from buttonvision.image_helpers import (
    crop_images_to_roi,
    find_project_root,
    list_image_paths,
    load_images,
    save_images,
)


def ensure_src_on_path(script_dir: Path) -> None:
    project_root_from_script = script_dir.parent.parent
    src_dir = project_root_from_script / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop each image to a centered square ROI and save the crops."
    )

    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument(
        "--side",
        type=int,
        default=1000,
        help="Side length in pixels for the centered square (default: 1000).",
    )
    parser.add_argument(
        "--max-images", type=int, default=0, help="Limit number of images processed (0 = all)."
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default=None,
        help="Process a single image by file stem (no extension).",
    )

    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args(argv)


def resolve_paths(
    script_dir: Path,
    project_root: Optional[Path],
    input_dir: Optional[Path],
    output_dir: Optional[Path],
) -> tuple[Path, Path, Path]:
    resolved_project_root = project_root or find_project_root(script_dir)
    resolved_input_dir = input_dir or (resolved_project_root / "data" / "raw")
    resolved_output_dir = output_dir or (resolved_project_root / "data" / "processed" / "roi")
    return resolved_project_root, resolved_input_dir, resolved_output_dir


def validate_side(side: int) -> None:
    if side <= 0:
        raise ValueError("--side must be > 0")


def _filter_image_paths_by_name(
    image_paths: list[Path],
    image_name: str,
    input_dir: Path,
) -> list[Path]:
    matches = [p for p in image_paths if p.stem == image_name]
    if not matches:
        raise RuntimeError(f"No image named '{image_name}' found in: {input_dir}")
    return matches


def collect_image_paths(input_dir: Path, max_images: int, image_name: Optional[str]) -> list[Path]:
    image_paths = list_image_paths(input_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_dir}")

    if image_name:
        return _filter_image_paths_by_name(image_paths, image_name, input_dir)

    if max_images and max_images > 0:
        image_paths = image_paths[:max_images]

    return image_paths


def load_and_crop_images(image_paths: list[Path], side: int) -> list:
    images = load_images(image_paths)
    cropped, _used_roi = crop_images_to_roi(images=images, center_square_side=side)
    return cropped


def save_cropped_images(cropped_images: list, image_paths: list[Path], output_dir: Path) -> None:
    save_images(cropped_images, image_paths, output_dir, suffix="_roi")


def print_summary(project_root: Path, input_dir: Path, output_dir: Path, image_count: int) -> None:
    print(f"\nProject root: {project_root}")
    print(f"Input dir:    {input_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Images:       {image_count}")
