from __future__ import annotations

from pathlib import Path

from extract_center_roi_helpers import (
    collect_image_paths,
    ensure_src_on_path,
    load_and_crop_images,
    parse_args,
    print_summary,
    resolve_paths,
    save_cropped_images,
    validate_side,
)


# Make src importable when running this script directly
SCRIPT_DIR = Path(__file__).resolve().parent
ensure_src_on_path(SCRIPT_DIR)


def main() -> None:
    args = parse_args()

    # Step 1: Resolve paths for the project, input, and output.
    project_root, input_dir, output_dir = resolve_paths(
        SCRIPT_DIR, args.project_root, args.input_dir, args.output_dir
    )

    # Step 2: Validate CLI inputs.
    validate_side(args.side)

    # Step 3: Collect input images (optionally limiting count).
    image_paths = collect_image_paths(input_dir, args.max_images, args.image_name)

    # Step 4: Load images and crop to the centered square ROI.
    cropped_images = load_and_crop_images(image_paths, args.side)

    # Step 5: Save the cropped images.
    save_cropped_images(cropped_images, image_paths, output_dir)

    # Step 6: Print a summary of what was processed.
    print_summary(project_root, input_dir, output_dir, len(image_paths))


if __name__ == "__main__":
    main()
