from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: capture image, extract centered ROI, then compute ring metrics."
    )
    parser.add_argument(
        "--side",
        type=int,
        default=1000,
        help="ROI square side length in pixels (default: 1000).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("full_pipeline"),
        help="Root output directory for full pipeline artifacts (default: full_pipeline).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Optional raw output directory override (default: <output-root>/raw).",
    )
    parser.add_argument(
        "--roi-dir",
        type=Path,
        default=None,
        help="Optional ROI output directory override (default: <output-root>/roi).",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional debug output directory override (default: <output-root>/ring_debug).",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional CSV output path override (default: <output-root>/metrics/ring_metrics.csv).",
    )
    return parser


def run_command(cmd: list[str], project_root: Path, step_name: str) -> str:
    print(f"\n[{step_name}] Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {result.returncode}")

    return result.stdout


def parse_capture_path(capture_stdout: str) -> Path:
    match = re.search(r"Saved image to:\s*(.+)", capture_stdout)
    if not match:
        raise RuntimeError("Could not find saved image path in capture output.")
    return Path(match.group(1).strip())


def find_debug_image_path(debug_dir: Path, image_name: str) -> Path | None:
    candidates = [
        debug_dir / f"{image_name}_roi_debug.jpg",
        debug_dir / f"{image_name}_roi_debug.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = build_arg_parser().parse_args()

    project_root = Path(__file__).resolve().parent
    python_exe = sys.executable

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()

    raw_dir = args.raw_dir if args.raw_dir is not None else output_root / "raw"
    roi_dir = args.roi_dir if args.roi_dir is not None else output_root / "roi"
    debug_dir = args.debug_dir if args.debug_dir is not None else output_root / "ring_debug"
    csv_path = args.csv_path if args.csv_path is not None else output_root / "metrics" / "ring_metrics.csv"

    if not raw_dir.is_absolute():
        raw_dir = (project_root / raw_dir).resolve()
    if not roi_dir.is_absolute():
        roi_dir = (project_root / roi_dir).resolve()
    if not debug_dir.is_absolute():
        debug_dir = (project_root / debug_dir).resolve()
    if not csv_path.is_absolute():
        csv_path = (project_root / csv_path).resolve()

    raw_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    capture_script = project_root / "data" / "scripts" / "capture_raw_camera_image.py"
    extract_script = project_root / "data" / "scripts" / "extract_center_roi.py"
    metrics_script = project_root / "data" / "scripts" / "analyze_roi_ring_metrics.py"

    capture_stdout = run_command(
        [python_exe, str(capture_script), "--output-dir", str(raw_dir)],
        project_root=project_root,
        step_name="Capture",
    )
    captured_image_path = parse_capture_path(capture_stdout)
    image_name = captured_image_path.stem

    run_command(
        [
            python_exe,
            str(extract_script),
            "--input-dir",
            str(raw_dir),
            "--output-dir",
            str(roi_dir),
            "--side",
            str(args.side),
            "--image-name",
            image_name,
        ],
        project_root=project_root,
        step_name="ROI Extraction",
    )

    run_command(
        [
            python_exe,
            str(metrics_script),
            "--raw-dir",
            str(raw_dir),
            "--roi-dir",
            str(roi_dir),
            "--csv-path",
            str(csv_path),
            "--debug-dir",
            str(debug_dir),
            "--image-name",
            image_name,
        ],
        project_root=project_root,
        step_name="Metrics",
    )
    debug_image_path = find_debug_image_path(debug_dir, image_name)

    print("\nPipeline succeeded.")
    print(f"Output root:      {output_root}")
    print(f"Captured image: {captured_image_path}")
    print(f"Raw directory:    {raw_dir}")
    print(f"ROI directory:    {roi_dir}")
    print(f"Debug directory:  {debug_dir}")
    if debug_image_path is not None:
        print(f"Debug image:      {debug_image_path}")
    else:
        print("Debug image:      not found")
    print(f"Metrics CSV:      {csv_path}")


if __name__ == "__main__":
    main()
