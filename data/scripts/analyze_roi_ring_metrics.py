from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Make src importable when running this script directly
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FROM_SCRIPT = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_ROOT_FROM_SCRIPT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from carrievision.image_helpers import find_project_root, list_image_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze LED ring coverage using ROI images with scale-invariant metrics."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--roi-dir", type=Path, default=None, help="Directory of ROI images to analyze.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Raw image directory used to verify every raw image has a corresponding ROI.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional output dir for debug overlays with annulus and lit-angle sectors.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional CSV path for per-image metrics.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.95,
        help="Pass threshold for angular coverage.",
    )
    parser.add_argument(
        "--min-colored-ratio",
        type=float,
        default=0.85,
        help="Pass threshold for colored_ratio.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _candidate_centers(img_bgr: np.ndarray) -> List[Tuple[float, float, float]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=80,
        param1=120,
        param2=28,
        minRadius=40,
        maxRadius=min(img_bgr.shape[:2]) // 2,
    )
    if circles is None:
        return []

    unique: List[Tuple[float, float, float]] = []
    for cx, cy, r in circles[0]:
        keep = True
        for ux, uy, _ in unique:
            if (cx - ux) ** 2 + (cy - uy) ** 2 < 20 ** 2:
                keep = False
                break
        if keep:
            unique.append((float(cx), float(cy), float(r)))
    return unique


def _ring_from_center_sat_profile(
    img_bgr: np.ndarray, cx: float, cy: float, rmax: int
) -> Optional[Tuple[int, int, float]]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]

    h, w = sat.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    rmax = min(rmax, int(min(h, w) / 2) - 2)
    if rmax <= 15:
        return None

    prof = np.zeros(rmax + 1, dtype=np.float32)
    for r in range(10, rmax + 1):
        vals = sat[(rr >= r) & (rr < r + 1)]
        if vals.size:
            prof[r] = np.median(vals)

    prof = np.convolve(prof, np.ones(5) / 5, mode="same")
    dp = np.diff(prof)

    lo, hi = 15, rmax - 2
    pos_edges = np.where(dp[lo:hi] > 2.5)[0] + lo
    neg_edges = np.where(dp[lo:hi] < -2.5)[0] + lo

    best: Optional[Tuple[int, int, float]] = None
    best_score = -1e9

    for r_in in pos_edges:
        candidates_out = neg_edges[(neg_edges > r_in + 6) & (neg_edges < r_in + 90)]
        if candidates_out.size == 0:
            continue

        for r_out in candidates_out:
            ring_mean = float(np.mean(prof[r_in : r_out + 1]))
            bg_in = np.mean(prof[max(5, r_in - 20) : max(6, r_in - 5)])
            bg_out = np.mean(prof[min(rmax, r_out + 5) : min(rmax, r_out + 20)])
            bg_mean = float(0.5 * (bg_in + bg_out))

            contrast = ring_mean - bg_mean
            if contrast < 2:
                continue

            rise = float(dp[r_in])
            fall = float(-dp[r_out])
            score = 0.6 * contrast + 0.2 * rise + 0.2 * fall

            thickness = r_out - r_in
            if thickness < 12 or thickness > 70:
                score -= 4

            if score > best_score:
                best_score = score
                best = (int(r_in), int(r_out), score)

    return best


def _circular_mean_hue(hue_0_179: np.ndarray) -> float:
    if hue_0_179.size == 0:
        return float("nan")
    ang = hue_0_179 / 180.0 * 2.0 * np.pi
    mean_ang = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
    if mean_ang < 0:
        mean_ang += 2.0 * np.pi
    return float(mean_ang / (2.0 * np.pi) * 180.0)


def _color_label_from_hue(h: float) -> str:
    if math.isnan(h):
        return "unknown"
    if h < 8 or h >= 170:
        return "red"
    if 8 <= h < 25:
        return "orange"
    if 25 <= h < 35:
        return "yellow"
    if 35 <= h < 95:
        return "green"
    if 95 <= h < 130:
        return "cyan/blue"
    return "magenta"


def analyze_ring(img_bgr: np.ndarray) -> Dict[str, float | int | str | Tuple[float, float] | np.ndarray]:
    h, w = img_bgr.shape[:2]
    target = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    centers = _candidate_centers(img_bgr)
    if not centers:
        raise RuntimeError("No circle candidates found.")

    best = None
    best_total = -1e9
    rmax = int(min(h, w) * 0.48)

    for cx, cy, _ in centers:
        ring = _ring_from_center_sat_profile(img_bgr, cx, cy, rmax=rmax)
        if ring is None:
            continue
        r_in, r_out, score = ring
        pos_penalty = np.linalg.norm(np.array([cx, cy]) - target) / max(min(h, w) * 0.6, 1)
        total = score - 0.8 * pos_penalty
        if total > best_total:
            best_total = total
            best = (cx, cy, r_in, r_out)

    if best is None:
        raise RuntimeError("Ring not found from candidate centers.")

    cx, cy, r_in, r_out = best

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]

    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring_mask = (rr >= r_in) & (rr <= r_out)

    bg_mask = ((rr >= max(0, r_in - 20)) & (rr < max(1, r_in - 5))) | (
        (rr > r_out + 5) & (rr <= r_out + 20)
    )

    ring_sat = sat[ring_mask]
    bg_sat = sat[bg_mask]
    if ring_sat.size == 0 or bg_sat.size == 0:
        raise RuntimeError("Insufficient pixels to estimate thresholds.")

    sat_thr = np.percentile(bg_sat, 95)
    sat_thr = min(sat_thr, np.percentile(ring_sat, 90))
    sat_thr = max(sat_thr, np.percentile(bg_sat, 70))

    color_mask = (sat > sat_thr) & ring_mask

    colored_ratio = float(np.count_nonzero(color_mask) / np.count_nonzero(ring_mask))

    theta = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
    bins = 360
    idx = np.floor(theta / (2 * np.pi) * bins).astype(np.int32)

    sums = np.zeros(bins, dtype=np.float64)
    cnts = np.zeros(bins, dtype=np.int32)
    ridx = np.where(ring_mask)
    np.add.at(sums, idx[ridx], sat[ridx])
    np.add.at(cnts, idx[ridx], 1)
    mean_sat_by_angle = sums / np.maximum(cnts, 1)

    angular_coverage = float(np.mean(mean_sat_by_angle > sat_thr))

    dominant_hue = _circular_mean_hue(hue[color_mask])
    color_label = _color_label_from_hue(dominant_hue)

    return {
        "center": (float(cx), float(cy)),
        "ring_r_in_px": int(r_in),
        "ring_r_out_px": int(r_out),
        "sat_threshold": float(sat_thr),
        "colored_ratio": colored_ratio,
        "angular_coverage": angular_coverage,
        "dominant_hue_cv": float(dominant_hue),
        "color_label": color_label,
        "ring_mask": ring_mask,
        "color_mask": color_mask,
        "mean_sat_by_angle": mean_sat_by_angle,
    }


def _render_debug_overlay(img_bgr: np.ndarray, metrics: Dict[str, object], out_path: Path) -> None:
    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]

    cx, cy = metrics["center"]  # type: ignore[index]
    r_in = int(metrics["ring_r_in_px"])  # type: ignore[arg-type]
    r_out = int(metrics["ring_r_out_px"])  # type: ignore[arg-type]
    sat_thr = float(metrics["sat_threshold"])  # type: ignore[arg-type]
    angular_coverage = float(metrics["angular_coverage"])  # type: ignore[arg-type]
    colored_ratio = float(metrics["colored_ratio"])  # type: ignore[arg-type]
    color_label = str(metrics["color_label"])
    mean_sat_by_angle = np.asarray(metrics["mean_sat_by_angle"])

    cv2.circle(overlay, (int(round(cx)), int(round(cy))), r_in, (255, 255, 0), 2)
    cv2.circle(overlay, (int(round(cx)), int(round(cy))), r_out, (0, 255, 255), 2)

    bins = mean_sat_by_angle.shape[0]
    for i in range(bins):
        if mean_sat_by_angle[i] <= sat_thr:
            continue
        ang = (i + 0.5) / bins * 2.0 * np.pi
        x1 = int(round(cx + r_in * np.cos(ang)))
        y1 = int(round(cy + r_in * np.sin(ang)))
        x2 = int(round(cx + r_out * np.cos(ang)))
        y2 = int(round(cy + r_out * np.sin(ang)))
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)

    text_lines = [
        f"color={color_label}",
        f"colored_ratio={colored_ratio:.3f}",
        f"angular_coverage={angular_coverage:.3f}",
    ]
    y = 24
    for line in text_lines:
        cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), overlay)
    if not ok:
        raise RuntimeError(f"Failed to save debug overlay: {out_path}")


def _write_csv(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image",
        "status",
        "color_label",
        "colored_ratio",
        "angular_coverage",
        "ring_r_in_px",
        "ring_r_out_px",
        "sat_threshold",
        "center_x",
        "center_y",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    project_root = args.project_root or find_project_root(SCRIPT_DIR)
    roi_dir = args.roi_dir or (project_root / "data" / "processed" / "roi")
    raw_dir = args.raw_dir or (project_root / "data" / "raw")

    roi_paths = list_image_paths(roi_dir)
    raw_paths = list_image_paths(raw_dir)

    if not roi_paths:
        raise RuntimeError(f"No ROI images found in: {roi_dir}")
    if not raw_paths:
        raise RuntimeError(f"No raw images found in: {raw_dir}")

    roi_name_set = {p.name for p in roi_paths}
    expected_roi_names = {f"{p.stem}_roi{p.suffix.lower()}" for p in raw_paths}
    missing_roi = sorted(expected_roi_names - roi_name_set)

    rows: List[Dict[str, object]] = []
    passed = 0
    failed = 0

    for roi_path in roi_paths:
        img = cv2.imread(str(roi_path), cv2.IMREAD_COLOR)
        if img is None:
            rows.append({
                "image": roi_path.name,
                "status": "error",
                "error": "Failed to read image",
            })
            failed += 1
            continue

        try:
            result = analyze_ring(img)
            coverage = float(result["angular_coverage"])
            ratio = float(result["colored_ratio"])
            status = "pass" if (coverage >= args.min_coverage and ratio >= args.min_colored_ratio) else "fail"

            if status == "pass":
                passed += 1
            else:
                failed += 1

            center_x, center_y = result["center"]  # type: ignore[misc]
            rows.append(
                {
                    "image": roi_path.name,
                    "status": status,
                    "color_label": result["color_label"],
                    "colored_ratio": f"{ratio:.6f}",
                    "angular_coverage": f"{coverage:.6f}",
                    "ring_r_in_px": result["ring_r_in_px"],
                    "ring_r_out_px": result["ring_r_out_px"],
                    "sat_threshold": f"{float(result['sat_threshold']):.3f}",
                    "center_x": f"{float(center_x):.2f}",
                    "center_y": f"{float(center_y):.2f}",
                    "error": "",
                }
            )

            if args.debug_dir:
                _render_debug_overlay(
                    img_bgr=img,
                    metrics=result,
                    out_path=args.debug_dir / f"{roi_path.stem}_debug.jpg",
                )

        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "image": roi_path.name,
                    "status": "error",
                    "color_label": "",
                    "colored_ratio": "",
                    "angular_coverage": "",
                    "ring_r_in_px": "",
                    "ring_r_out_px": "",
                    "sat_threshold": "",
                    "center_x": "",
                    "center_y": "",
                    "error": str(exc),
                }
            )
            failed += 1

    print(f"Project root: {project_root}")
    print(f"Raw images:   {len(raw_paths)}")
    print(f"ROI images:   {len(roi_paths)}")
    print(f"Missing ROI:  {len(missing_roi)}")
    if missing_roi:
        print("Missing ROI file(s):")
        for name in missing_roi:
            print(f"  - {name}")

    print("\nPer-image metrics:")
    print("image,status,color,colored_ratio,angular_coverage")
    for row in rows:
        print(
            f"{row.get('image','')},{row.get('status','')},{row.get('color_label','')},"
            f"{row.get('colored_ratio','')},{row.get('angular_coverage','')}"
        )

    print("\nSummary:")
    print(f"  Pass: {passed}")
    print(f"  Fail/Error: {failed}")

    if args.csv_path:
        _write_csv(args.csv_path, rows)
        print(f"CSV written: {args.csv_path}")


if __name__ == "__main__":
    main()
