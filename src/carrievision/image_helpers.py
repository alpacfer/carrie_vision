from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import csv
import json

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

Circle = Tuple[int, int, int]   # (x, y, r)
Roi = Tuple[int, int, int, int] # (x, y, w, h)
ImageOrPath = Union[np.ndarray, str, Path]


@dataclass(frozen=True)
class HoughParams:
    min_radius: int
    max_radius: int
    dp: float = 1.2
    min_dist: Optional[float] = None
    param1: float = 120.0
    param2: float = 35.0
    blur_ksize: int = 9
    blur_sigma: float = 2.0

    def validate(self) -> None:
        if self.min_radius <= 0 or self.max_radius <= 0:
            raise ValueError("min_radius and max_radius must be > 0")
        if self.max_radius < self.min_radius:
            raise ValueError("max_radius must be >= min_radius")


def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Find project root by looking for .git or data/raw.
    Falls back to current working directory.
    """
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / ".git").exists() or (p / "data" / "raw").exists():
            return p
    return start


def list_image_paths(input_dir: Path, recursive: bool = False) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    paths = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    paths.sort()
    return paths


def imread_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def load_images(image_paths: Sequence[Path]) -> List[np.ndarray]:
    return [imread_bgr(p) for p in image_paths]


def _clip_roi_to_image(roi: Roi, w: int, h: int) -> Roi:
    x, y, rw, rh = roi

    x = max(0, int(x))
    y = max(0, int(y))
    rw = max(1, int(rw))
    rh = max(1, int(rh))

    rw = min(rw, w - x)
    rh = min(rh, h - y)

    if rw <= 0 or rh <= 0:
        raise ValueError(f"ROI {roi} is outside image bounds (w={w}, h={h})")

    return (x, y, rw, rh)


def _center_square_roi(img_w: int, img_h: int, side: int) -> Roi:
    side = int(side)
    if side <= 0:
        raise ValueError("center square side must be > 0")

    side = min(side, img_w, img_h)
    x = (img_w - side) // 2
    y = (img_h - side) // 2
    return (x, y, side, side)


def crop_images_to_roi(
    images: Sequence[np.ndarray],
    roi: Optional[Roi] = None,
    center_square_side: int = 256,
) -> Tuple[List[np.ndarray], Roi]:
    """
    Crop all images using the same base ROI.
    If roi is None, computes centered square from first image.
    """
    if not images:
        raise ValueError("images is empty")

    h0, w0 = images[0].shape[:2]
    base_roi = roi if roi is not None else _center_square_roi(w0, h0, center_square_side)

    cropped: List[np.ndarray] = []
    for img in images:
        h, w = img.shape[:2]
        x, y, rw, rh = _clip_roi_to_image(base_roi, w=w, h=h)
        cropped.append(img[y:y + rh, x:x + rw].copy())

    return cropped, base_roi


def detect_circles_hough_batch(
    images: Sequence[np.ndarray],
    params: HoughParams,
) -> List[List[Circle]]:
    params.validate()

    if not images:
        return []

    min_dist = float(params.min_radius) if params.min_dist is None else float(params.min_dist)

    k = int(params.blur_ksize)
    k = max(3, k)
    if k % 2 == 0:
        k += 1

    results: List[List[Circle]] = []

    for img_bgr in images:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (k, k), params.blur_sigma)

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=params.dp,
            minDist=min_dist,
            param1=params.param1,
            param2=params.param2,
            minRadius=int(params.min_radius),
            maxRadius=int(params.max_radius),
        )

        if circles is None:
            results.append([])
            continue

        circles = np.round(circles[0]).astype(int)
        detections = [(int(x), int(y), int(r)) for x, y, r in circles]
        results.append(detections)

    return results


def visualize_all_circles(
    images: Sequence[np.ndarray],
    all_circles: Sequence[Sequence[Circle]],
    circle_color: Tuple[int, int, int] = (255, 0, 0),  # BGR
    center_color: Tuple[int, int, int] = (0, 0, 255),  # BGR
    thickness: int = 2,
    center_radius: int = 2,
) -> List[np.ndarray]:
    if len(images) != len(all_circles):
        raise ValueError(
            f"images ({len(images)}) and all_circles ({len(all_circles)}) "
            "must have same length"
        )

    out_images: List[np.ndarray] = []
    for img, circles in zip(images, all_circles):
        vis = img.copy()
        for x, y, r in circles:
            cv2.circle(vis, (x, y), r, circle_color, thickness)
            cv2.circle(vis, (x, y), center_radius, center_color, -1)
        out_images.append(vis)

    return out_images


def save_images(
    images: Sequence[np.ndarray],
    source_paths: Sequence[Path],
    output_dir: Path,
    suffix: str = "",
    ext: Optional[str] = None,
) -> List[Path]:
    if len(images) != len(source_paths):
        raise ValueError("images and source_paths must have same length")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []

    for img, src in zip(images, source_paths):
        extension = ext if ext else src.suffix.lower()
        if not extension:
            extension = ".png"

        out_name = f"{src.stem}{suffix}{extension}"
        out_path = output_dir / out_name

        ok = cv2.imwrite(str(out_path), img)
        if not ok:
            raise RuntimeError(f"Failed to save image: {out_path}")

        out_paths.append(out_path)

    return out_paths


def write_detections_json(
    image_paths: Sequence[Path],
    all_circles: Sequence[Sequence[Circle]],
    output_json: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = []
    for p, circles in zip(image_paths, all_circles):
        payload.append(
            {
                "image": p.name,
                "count": len(circles),
                "circles": [{"x": int(x), "y": int(y), "r": int(r)} for x, y, r in circles],
            }
        )

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_detections_csv(
    image_paths: Sequence[Path],
    all_circles: Sequence[Sequence[Circle]],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "circle_idx", "x", "y", "r"])

        for p, circles in zip(image_paths, all_circles):
            for idx, (x, y, r) in enumerate(circles):
                writer.writerow([p.name, idx, int(x), int(y), int(r)])
