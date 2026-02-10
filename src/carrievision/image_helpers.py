from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

Roi = Tuple[int, int, int, int] # (x, y, w, h)


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
