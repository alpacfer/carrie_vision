"""Carrie Vision - A simple computer vision library."""

__version__ = "0.1.0"

from .image_processor import ImageProcessor
from .filters import apply_grayscale, apply_blur

__all__ = ["ImageProcessor", "apply_grayscale", "apply_blur"]
