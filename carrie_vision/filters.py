"""Image filtering functions."""

import numpy as np


def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.
    
    Args:
        image: RGB image as numpy array with shape (H, W, 3)
    
    Returns:
        Grayscale image as numpy array with shape (H, W)
    
    Raises:
        ValueError: If image is not RGB
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB with shape (H, W, 3)")
    
    # Standard luminosity method for RGB to grayscale conversion
    grayscale = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return grayscale.astype(image.dtype)


def apply_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a simple box blur to an image.
    
    Args:
        image: Input image as numpy array
        kernel_size: Size of the blur kernel (must be odd)
    
    Returns:
        Blurred image as numpy array
    
    Raises:
        ValueError: If kernel_size is invalid
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array")
    
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number")
    
    if len(image.shape) == 2:
        # Grayscale image
        return _blur_2d(image, kernel_size)
    elif len(image.shape) == 3:
        # RGB image - blur each channel separately
        blurred = np.zeros_like(image)
        for c in range(image.shape[2]):
            blurred[:, :, c] = _blur_2d(image[:, :, c], kernel_size)
        return blurred
    else:
        raise ValueError("Image must be 2D or 3D array")


def _blur_2d(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply box blur to a 2D array.
    
    Args:
        image: 2D numpy array
        kernel_size: Size of the blur kernel
    
    Returns:
        Blurred 2D array
    """
    h, w = image.shape
    pad = kernel_size // 2
    blurred = np.zeros_like(image, dtype=np.float64)
    
    # Pad the image
    padded = np.pad(image, pad, mode='edge')
    
    # Apply box filter
    for i in range(h):
        for j in range(w):
            blurred[i, j] = np.mean(
                padded[i:i+kernel_size, j:j+kernel_size]
            )
    
    return blurred.astype(image.dtype)
