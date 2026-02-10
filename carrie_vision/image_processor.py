"""Image processing core functionality."""

import numpy as np
from typing import Tuple, Optional


class ImageProcessor:
    """Main class for processing images."""
    
    def __init__(self, image: Optional[np.ndarray] = None):
        """
        Initialize the ImageProcessor.
        
        Args:
            image: A numpy array representing an image (optional)
        """
        self.image = image
    
    def load_image(self, image: np.ndarray) -> None:
        """
        Load an image into the processor.
        
        Args:
            image: A numpy array representing an image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")
        self.image = image
    
    def get_dimensions(self) -> Tuple[int, ...]:
        """
        Get the dimensions of the loaded image.
        
        Returns:
            Tuple of image dimensions
        
        Raises:
            ValueError: If no image is loaded
        """
        if self.image is None:
            raise ValueError("No image loaded")
        return self.image.shape
    
    def resize(self, width: int, height: int) -> np.ndarray:
        """
        Resize the image to the specified dimensions.
        
        Args:
            width: Target width
            height: Target height
        
        Returns:
            Resized image as numpy array
        
        Raises:
            ValueError: If no image is loaded or invalid dimensions
        """
        if self.image is None:
            raise ValueError("No image loaded")
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        
        # Simple nearest-neighbor resize
        h, w = self.image.shape[:2]
        if len(self.image.shape) == 3:
            resized = np.zeros((height, width, self.image.shape[2]), dtype=self.image.dtype)
        else:
            resized = np.zeros((height, width), dtype=self.image.dtype)
        
        for i in range(height):
            for j in range(width):
                src_i = int(i * h / height)
                src_j = int(j * w / width)
                resized[i, j] = self.image[src_i, src_j]
        
        return resized
    
    def validate_image(self) -> bool:
        """
        Validate that the image is properly formatted.
        
        Returns:
            True if image is valid, False otherwise
        """
        if self.image is None:
            return False
        if not isinstance(self.image, np.ndarray):
            return False
        if self.image.size == 0:
            return False
        return True
