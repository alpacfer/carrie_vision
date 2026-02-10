"""Tests for the ImageProcessor class."""

import pytest
import numpy as np
from carrie_vision.image_processor import ImageProcessor


class TestImageProcessor:
    """Test cases for ImageProcessor class."""
    
    def test_init_without_image(self):
        """Test initialization without an image."""
        processor = ImageProcessor()
        assert processor.image is None
    
    def test_init_with_image(self):
        """Test initialization with an image."""
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        processor = ImageProcessor(test_image)
        assert processor.image is not None
        assert processor.image.shape == (10, 10, 3)
    
    def test_load_image(self):
        """Test loading an image."""
        processor = ImageProcessor()
        test_image = np.zeros((20, 20, 3), dtype=np.uint8)
        processor.load_image(test_image)
        assert processor.image is not None
        assert processor.image.shape == (20, 20, 3)
    
    def test_load_image_with_invalid_type(self):
        """Test loading an image with invalid type."""
        processor = ImageProcessor()
        with pytest.raises(TypeError, match="Image must be a numpy array"):
            processor.load_image([1, 2, 3])
    
    def test_get_dimensions_with_image(self):
        """Test getting dimensions of a loaded image."""
        test_image = np.zeros((15, 20, 3), dtype=np.uint8)
        processor = ImageProcessor(test_image)
        dimensions = processor.get_dimensions()
        assert dimensions == (15, 20, 3)
    
    def test_get_dimensions_without_image(self):
        """Test getting dimensions without a loaded image."""
        processor = ImageProcessor()
        with pytest.raises(ValueError, match="No image loaded"):
            processor.get_dimensions()
    
    def test_resize_valid_dimensions(self):
        """Test resizing an image with valid dimensions."""
        test_image = np.ones((10, 10, 3), dtype=np.uint8) * 100
        processor = ImageProcessor(test_image)
        resized = processor.resize(5, 5)
        assert resized.shape == (5, 5, 3)
        # Check that values are preserved (simple nearest-neighbor)
        assert np.all(resized == 100)
    
    def test_resize_grayscale_image(self):
        """Test resizing a grayscale image."""
        test_image = np.ones((10, 10), dtype=np.uint8) * 50
        processor = ImageProcessor(test_image)
        resized = processor.resize(20, 20)
        assert resized.shape == (20, 20)
        assert np.all(resized == 50)
    
    def test_resize_without_image(self):
        """Test resizing without a loaded image."""
        processor = ImageProcessor()
        with pytest.raises(ValueError, match="No image loaded"):
            processor.resize(10, 10)
    
    def test_resize_with_invalid_dimensions(self):
        """Test resizing with invalid dimensions."""
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        processor = ImageProcessor(test_image)
        
        with pytest.raises(ValueError, match="Width and height must be positive"):
            processor.resize(0, 10)
        
        with pytest.raises(ValueError, match="Width and height must be positive"):
            processor.resize(10, -5)
    
    def test_validate_image_with_valid_image(self):
        """Test validation with a valid image."""
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        processor = ImageProcessor(test_image)
        assert processor.validate_image() is True
    
    def test_validate_image_without_image(self):
        """Test validation without an image."""
        processor = ImageProcessor()
        assert processor.validate_image() is False
    
    def test_validate_image_with_empty_array(self):
        """Test validation with an empty array."""
        test_image = np.array([])
        processor = ImageProcessor(test_image)
        assert processor.validate_image() is False
    
    def test_validate_image_with_invalid_type(self):
        """Test validation with an invalid type."""
        processor = ImageProcessor()
        processor.image = [1, 2, 3]  # Not a numpy array
        assert processor.validate_image() is False
