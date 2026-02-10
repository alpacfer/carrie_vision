"""Tests for image filtering functions."""

import pytest
import numpy as np
from carrie_vision.filters import apply_grayscale, apply_blur


class TestApplyGrayscale:
    """Test cases for apply_grayscale function."""
    
    def test_grayscale_conversion(self):
        """Test basic RGB to grayscale conversion."""
        # Create a simple RGB image
        rgb_image = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 255], [0, 0, 0], [128, 128, 128]]
        ], dtype=np.uint8)
        
        gray = apply_grayscale(rgb_image)
        
        # Check shape
        assert gray.shape == (2, 3)
        
        # Check that white is still white and black is still black
        assert gray[1, 0] == 255  # White pixel
        assert gray[1, 1] == 0    # Black pixel
    
    def test_grayscale_with_red_image(self):
        """Test grayscale conversion with a pure red image."""
        rgb_image = np.ones((5, 5, 3), dtype=np.uint8) * [255, 0, 0]
        gray = apply_grayscale(rgb_image)
        
        # Red channel contributes 0.299 to grayscale
        expected = int(255 * 0.299)
        assert np.all(gray == expected)
    
    def test_grayscale_with_green_image(self):
        """Test grayscale conversion with a pure green image."""
        rgb_image = np.ones((5, 5, 3), dtype=np.uint8) * [0, 255, 0]
        gray = apply_grayscale(rgb_image)
        
        # Green channel contributes 0.587 to grayscale
        expected = int(255 * 0.587)
        assert np.all(gray == expected)
    
    def test_grayscale_with_blue_image(self):
        """Test grayscale conversion with a pure blue image."""
        rgb_image = np.ones((5, 5, 3), dtype=np.uint8) * [0, 0, 255]
        gray = apply_grayscale(rgb_image)
        
        # Blue channel contributes 0.114 to grayscale
        expected = int(255 * 0.114)
        assert np.all(gray == expected)
    
    def test_grayscale_with_invalid_type(self):
        """Test grayscale conversion with invalid input type."""
        with pytest.raises(TypeError, match="Image must be a numpy array"):
            apply_grayscale([1, 2, 3])
    
    def test_grayscale_with_2d_array(self):
        """Test grayscale conversion with 2D array instead of 3D."""
        gray_image = np.ones((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="Image must be RGB with shape"):
            apply_grayscale(gray_image)
    
    def test_grayscale_with_wrong_channels(self):
        """Test grayscale conversion with wrong number of channels."""
        rgba_image = np.ones((10, 10, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Image must be RGB with shape"):
            apply_grayscale(rgba_image)


class TestApplyBlur:
    """Test cases for apply_blur function."""
    
    def test_blur_grayscale_image(self):
        """Test blurring a grayscale image."""
        # Create a simple test pattern
        image = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 255, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        blurred = apply_blur(image, kernel_size=3)
        
        # Check shape is preserved
        assert blurred.shape == image.shape
        
        # Center pixel should be less than 255 due to averaging
        assert blurred[2, 2] < 255
        
        # Neighboring pixels should have non-zero values
        assert blurred[1, 2] > 0
        assert blurred[2, 1] > 0
    
    def test_blur_rgb_image(self):
        """Test blurring an RGB image."""
        # Create a simple RGB test pattern
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        image[2, 2] = [255, 128, 64]
        
        blurred = apply_blur(image, kernel_size=3)
        
        # Check shape is preserved
        assert blurred.shape == image.shape
        
        # Center pixel should be less than original due to averaging
        assert blurred[2, 2, 0] < 255
        assert blurred[2, 2, 1] < 128
        assert blurred[2, 2, 2] < 64
    
    def test_blur_with_different_kernel_sizes(self):
        """Test blurring with different kernel sizes."""
        image = np.ones((10, 10), dtype=np.uint8) * 100
        
        # Test with kernel size 1 (should return nearly the same image)
        blurred_1 = apply_blur(image, kernel_size=1)
        assert np.allclose(blurred_1, image)
        
        # Test with kernel size 3
        blurred_3 = apply_blur(image, kernel_size=3)
        assert blurred_3.shape == image.shape
        
        # Test with kernel size 5
        blurred_5 = apply_blur(image, kernel_size=5)
        assert blurred_5.shape == image.shape
    
    def test_blur_with_invalid_type(self):
        """Test blurring with invalid input type."""
        with pytest.raises(TypeError, match="Image must be a numpy array"):
            apply_blur([1, 2, 3])
    
    def test_blur_with_invalid_kernel_size(self):
        """Test blurring with invalid kernel size."""
        image = np.ones((10, 10), dtype=np.uint8)
        
        # Even kernel size
        with pytest.raises(ValueError, match="Kernel size must be a positive odd number"):
            apply_blur(image, kernel_size=2)
        
        # Negative kernel size
        with pytest.raises(ValueError, match="Kernel size must be a positive odd number"):
            apply_blur(image, kernel_size=-1)
        
        # Zero kernel size
        with pytest.raises(ValueError, match="Kernel size must be a positive odd number"):
            apply_blur(image, kernel_size=0)
    
    def test_blur_with_invalid_dimensions(self):
        """Test blurring with invalid image dimensions."""
        # 1D array
        image_1d = np.ones(10, dtype=np.uint8)
        with pytest.raises(ValueError, match="Image must be 2D or 3D array"):
            apply_blur(image_1d)
        
        # 4D array
        image_4d = np.ones((10, 10, 3, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="Image must be 2D or 3D array"):
            apply_blur(image_4d)
    
    def test_blur_preserves_dtype(self):
        """Test that blur preserves the input data type."""
        image_uint8 = np.ones((10, 10), dtype=np.uint8) * 100
        blurred = apply_blur(image_uint8)
        assert blurred.dtype == np.uint8
        
        image_float = np.ones((10, 10), dtype=np.float32) * 100
        blurred_float = apply_blur(image_float)
        assert blurred_float.dtype == np.float32
