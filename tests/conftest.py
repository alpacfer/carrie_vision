"""Test configuration and fixtures."""

import pytest
import numpy as np


@pytest.fixture
def simple_rgb_image():
    """Create a simple RGB image for testing."""
    return np.zeros((10, 10, 3), dtype=np.uint8)


@pytest.fixture
def simple_grayscale_image():
    """Create a simple grayscale image for testing."""
    return np.zeros((10, 10), dtype=np.uint8)


@pytest.fixture
def test_pattern_image():
    """Create a test pattern image."""
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    # Add some patterns
    image[5:15, 5:15, 0] = 255  # Red square
    image[10:20, 0:10, 1] = 255  # Green rectangle
    image[0:10, 10:20, 2] = 255  # Blue rectangle
    return image
