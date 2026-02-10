# Carrie Vision

A simple computer vision library for basic image processing operations.

## Features

- **ImageProcessor**: Core class for loading and processing images
  - Image loading and validation
  - Image resizing with nearest-neighbor interpolation
  - Image dimension retrieval

- **Filters**: Image filtering operations
  - Grayscale conversion (RGB to grayscale)
  - Box blur filter with configurable kernel size

## Installation

```bash
pip install -r requirements.txt
```

For development:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Usage

### Basic Image Processing

```python
import numpy as np
from carrie_vision import ImageProcessor

# Create a sample image
image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

# Initialize processor
processor = ImageProcessor(image)

# Get image dimensions
dims = processor.get_dimensions()
print(f"Image dimensions: {dims}")

# Resize image
resized = processor.resize(50, 50)
print(f"Resized to: {resized.shape}")

# Validate image
is_valid = processor.validate_image()
print(f"Image valid: {is_valid}")
```

### Applying Filters

```python
import numpy as np
from carrie_vision import apply_grayscale, apply_blur

# Create a sample RGB image
rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

# Convert to grayscale
gray_image = apply_grayscale(rgb_image)
print(f"Grayscale shape: {gray_image.shape}")

# Apply blur
blurred = apply_blur(rgb_image, kernel_size=5)
print(f"Blurred shape: {blurred.shape}")
```

## Running Tests

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=carrie_vision --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_image_processor.py
```

## Development

This project uses pytest for testing. Test files are located in the `tests/` directory.

### Test Structure
- `tests/test_image_processor.py`: Tests for ImageProcessor class
- `tests/test_filters.py`: Tests for filtering functions
- `tests/conftest.py`: Shared test fixtures

## License

MIT