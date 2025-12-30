import numpy as np
import os
from typing import Tuple, Optional
from PIL import Image

# Try importing imageio/tifffile
try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

class ImageLoader:
    """Universal loader for TIFF, PNG, and other image formats."""
    
    @staticmethod
    def load_file(path: str, min_value: float = 1.0) -> np.ndarray:
        """
        Detects format by extension, loads image, converts to float,
        handles RGB->Gray conversion, and removes zeros.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        _, ext = os.path.splitext(path.lower())
        
        # 1. LOAD DATA
        if ext in ['.tif', '.tiff'] and HAS_TIFFFILE:
            image = tifffile.imread(path)
        elif HAS_IMAGEIO:
            image = iio.imread(path)
        else:
            image = np.array(Image.open(path))

        # 2. PREPROCESS
        image = image.astype(np.float32)
        
        # Handle RGB (H, W, 3) or RGBA (H, W, 4) -> Grayscale (H, W)
        if image.ndim == 3:
            image = np.mean(image, axis=2)
            
        # 3. SANITIZE (No zeros for Log transform)
        image = np.nan_to_num(image)
        image = np.maximum(image, min_value)
        
        return image

class SyntheticGenerator:
    """Generates synthetic SAR patterns."""
    
    @staticmethod
    def get_data(noise_level: float = 0.25, shape: Tuple[int, int] = (400, 400)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates clean and noised synthetic SAR images.
        Returns: (clean, noised)
        """
        x, y = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]))
        
        clean = 100 * (x + y) + 50
        
        # Add circle feature
        mask_circle = (x - 0.6)**2 + (y - 0.6)**2 < 0.05
        clean[mask_circle] += 150
        
        # Add dark square feature
        mask_dark = (np.abs(x - 0.3) < 0.1) & (np.abs(y - 0.3) < 0.1)
        clean[mask_dark] = 1.0 
        
        # Add Speckle Noise (Gamma distributed)
        k = 1 / (noise_level ** 2)
        noise = np.random.gamma(k, 1.0, shape) / k
        
        noised = clean * noise
        noised = np.maximum(noised, 1.0)
        
        return clean, noised