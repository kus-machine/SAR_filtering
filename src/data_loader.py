import numpy as np
import os
import sys

# Try importing imageio for generic image formats (PNG, JPG)
try:
    import imageio.v3 as iio
except ImportError:
    iio = None

# Try importing tifffile for scientific TIFFs
try:
    import tifffile
except ImportError:
    tifffile = None

from PIL import Image

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
        if ext in ['.tif', '.tiff']:
            image = ImageLoader._load_tiff(path)
        else:
            image = ImageLoader._load_generic(path)

        # 2. PREPROCESS (Common for all formats)
        image = image.astype(np.float32)
        
        # Handle RGB (H, W, 3) or RGBA (H, W, 4) -> Grayscale (H, W)
        if image.ndim == 3:
            # Simple average or ITU-R 601-2 luma transform could be used.
            # For SAR/Scientific data stored in PNG, usually channels are identical.
            # Let's use simple mean to flatten channels.
            image = np.mean(image, axis=2)
            
        # 3. SANITIZE (No zeros for Log transform)
        image = np.nan_to_num(image)
        image = np.maximum(image, min_value)
        
        return image

    @staticmethod
    def _load_tiff(path):
        if tifffile:
            return tifffile.imread(path)
        else:
            return np.array(Image.open(path))

    @staticmethod
    def _load_generic(path):
        """Loads PNG, JPG, BMP, etc."""
        if iio:
            return iio.imread(path)
        else:
            return np.array(Image.open(path))

class SyntheticGenerator:
    """Generates synthetic SAR patterns."""
    
    @staticmethod
    def get_data(noise_level=0.25, shape=(400, 400)):
        x, y = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]))
        
        clean = 100 * (x + y) + 50
        mask_circle = (x - 0.6)**2 + (y - 0.6)**2 < 0.05
        clean[mask_circle] += 150
        mask_dark = (np.abs(x - 0.3) < 0.1) & (np.abs(y - 0.3) < 0.1)
        clean[mask_dark] = 1.0 
        
        # Noise
        k = 1 / (noise_level ** 2)
        noise = np.random.gamma(k, 1.0, shape) / k

        noised = clean * noise
        noised = np.maximum(noised, 1.0)

        # Гарантуємо, що і на зашумленому немає абсолютних нулів
        # noised_image = np.maximum(noised_image, 1.0)
        # !!! обмеження максимуму в 255 дає викривлення нормального розподілу, noise має виводити картинку за рамки 255 або бути незначним
        # noised_image = np.clip(noised_image, 1.0, 255.0)

        
        return clean, noised