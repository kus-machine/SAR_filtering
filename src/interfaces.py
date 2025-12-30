from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class EncodeResult:
    decoded_image: np.ndarray
    file_size_bytes: int
    bpp: float

class BaseCodec(ABC):
    """Abstract base class for all image codecs."""
    
    @abstractmethod
    def compress_decompress(self, image: np.ndarray, q: int) -> EncodeResult:
        """
        Compresses and immediately decompresses the image.
        Args:
            image: Input image (float or uint8).
            q: Quality parameter (meaning depends on codec).
        Returns:
            EncodeResult containing decoded image, size, and bpp.
        """
        pass
        
    @abstractmethod
    def save_to_file(self, image: np.ndarray, q: int, output_path: str) -> int:
        """
        Saves the compressed stream to a file.
        Returns: file size in bytes.
        """
        pass

class MetricRegistry:
    """Registry for managing available quality metrics."""
    _metrics: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._metrics[name] = func
            return func
        return decorator

    @classmethod
    def get_metric(cls, name: str):
        return cls._metrics.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        return cls._metrics

class PlotterInterface(ABC):
    """Abstract base class for UI plotters."""
    
    @abstractmethod
    def plot_curves(self, results_dict: Dict[str, Any], oop_points: Dict[str, Any]):
        """
        Visualizes the rate-distortion curves and OOP.
        """
        pass
