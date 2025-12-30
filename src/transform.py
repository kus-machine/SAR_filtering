import numpy as np
from .config import VSTConfig

class VarianceStabilizer:
    def __init__(self, config: VSTConfig):
        self.cfg = config

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Forward Transform: Linear -> Log domain.
        y = a * log_b(image)
        """
        # Protect against zeros/negatives
        img_safe = np.maximum(image, self.cfg.epsilon)
        # log_b(x) = ln(x) / ln(b)
        return self.cfg.a * (np.log(img_safe) / np.log(self.cfg.b))

    def inverse(self, transformed_image: np.ndarray) -> np.ndarray:
        """
        Inverse Transform: Log -> Linear domain.
        x = b ^ (y / a)
        """
        exponent = transformed_image / self.cfg.a
        return np.power(self.cfg.b, exponent)