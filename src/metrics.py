import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from .interfaces import MetricRegistry

# Check for psnr-hvsm
try:
    from psnr_hvsm import psnr_hvs_hvsm
    HAS_HVSM = True
except ImportError:
    HAS_HVSM = False

class QualityMetrics:
    """Namespace for metric calculations."""

    @staticmethod
    @MetricRegistry.register("psnr")
    def compute_psnr(gt: np.ndarray, dist: np.ndarray, data_range=None) -> float:
        """Peak Signal-to-Noise Ratio."""
        if data_range is None: data_range = gt.max() - gt.min()
        return psnr(gt, dist, data_range=data_range)

    @staticmethod
    @MetricRegistry.register("ssim")
    def compute_ssim(gt: np.ndarray, dist: np.ndarray, data_range=None) -> float:
        """Structural Similarity Index."""
        if data_range is None: data_range = gt.max() - gt.min()
        return ssim(gt, dist, data_range=data_range)

    @staticmethod
    def compute_hvs_metrics(gt: np.ndarray, dist: np.ndarray, data_range=None) -> tuple:
        """
        Calculates (PSNR-HVS, PSNR-HVS-M).
        Returns: (psnr_hvs, psnr_hvsm)
        """
        if not HAS_HVSM:
            return 0.0, 0.0
            
        if data_range is None: 
            data_range = gt.max() - gt.min()
        
        # Normalize to [0, 1] for the library
        min_val = gt.min()
        img1 = (gt - min_val) / data_range
        img2 = (dist - min_val) / data_range
        
        # Crop to 8x8 blocks
        h, w = img1.shape
        h = (h // 8) * 8
        w = (w // 8) * 8
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
        
        return psnr_hvs_hvsm(img1, img2)

    @staticmethod
    @MetricRegistry.register("psnr_hvs")
    def compute_psnr_hvs(gt: np.ndarray, dist: np.ndarray, data_range=None) -> float:
        hvs, _ = QualityMetrics.compute_hvs_metrics(gt, dist, data_range)
        return hvs

    @staticmethod
    @MetricRegistry.register("psnr_hvsm")
    def compute_psnr_hvsm(gt: np.ndarray, dist: np.ndarray, data_range=None) -> float:
        _, hvsm = QualityMetrics.compute_hvs_metrics(gt, dist, data_range)
        return hvsm

class NoiseEstimator:
    @staticmethod
    def calculate_exact_sigma(img_log_noised: np.ndarray, img_log_original: np.ndarray) -> float:
        """std(Noised - Original)"""
        return np.std(img_log_noised - img_log_original)

    @staticmethod
    def estimate_blind_sigma(img_log: np.ndarray) -> float:
        """MAD-based estimation."""
        from scipy.signal import convolve2d
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        high_freq = convolve2d(img_log, kernel, mode='same', boundary='symm')
        mad = np.median(np.abs(high_freq - np.median(high_freq)))
        # 1.4826 converts MAD to Sigma for Gaussian distribution
        # * 4.5 is empirical calibration for VST log domain
        return (1.4826 * mad) * 4.5
