import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class NoiseEstimator:
    # ... (Keep existing methods: calculate_exact_sigma, estimate_blind_sigma) ...
    
    @staticmethod
    def calculate_exact_sigma(img_log_noised: np.ndarray, img_log_original: np.ndarray) -> float:
        noise_map = img_log_noised - img_log_original
        return np.std(noise_map)

    @staticmethod
    def estimate_blind_sigma(img_log: np.ndarray) -> float:
        from scipy.signal import convolve2d
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        high_freq = convolve2d(img_log, kernel, mode='same', boundary='symm')
        mad = np.median(np.abs(high_freq - np.median(high_freq)))
        sigma_est = (1.4826 * mad) / np.sqrt(20.0)
        return sigma_est * 4.5

class QualityMetrics:
    """New class for full-reference metrics."""
    
    @staticmethod
    def compute_psnr(ground_truth: np.ndarray, distorted: np.ndarray, data_range=None) -> float:
        print(ground_truth.shape, distorted.shape)
        if data_range is None:
            data_range = ground_truth.max() - ground_truth.min()
        return psnr(ground_truth, distorted, data_range=data_range)

    @staticmethod
    def compute_ssim(ground_truth: np.ndarray, distorted: np.ndarray, data_range=None) -> float:
        if data_range is None:
            data_range = ground_truth.max() - ground_truth.min()
        # ssim requires specifying data_range for float data
        return ssim(ground_truth, distorted, data_range=data_range)