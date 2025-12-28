import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Спробуємо імпортувати бібліотеку. Якщо її немає - кинемо зрозумілу помилку.
try:
    from psnr_hvsm import psnr_hvs_hvsm
except ImportError:
    raise ImportError("Please install the library: pip install psnr-hvsm")

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

class QualityMetrics:
    @staticmethod
    def compute_psnr(gt: np.ndarray, dist: np.ndarray, data_range=None) -> float:
        if data_range is None: data_range = gt.max() - gt.min()
        return psnr(gt, dist, data_range=data_range)

    @staticmethod
    def compute_ssim(gt: np.ndarray, dist: np.ndarray, data_range=None) -> float:
        if data_range is None: data_range = gt.max() - gt.min()
        return ssim(gt, dist, data_range=data_range)

    @staticmethod
    def compute_hvs(gt: np.ndarray, dist: np.ndarray, data_range=None) -> tuple:
        """
        Calculates (PSNR-HVS, PSNR-HVS-M) using the 'psnr-hvsm' library.
        """
        if data_range is None: 
            data_range = gt.max() - gt.min()
        
        # Бібліотека вимагає зображення в діапазоні [0, 1]
        # Ми нормалізуємо дані, використовуючи data_range
        
        # 1. Знаходимо мінімум, щоб зсунути до 0
        min_val = gt.min()
        
        # 2. Нормалізація
        img1 = (gt - min_val) / data_range
        img2 = (dist - min_val) / data_range
        
        # 3. Обрізка під розмір 8x8 (вимога бібліотеки)
        h, w = img1.shape
        h = (h // 8) * 8
        w = (w // 8) * 8
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
        
        # 4. Виклик бібліотеки
        # Вона повертає (psnr_hvs, psnr_hvsm)
        return psnr_hvs_hvsm(img1, img2)