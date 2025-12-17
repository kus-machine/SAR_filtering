import numpy as np
from scipy.signal import convolve2d

class NoiseEstimator:
    """Клас для оцінки параметрів шуму."""
    
    @staticmethod
    def calculate_exact_sigma(img_log_noised: np.ndarray, img_log_original: np.ndarray) -> float:
        """
        Точний розрахунок STD шуму за наявності еталону.
        Noise = Log(Noised) - Log(Original)
        """
        # Різниця дає чистий шум + невеликий зсув (bias)
        noise_map = img_log_noised - img_log_original
        
        # Нас цікавить розкид (std), а не середнє значення
        return np.std(noise_map)

    @staticmethod
    def estimate_blind_sigma(img_log: np.ndarray) -> float:
        """
        Оцінка STD шуму без еталону (метод MAD).
        Використовується при обробці реальних супутникових знімків.
        """
        # 1. Виділяємо високочастотні деталі (шум + краї)
        # Використовуємо простий лапласіан-подібний оператор
        kernel = np.array([[0, -1, 0], 
                           [-1, 4, -1], 
                           [0, -1, 0]])
        
        high_freq = convolve2d(img_log, kernel, mode='same', boundary='symm')
        
        # 2. Розрахунок MAD (Median Absolute Deviation)
        # Медіана стійка до сильних перепадів (країв об'єктів), 
        # тому вона оцінює саме шум, ігноруючи контури.
        mad = np.median(np.abs(high_freq - np.median(high_freq)))
        
        # 3. Конвертація MAD в Sigma для нормального розподілу
        # Коефіцієнт 1.4826 - стандартна константа для Gaussian distribution
        # Додатковий дільник коригує вплив ядра згортки
        sigma_est = (1.4826 * mad) / np.sqrt(20.0) # Емпіричний дільник для цього ядра
        
        # Для простоти часто використовують спрощений варіант Donoho (Wavelet HH band),
        # але для spatial domain цей метод теж непоганий.
        
        return sigma_est * 4.5 # Емпіричне калібрування під наш VST