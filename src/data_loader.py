import numpy as np
import os

# Спробуємо імпортувати бібліотеки для читання
try:
    import tifffile
    def read_tiff(path): return tifffile.imread(path)
except ImportError:
    from PIL import Image
    def read_tiff(path): return np.array(Image.open(path))

class ImageLoader:
    """Універсальний завантажувач для одного файлу."""
    
    @staticmethod
    def load_file(path: str, min_value: float = 1.0) -> np.ndarray:
        """
        Завантажує файл та замінює нулі/малі значення.
        
        Args:
            path: шлях до файлу.
            min_value: значення, яким замінюються 0 (щоб уникнути log(0)). 
                       Для 16-бітних SAR даних 1.0 є безпечним мінімумом.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не знайдено: {path}")
            
        # Читання
        image = read_tiff(path).astype(np.float32)
        
        # Санітизація даних (заміна 0 та NaN)
        # np.nan_to_num замінює NaN на 0, потім clip піднімає все до min_value
        image = np.nan_to_num(image)
        # image = np.maximum(image, min_value)
        
        return image

class SyntheticGenerator:
    """Генератор патернів (оновлений)."""
    
    @staticmethod
    def get_data(noise_level=0.25, shape=(400, 400)):
        x, y = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]))
        
        # 1. Базовий сигнал (Градієнт)
        clean = 100 * (x + y) + 50
        
        # 2. Яскравий об'єкт (Коло)
        mask_circle = (x - 0.6)**2 + (y - 0.6)**2 < 0.05
        clean[mask_circle] += 150
        
        # 3. Чорний об'єкт (Квадрат)
        # Імітує тінь або воду (інтенсивність ~1)
        mask_dark = (np.abs(x - 0.3) < 0.1) & (np.abs(y - 0.3) < 0.1)
        clean[mask_dark] = 1.0 
        
        # Генерація шуму (Gamma distribution for Speckle)
        k = 1 / (noise_level ** 2)
        noise = np.random.gamma(k, 1.0, shape) / k
        
        noised_image = clean * noise
        
        # Гарантуємо, що і на зашумленому немає абсолютних нулів
        # noised_image = np.maximum(noised_image, 1.0)
        # !!! обмеження максимуму в 255 дає викривлення нормального розподілу, noise має виводити картинку за рамки 255 або бути незначним
        # noised_image = np.clip(noised_image, 1.0, 255.0)
        
        return clean, noised_image