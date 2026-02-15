
import time
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from src.config import AppConfig, VSTConfig
from src.transform import VarianceStabilizer
from src.codec import BPGCodec

def load_or_generate_image(path: str, size: tuple = (512, 512)) -> np.ndarray:
    try:
        if Path(path).exists():
            img = iio.imread(path)
            # Convert to float [0, 1]
            if img.dtype == np.uint8:
                img = img.astype(float) / 255.0
            elif img.dtype == np.uint16:
                img = img.astype(float) / 65535.0
            
            # Handle RGB to Grayscale if needed
            if img.ndim == 3:
                img = np.mean(img, axis=2)
            return img
    except Exception as e:
        print(f"Could not load image from {path}: {e}")
    
    print("Generating random image...")
    return np.random.rand(*size)

def benchmark():
    # Setup
    config = AppConfig()
    vst = VarianceStabilizer(config.vst)
    # Ensure BPG path is correct regarding where the script is run
    if not Path(config.bpg_path).exists():
        # Fallback for running from root
        potential_path = Path('bpg-0.9.8-win64')
        if potential_path.exists():
            config.bpg_path = str(potential_path)
            
    bpg = BPGCodec(config.bpg_path)
    
    # Load Image
    image = load_or_generate_image(config.data.path_original)
    print(f"Image shape: {image.shape}")
    
    iterations = 100
    warmup = 5
    
    vst_times = []
    bpg_times = []
    inv_vst_times = []
    
    print(f"Starting benchmark ({iterations} iterations)...")
    
    for i in range(iterations + warmup):
        # 1. Forward VST
        t0 = time.perf_counter()
        vst_img = vst.forward(image)
        t1 = time.perf_counter()
        
        # 2. BPG Codec (Encode + Decode)
        # Using q=30 as a representative quality
        t2 = time.perf_counter()
        # Note: compress_decompress returns an object with decoded_image
        res = bpg.compress_decompress(vst_img, q=30)
        decoded_img = res.decoded_image
        t3 = time.perf_counter()
        
        # 3. Inverse VST
        t4 = time.perf_counter()
        final_img = vst.inverse(decoded_img)
        t5 = time.perf_counter()
        
        if i >= warmup:
            vst_times.append((t1 - t0) * 1000) # ms
            bpg_times.append((t3 - t2) * 1000) # ms
            inv_vst_times.append((t5 - t4) * 1000) # ms
            
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{iterations + warmup}")

    # Results
    print("\nBenchmark Results (Avg ± Std Dev):")
    print("-" * 40)
    print(f"Forward VST : {np.mean(vst_times):.4f} ms ± {np.std(vst_times):.4f} ms")
    print(f"BPG Codec   : {np.mean(bpg_times):.4f} ms ± {np.std(bpg_times):.4f} ms")
    print(f"Inverse VST : {np.mean(inv_vst_times):.4f} ms ± {np.std(inv_vst_times):.4f} ms")
    print("-" * 40)
    print(f"Total Loop  : {np.mean(vst_times) + np.mean(bpg_times) + np.mean(inv_vst_times):.4f} ms")

if __name__ == "__main__":
    benchmark()
