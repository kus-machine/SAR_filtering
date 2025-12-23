import numpy as np
import time
from .config import VSTConfig
from .transform import VarianceStabilizer
from .metrics import QualityMetrics
from .codec import BPGCodec

class RateDistortionRunner:
    def __init__(self, codec: BPGCodec):
        self.codec = codec

    def run_curve(self, img_clean, img_noised, vst_config: VSTConfig, q_range: list, progress_callback=None):
        """
        Runs the BPG compression pipeline for a list of Q values.
        
        Returns:
            dict: {
                'q': [], 
                'bpp': [], 
                'psnr': [], 
                'ssim': [],
                'file_size_kb': []
            }
        """
        results = {
            'q': [], 'bpp': [], 'psnr': [], 'ssim': [], 'file_size_kb': []
        }
        
        # 1. Prepare VST
        vst = VarianceStabilizer(vst_config)
        img_log = vst.forward(img_noised)
        
        # Determine Reference for metrics
        # If we have ground truth (img_clean), use it. 
        # Otherwise, use img_noised (measure reconstruction fidelity).
        ref_img = img_clean if img_clean is not None else img_noised
        
        total_steps = len(q_range)
        
        for idx, q in enumerate(q_range):
            # 2. Compress/Decompress in Log Domain
            # We catch errors here to ensure one bad Q doesn't crash the whole loop
            try:
                img_log_dec, f_size, bpp = self.codec.compress_decompress(img_log, q=q)
                
                # 3. Inverse VST
                img_restored = vst.inverse(img_log_dec)
                
                # 4. Calculate Metrics
                val_psnr = QualityMetrics.compute_psnr(ref_img, img_restored)
                val_ssim = QualityMetrics.compute_ssim(ref_img, img_restored)
                
                # Store
                results['q'].append(q)
                results['bpp'].append(bpp)
                results['psnr'].append(val_psnr)
                results['ssim'].append(val_ssim)
                results['file_size_kb'].append(f_size / 1024.0)
                
            except Exception as e:
                print(f"Error at q={q}: {e}")
            
            # Update progress bar if provided
            if progress_callback:
                progress_callback(idx + 1, total_steps)
                
        return results