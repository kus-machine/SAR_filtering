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
        results = {
            'q': [], 'bpp': [], 'file_size_kb': [],
            'psnr': [], 'ssim': [], 
            'psnr_hvs': [], 'psnr_hvsm': [] # <--- НОВІ ПОЛЯ
        }
        
        vst = VarianceStabilizer(vst_config)
        img_log = vst.forward(img_noised)
        ref_img = img_clean if img_clean is not None else img_noised
        
        total = len(q_range)
        for idx, q in enumerate(q_range):
            try:
                img_log_dec, f_size, bpp = self.codec.compress_decompress(img_log, q=q)
                img_restored = vst.inverse(img_log_dec)
                
                # Metrics
                p = QualityMetrics.compute_psnr(ref_img, img_restored)
                s = QualityMetrics.compute_ssim(ref_img, img_restored)
                # HVS Metrics
                hvs, hvsm = QualityMetrics.compute_hvs(ref_img, img_restored)
                
                results['q'].append(q)
                results['bpp'].append(bpp)
                results['file_size_kb'].append(f_size / 1024.0)
                results['psnr'].append(p)
                results['ssim'].append(s)
                results['psnr_hvs'].append(hvs)
                results['psnr_hvsm'].append(hvsm)
                print("psnrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
            except Exception as e:
                print(f"Err q={q}: {e}")
            
            if progress_callback: progress_callback(idx + 1, total)
                
        return results