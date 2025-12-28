import numpy as np
from .config import VSTConfig
from .transform import VarianceStabilizer
from .metrics import QualityMetrics
from .codec import BPGCodec

class RateDistortionRunner:
    def __init__(self, codec: BPGCodec):
        self.codec = codec

    def run_curve(self, img_clean, img_noised, vst_config: VSTConfig, q_range: list, use_vst: bool = True, progress_callback=None):
        """
        Runs compression loop.
        Args:
            use_vst (bool): If True, applies VST -> Compress -> InvVST.
                            If False, applies Compress -> Decompress (Linear domain).
        """
        results = {
            'q': [], 'bpp': [], 'file_size_kb': [], 'cr': [], # Compression Ratio
            'psnr': [], 'ssim': [], 
            'psnr_hvs': [], 'psnr_hvsm': []
        }
        
        # 1. Prepare Data
        if use_vst:
            vst = VarianceStabilizer(vst_config)
            img_to_compress = vst.forward(img_noised)
        else:
            img_to_compress = img_noised

        # Reference is always the clean image (or noised if clean not avail)
        ref_img = img_clean if img_clean is not None else img_noised
        
        # Original size in bytes (assuming 8-bit grayscale for CR calc, as per paper)
        h, w = img_noised.shape
        original_size_bytes = h * w 
        
        total = len(q_range)
        
        for idx, q in enumerate(q_range):
            try:
                # 2. Compression
                # Codec handles float->8bit->BPG->float normalization automatically
                img_decoded, f_size_bytes, bpp = self.codec.compress_decompress(img_to_compress, q=q)
                
                # 3. Restoration
                if use_vst:
                    img_restored = vst.inverse(img_decoded)
                else:
                    img_restored = img_decoded
                
                # 4. Metrics
                # Important: Calculate metrics against the CLEAN reference to find OOP
                p = QualityMetrics.compute_psnr(ref_img, img_restored)
                s = QualityMetrics.compute_ssim(ref_img, img_restored)
                hvs, hvsm = QualityMetrics.compute_hvs(ref_img, img_restored)
                
                # CR = Original Size / Compressed Size
                cr = original_size_bytes / f_size_bytes if f_size_bytes > 0 else 0
                
                results['q'].append(q)
                results['bpp'].append(bpp)
                results['file_size_kb'].append(f_size_bytes / 1024.0)
                results['cr'].append(cr)
                results['psnr'].append(p)
                results['ssim'].append(s)
                results['psnr_hvs'].append(hvs)
                results['psnr_hvsm'].append(hvsm)
                
            except Exception as e:
                print(f"Err q={q}: {e}")
            
            if progress_callback: progress_callback(idx + 1, total)
                
        return results