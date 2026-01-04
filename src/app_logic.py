import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass

from .config import VSTConfig
from .codec import BPGCodec
from .experiments import RateDistortionRunner
from .data_loader import SyntheticGenerator, ImageLoader
from .transform import VarianceStabilizer
from .interfaces import EncodeResult

@dataclass
class AnalysisResult:
    metrics_df: pd.DataFrame
    curves: Dict[str, Any] # q, psnr, etc lists
    oop_points: Dict[str, Dict[str, float]] # 'linear': {...}, 'vst': {...}
    source_image: np.ndarray # Noised
    ref_image: np.ndarray    # Original
    file_ext: str            # Original extension or .png for gen
    oop_image_lin: Optional[np.ndarray] = None
    oop_image_vst: Optional[np.ndarray] = None

class AnalysisController:
    def __init__(self, bpg_path: str = 'bpg-0.9.8-win64'):
        self.codec = BPGCodec(bpg_path)
        self.runner = RateDistortionRunner(self.codec)
        self.last_result: Optional[AnalysisResult] = None
        
        # Cache for generator to avoid regeneration if params confirm
        self._cached_gen_data = None
        self._cached_noise_level = -1.0

    def get_data(self, source_type: str, 
                 noise_level: float = 0.0, 
                 path_noised: str = "", 
                 path_original: str = "") -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Retrieves data based on source type.
        Returns: (Ref, Noised, FileExtension)
        """
        if source_type == 'gen':
            if self._cached_gen_data is None or noise_level != self._cached_noise_level:
                self._cached_gen_data = SyntheticGenerator.get_data(noise_level)
                self._cached_noise_level = noise_level
            return self._cached_gen_data[0], self._cached_gen_data[1], '.png'
            
        elif source_type == 'file':
            try:
                if not os.path.exists(path_noised): return None, None, ""
                
                i_n = ImageLoader.load_file(path_noised)
                if os.path.exists(path_original):
                    i_o = ImageLoader.load_file(path_original)
                else:
                    i_o = None # No reference mode?
                
                _, ext = os.path.splitext(path_noised)
                return i_o, i_n, ext
            except Exception as e:
                print(f"Data Load Error: {e}")
                return None, None, ""
        return None, None, ""

    def run_analysis(self, 
                     source_type: str,
                     noise_level: float,
                     path_noised: str,
                     path_original: str,
                     vst_a: float, vst_b: float,
                     q_start: int, q_end: int, q_step: int,
                     oop_metric: str = 'psnr') -> AnalysisResult:
        
        img_ref, img_noised, file_ext = self.get_data(source_type, noise_level, path_noised, path_original)
        
        if img_noised is None:
            raise ValueError("Could not load image data")
            
        vst_cfg = VSTConfig(a=vst_a, b=vst_b)
        q_rng = list(range(q_start, q_end + 1, q_step))
        
        # 1. Run Curves
        res_vst = self.runner.run_curve(img_ref, img_noised, vst_cfg, q_rng, use_vst=True)
        res_lin = self.runner.run_curve(img_ref, img_noised, vst_cfg, q_rng, use_vst=False)
        
        # 2. Find OOPs
        def find_oop(res):
            metric_key = oop_metric
            if metric_key not in res or not res[metric_key]: 
                 # Fallback to PSNR if metric not found (e.g. psnr_hvsm missing)
                 metric_key = 'psnr'
            
            if not res[metric_key]: return {}, -1
            
            idx = np.argmax(res[metric_key])
            return {k: res[k][idx] for k in res.keys()}, int(res['q'][idx])
            
        oop_vst, q_vst = find_oop(res_vst)
        oop_lin, q_lin = find_oop(res_lin)
        
        # 3. Re-generate OOP images
        def get_compressed_image(img, q, use_vst_loc):
            if q == -1: return None
            if use_vst_loc:
                vst = VarianceStabilizer(vst_cfg)
                to_compress = vst.forward(img)
            else:
                to_compress = img
                
            res = self.codec.compress_decompress(to_compress, q=q)
            decoded = res.decoded_image
            
            if use_vst_loc:
                return vst.inverse(decoded)
            return decoded

        img_oop_vst = get_compressed_image(img_noised, q_vst, True)
        img_oop_lin = get_compressed_image(img_noised, q_lin, False)
        
        # 4. DataFrame
        def get_fmt(val, fmt=".2f"):
            return f"{val:{fmt}}" if isinstance(val, (int, float)) else str(val)

        # Helper to calc MSE for OOP if not directly available (but we can compute it manually or use metrics)
        def get_oop_mse(img_oop, img_ref):
             if img_oop is None or img_ref is None: return 0.0
             return np.mean((img_oop - img_ref)**2)

        mse_lin = get_oop_mse(img_oop_lin, img_ref)
        mse_vst = get_oop_mse(img_oop_vst, img_ref)

        df = pd.DataFrame([
            {
                'Method': 'Standard space', 
                'Q(OOP)': oop_lin.get('q', 0), 
                f'{oop_metric.upper()}(OOP)': get_fmt(oop_lin.get(oop_metric, 0)),
                'PSNR': get_fmt(oop_lin.get('psnr', 0)), 
                'HVS-M': get_fmt(oop_lin.get('psnr_hvsm', 0)), 
                'MSE': get_fmt(mse_lin, ".2f"),
                'Filesize (KB)': get_fmt(oop_lin.get('file_size_kb', 0), ".1f"),
                'CR': get_fmt(oop_lin.get('cr', 0), ".1f")
            },
            {
                'Method': 'VST space', 
                'Q(OOP)': oop_vst.get('q', 0), 
                f'{oop_metric.upper()}(OOP)': get_fmt(oop_vst.get(oop_metric, 0)),
                'PSNR': get_fmt(oop_vst.get('psnr', 0)), 
                'HVS-M': get_fmt(oop_vst.get('psnr_hvsm', 0)), 
                'MSE': get_fmt(mse_vst, ".2f"),
                'Filesize (KB)': get_fmt(oop_vst.get('file_size_kb', 0), ".1f"),
                'CR': get_fmt(oop_vst.get('cr', 0), ".1f")
            }
        ])
        
        result = AnalysisResult(
            metrics_df=df,
            curves={'linear': res_lin, 'vst': res_vst},
            oop_points={'linear': oop_lin, 'vst': oop_vst},
            source_image=img_noised,
            ref_image=img_ref,
            file_ext=file_ext,
            oop_image_lin=img_oop_lin,
            oop_image_vst=img_oop_vst
        )
        self.last_result = result
        return result

    def save_oop_image(self, result: AnalysisResult, method: str, output_dir: str = "results") -> str:
        """
        Saves the visual result of the OOP for the given method ('vst' or 'linear').
        Returns the path to the saved file.
        """
        if method not in ['vst', 'linear']: return ""
        
        img = result.oop_image_vst if method == 'vst' else result.oop_image_lin
        if img is None: return ""
        
        # Determine filename
        # Base on input filename if available (not easily accessible here without plumbing, 
        # but we can pass it or just use generic since user asked for 'save_oop_images option')
        # Let's use a generic name pattern: "Image_OOP_{method}.png"
        
        os.makedirs(output_dir, exist_ok=True)
        fname = f"Image_OOP_{method}.png"
        path = os.path.join(output_dir, fname)
        
        try:
            import imageio.v3 as iio
            # Normalize to uint8 [0, 255]
            d_min, d_max = img.min(), img.max()
            if d_max > d_min:
                norm = ((img - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
            else:
                norm = img.astype(np.uint8)
                
            iio.imwrite(path, norm)
            print(f"Saved OOP image: {path}")
            return path
        except Exception as e:
            print(f"Failed to save OOP image: {e}")
            return ""

    def save_results_csv(self, result: AnalysisResult, path: str):
        try:
            result.metrics_df.to_csv(path, index=False)
        except Exception as e:
            print(f"Failed to save CSV: {e}")
