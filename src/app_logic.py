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
                     q_start: int, q_end: int, q_step: int) -> AnalysisResult:
        
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
            # OOP defined as max PSNR for now (or maybe HVS-M in future)
            # User logic was max PSNR
            idx = np.argmax(res['psnr'])
            return {k: res[k][idx] for k in res.keys()}, idx
            
        oop_vst, _ = find_oop(res_vst)
        oop_lin, _ = find_oop(res_lin)
        
        # 3. DataFrame
        # Construct summary dataframe for display
        df = pd.DataFrame([
            {
                'Method': 'Standard', 
                'Q(OOP)': oop_lin['q'], 
                'PSNR': f"{oop_lin['psnr']:.2f}", 
                'HVS-M': f"{oop_lin.get('psnr_hvsm', 0):.2f}", 
                'CR': f"{oop_lin['cr']:.1f}"
            },
            {
                'Method': 'Proposed (VST)', 
                'Q(OOP)': oop_vst['q'], 
                'PSNR': f"{oop_vst['psnr']:.2f}", 
                'HVS-M': f"{oop_vst.get('psnr_hvsm', 0):.2f}", 
                'CR': f"{oop_vst['cr']:.1f}"
            }
        ])
        
        result = AnalysisResult(
            metrics_df=df,
            curves={'linear': res_lin, 'vst': res_vst},
            oop_points={'linear': oop_lin, 'vst': oop_vst},
            source_image=img_noised,
            ref_image=img_ref,
            file_ext=file_ext
        )
        self.last_result = result
        return result

    def save_oop_image(self, result: AnalysisResult, method: str, output_dir: str = "results") -> str:
        """
        Saves the visual result of the OOP for the given method ('vst' or 'linear').
        Returns the path to the saved file.
        """
        if method not in ['vst', 'linear']: return ""
        
        oop = result.oop_points[method]
        q = oop['q']
        
        # Re-run compression to get the image
        # This is slightly inefficient (running again), but cleaner than storing all images in memory
        vst_cfg = VSTConfig() # We need the actual config used! 
        # TODO: Store config in Result? For now assume params from UI passed or standard.
        # Actually, self.last_result doesn't store VST params used. 
        # We should probably pass them or store them.
        # Let's assume standard config or what was passed last? 
        # Better: Re-instantiate based on what we know, or just do it in run_analysis.
        pass 
        # Optimization: Codec can save directly.
        # But we need domain locic.
        
        return "Not implemented fully yet, need VST Config passing"

    def save_results_csv(self, result: AnalysisResult, path: str):
        result.metrics_df.to_csv(path, index=False)
