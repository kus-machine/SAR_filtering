import numpy as np
from typing import List, Dict, Any, Callable, Optional
from .config import VSTConfig
from .transform import VarianceStabilizer
from .interfaces import BaseCodec, MetricRegistry
from .metrics import QualityMetrics # triggers registration

class RateDistortionRunner:
    def __init__(self, codec: BaseCodec, metrics_to_compute: Optional[List[str]] = None):
        self.codec = codec
        self.metrics_to_compute = metrics_to_compute or list(MetricRegistry.get_all().keys())

    def run_curve(self, 
                  img_clean: np.ndarray, 
                  img_noised: np.ndarray, 
                  vst_config: VSTConfig, 
                  q_range: List[int], 
                  use_vst: bool = True, 
                  progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, List[Any]]:
        
        results = {m: [] for m in self.metrics_to_compute}
        results.update({
            'q': [], 'bpp': [], 'file_size_kb': [], 'cr': [], 'mse_codec': []
        })
        
        if use_vst:
            vst = VarianceStabilizer(vst_config)
            img_to_compress = vst.forward(img_noised)
        else:
            img_to_compress = img_noised

        # If img_clean is None, we might compare against noised (though usually bad practice),
        # but the caller logic seems to handle this.
        ref_img = img_clean if img_clean is not None else img_noised
        h, w = img_noised.shape
        original_size_bytes = h * w 
        
        total = len(q_range)
        
        for idx, q in enumerate(q_range):
            try:
                # 1. Compress/Decompress
                # Now returns EncodeResult
                res = self.codec.compress_decompress(img_to_compress, q=q)
                img_decoded = res.decoded_image
                f_size_bytes = res.file_size_bytes
                bpp = res.bpp
                
                # 2. MSE of Codec (Internal domain)
                mse_internal = np.mean((img_to_compress - img_decoded) ** 2)

                # 3. Inverse Transform (if needed)
                if use_vst:
                    img_restored = vst.inverse(img_decoded)
                else:
                    img_restored = img_decoded
                
                # 4. Metrics
                results['q'].append(q)
                results['bpp'].append(bpp)
                results['file_size_kb'].append(f_size_bytes / 1024.0)
                results['mse_codec'].append(mse_internal)
                
                cr = original_size_bytes / f_size_bytes if f_size_bytes > 0 else 0
                results['cr'].append(cr)
                
                # 5. Dynamic Metrics
                for metric_name in self.metrics_to_compute:
                    func = MetricRegistry.get_metric(metric_name)
                    if func:
                        # Some metrics return tuple, we might need to handle them?
                        # But our standard metrics return float.
                        # HVS/HVSM are special in the old code, they were computed together.
                        # Now they are registered separately.
                        val = func(ref_img, img_restored)
                        results[metric_name].append(val)
                
            except Exception as e:
                print(f"Err q={q}: {e}")
                import traceback
                traceback.print_exc()
            
            if progress_callback: progress_callback(idx + 1, total)
                
        return results
