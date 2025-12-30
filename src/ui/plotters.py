import matplotlib.pyplot as plt
from typing import Dict, Any
from ..interfaces import PlotterInterface

class MatplotlibPlotter(PlotterInterface):
    def plot_curves(self, results: Dict[str, Any], oop_points: Dict[str, Any]):
        """
        Visualizes the rate-distortion curves and OOP.
        results: {'linear': res_lin, 'vst': res_vst}
        oop_points: {'linear': oop_lin, 'vst': oop_vst}
        """
        res_lin = results['linear']
        res_vst = results['vst']
        oop_lin = oop_points['linear']
        oop_vst = oop_points['vst']
        
        # We need to determine what metrics to plot based on what's available
        # But for now hardcode safe defaults based on standard requests + dynamic check
        
        # Standard: PSNR, HVS-M, Codec MSE
        fig, axes = plt.subplots(1, 3, figsize=(21, 5))
        
        # 1. PSNR
        ax1 = axes[0]
        ax1.plot(res_lin['q'], res_lin['psnr'], 'b--', label='Linear')
        ax1.plot(res_vst['q'], res_vst['psnr'], 'r-', label='VST')
        ax1.scatter(oop_lin['q'], oop_lin['psnr'], s=150, c='blue', marker='*', label='OOP Linear')
        ax1.scatter(oop_vst['q'], oop_vst['psnr'], s=150, c='red', marker='*', label='OOP VST')
        ax1.set_title("PSNR vs Q")
        ax1.set_xlabel("Q")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. HVS-M (Check if exists)
        ax2 = axes[1]
        has_hvsm = 'psnr_hvsm' in res_lin and len(res_lin['psnr_hvsm']) > 0
        if has_hvsm:
            ax2.plot(res_lin['q'], res_lin['psnr_hvsm'], 'b--', label='Linear')
            ax2.plot(res_vst['q'], res_vst['psnr_hvsm'], 'r-', label='VST')
            ax2.scatter(oop_lin['q'], oop_lin['psnr_hvsm'], s=150, c='blue', marker='*')
            ax2.scatter(oop_vst['q'], oop_vst['psnr_hvsm'], s=150, c='red', marker='*')
            ax2.set_title("HVS-M vs Q")
        else:
            ax2.text(0.5, 0.5, "HVS-M Not Available", ha='center')
            
        ax2.set_xlabel("Q")
        ax2.grid(True, alpha=0.3)

        # 3. MSE Codec
        ax3 = axes[2]
        if 'mse_codec' in res_lin:
            ax3.plot(res_lin['q'], res_lin['mse_codec'], 'b--', label='Linear')
            ax3.plot(res_vst['q'], res_vst['mse_codec'], 'r-', label='VST')
            ax3.set_title("Codec MSE (Log/Lin Domain)")
        else:
            ax3.text(0.5, 0.5, "MSE Not Available", ha='center')
            
        ax3.set_xlabel("Q")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
