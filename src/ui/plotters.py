import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import os
from ..interfaces import PlotterInterface
from ..config import AppConfig

class MatplotlibPlotter(PlotterInterface):
    def __init__(self, config: AppConfig):
        self.cfg = config
        
    def _save_plot(self, fig, filename: str):
        """Helper to save plot if enabled."""
        if self.cfg.plotting.save_plots:
            os.makedirs(self.cfg.export.results_dir, exist_ok=True)
            
            base_name = "Generated"
            if self.cfg.data.source_type == 'file':
                # Extract basename without extension
                fname = os.path.basename(self.cfg.data.path_noised)
                base_name, _ = os.path.splitext(fname)
            
            save_name = f"{base_name}_{filename}.{self.cfg.plotting.save_format}"
            path = os.path.join(self.cfg.export.results_dir, save_name)
            
            fig.savefig(path, dpi=self.cfg.plotting.dpi, bbox_inches='tight')
            print(f"Saved plot: {path}")

    def plot_curves(self, results: Dict[str, Any], oop_points: Dict[str, Any]):
        """
        Visualizes the rate-distortion curves and OOP.
        """
        res_lin = results['linear']
        res_vst = results['vst']
        oop_lin = oop_points['linear']
        oop_vst = oop_points['vst']
        
        # Unpack styling
        colors = self.cfg.plotting.colors
        markers = self.cfg.plotting.markers
        figsize = self.cfg.plotting.figsize
        
        # We create a single figure for notebook display
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Helper to plot on axis
        def plot_psnr(ax):
            ax.plot(res_lin['q'], res_lin['psnr'], color=colors['linear'], linestyle=markers['linear'], label='Linear')
            ax.plot(res_vst['q'], res_vst['psnr'], color=colors['vst'], linestyle=markers['vst'], label='VST')
            ax.scatter(oop_lin['q'], oop_lin['psnr'], s=150, c=colors['linear'], marker='*', label='OOP Linear')
            ax.scatter(oop_vst['q'], oop_vst['psnr'], s=150, c=colors['vst'], marker='*', label='OOP VST')
            ax.set_title("PSNR vs Q")
            ax.set_xlabel("Q")
            ax.grid(True, alpha=0.3)
            ax.legend()

        def plot_hvsm(ax):
            has_hvsm = 'psnr_hvsm' in res_lin and len(res_lin['psnr_hvsm']) > 0
            if has_hvsm:
                ax.plot(res_lin['q'], res_lin['psnr_hvsm'], color=colors['linear'], linestyle=markers['linear'], label='Linear')
                ax.plot(res_vst['q'], res_vst['psnr_hvsm'], color=colors['vst'], linestyle=markers['vst'], label='VST')
                ax.scatter(oop_lin['q'], oop_lin['psnr_hvsm'], s=150, c=colors['linear'], marker='*', label='OOP Linear')
                ax.scatter(oop_vst['q'], oop_vst['psnr_hvsm'], s=150, c=colors['vst'], marker='*', label='OOP VST')
                ax.set_title("HVS-M vs Q")
                ax.legend() # Added Legend
            else:
                ax.text(0.5, 0.5, "HVS-M Not Available", ha='center')
            ax.set_xlabel("Q")
            ax.grid(True, alpha=0.3)

        def plot_mse(ax):
            if 'mse_codec' in res_lin:
                ax.plot(res_lin['q'], res_lin['mse_codec'], color=colors['linear'], linestyle=markers['linear'], label='Linear')
                ax.plot(res_vst['q'], res_vst['mse_codec'], color=colors['vst'], linestyle=markers['vst'], label='VST')
                ax.set_title("Codec MSE (Log/Lin Domain)")
            else:
                ax.text(0.5, 0.5, "MSE Not Available", ha='center')
            ax.set_xlabel("Q")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 1. PSNR
        plot_psnr(axes[0])
        # 2. HVS-M
        plot_hvsm(axes[1])
        # 3. MSE Codec
        plot_mse(axes[2])
        
        plt.tight_layout()
        
        # Save Combined (optional, but keep for consistency) or skip
        # self._save_plot(fig, "Curves_Combined")
        
        # Save Separate High-Res Plots
        if self.cfg.plotting.save_plots:
            # Create a separate figure for each to ensure simple saving without complex subplot slicing
            # PSNR
            f1, a1 = plt.subplots(figsize=(8, 6))
            plot_psnr(a1)
            self._save_plot(f1, "Plot_PSNR")
            plt.close(f1)
            
            # HVS-M
            f2, a2 = plt.subplots(figsize=(8, 6))
            plot_hvsm(a2)
            self._save_plot(f2, "Plot_HVSM")
            plt.close(f2)
            
            # MSE
            f3, a3 = plt.subplots(figsize=(8, 6))
            plot_mse(a3)
            self._save_plot(f3, "Plot_MSE")
            plt.close(f3)

        if self.cfg.plotting.show_plots:
            plt.show() # Display the combined one in notebook
        else:
            plt.close(fig)

    def plot_error_maps(self, result: Any):
        """
        Plots relative error maps for OOP images.
        """
        from ..metrics import QualityMetrics
        import numpy as np
        
        img_lin = result.oop_image_lin
        img_vst = result.oop_image_vst
        ref = result.ref_image
        
        if img_lin is None or img_vst is None or ref is None:
            print("OOP Images not available for error map plotting.")
            return

        map_lin = QualityMetrics.compute_relative_error_map(ref, img_lin)
        map_vst = QualityMetrics.compute_relative_error_map(ref, img_vst)
        
        # Notebook Display: Combined
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        cmap_name = self.cfg.plotting.cmap
        
        def plot_map(ax, data, title, q):
            im = ax.imshow(data, cmap=cmap_name, vmin=0, vmax=255)
            ax.set_title(f"{title}\nQ={q}\n(Blue: Loss, White: Accurate, Red: Excess)")
            ax.axis('off')
            return im

        im1 = plot_map(axes[0], map_lin, "Relative Error Map (Standard)", result.oop_points['linear'].get('q', '?'))
        cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cb1.set_label('Relative Error Bias (128=0%)')
        
        im2 = plot_map(axes[1], map_vst, "Relative Error Map (VST)", result.oop_points['vst'].get('q', '?'))
        cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cb2.set_label('Relative Error Bias (128=0%)')
        
        plt.tight_layout()
        
        # Save Separate
        if self.cfg.plotting.save_plots:
             # Lin
             f1, a1 = plt.subplots(figsize=(8, 8))
             i1 = plot_map(a1, map_lin, "Relative Error Map (Standard)", result.oop_points['linear'].get('q', '?'))
             # c1 = f1.colorbar(i1, ax=a1, fraction=0.046, pad=0.04)
             self._save_plot(f1, "ErrorMap_Linear")
             plt.close(f1)
             
             # VST
             f2, a2 = plt.subplots(figsize=(8, 8))
             i2 = plot_map(a2, map_vst, "Relative Error Map (VST)", result.oop_points['vst'].get('q', '?'))
             # c2 = f2.colorbar(i2, ax=a2, fraction=0.046, pad=0.04)
             self._save_plot(f2, "ErrorMap_VST")
             plt.close(f2)
        
        if self.cfg.plotting.show_plots:
            plt.show()
        else:
            plt.close(fig)
