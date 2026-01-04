from typing import Optional, Dict, Any
import ipywidgets as widgets
from IPython.display import display
from .widgets import InputPanel
from .plotters import MatplotlibPlotter
from ..app_logic import AnalysisController
from ..config import AppConfig

class AnalysisUI:
    def __init__(self, config: Optional[AppConfig] = None):
        if config is None:
            self.cfg = AppConfig()
        else:
            self.cfg = config
            
        self.controller = AnalysisController(bpg_path=self.cfg.bpg_path)
        self.panel = InputPanel(self.cfg)
        self.plotter = MatplotlibPlotter(self.cfg)
        
        # Action Buttons
        self.btn_run = widgets.Button(description='Run Analysis', button_style='primary', icon='play')
        self.btn_save_csv = widgets.Button(description='Save CSV', button_style='success', icon='file-text')
        # Button for manual plot saving removed as per request (auto-save preferred)
        # self.btn_save_plot = widgets.Button(description='Save Plots', button_style='info', icon='image')
        
        self.btn_run.on_click(self.on_run)
        self.btn_save_csv.on_click(self.on_save_csv)
        
        self.output = widgets.Output()
        self.prog_bar = widgets.IntProgress(value=0, min=0, max=100, layout=widgets.Layout(width='100%'))
        self.prog_bar.layout.visibility = 'hidden'

        self.layout = widgets.VBox([
            widgets.HTML("<h2>Advanced SAR Analysis Framework</h2>"),
            self.panel.widget,
            widgets.HBox([self.btn_run, self.prog_bar]),
            widgets.HBox([self.btn_save_csv]),
            self.output
        ])

    def on_run(self, b):
        self.output.clear_output()
        self.btn_run.disabled = True
        self.prog_bar.layout.visibility = 'visible'
        
        # Get updated config from panel
        self.cfg = self.panel.get_config_update()
        
        try:
            with self.output:
                res = self.controller.run_analysis(
                    source_type=self.cfg.data.source_type,
                    noise_level=self.cfg.data.gen_noise_level,
                    path_noised=self.cfg.data.path_noised,
                    path_original=self.cfg.data.path_original,
                    vst_a=self.cfg.vst.a,
                    vst_b=self.cfg.vst.b,
                    q_start=self.cfg.experiment.q_start,
                    q_end=self.cfg.experiment.q_end,
                    q_step=self.cfg.experiment.q_step,
                    oop_metric=self.cfg.experiment.oop_metric
                )
                
                # Display DataFrame
                display(res.metrics_df)
                
                # Auto-Save Results if configured
                if self.cfg.export.save_csv:
                     self.on_save_csv(None)

                # Plot (will auto-save if configured)
                self.plotter.plot_curves(res.curves, res.oop_points)
                self.plotter.plot_error_maps(res)
                
                # Auto-Save OOP Image logic
                if self.cfg.export.save_oop_images:
                    self.controller.save_oop_image(res, 'linear', self.cfg.export.results_dir)
                    self.controller.save_oop_image(res, 'vst', self.cfg.export.results_dir)
        
        except Exception as e:
            with self.output:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self.btn_run.disabled = False
            self.prog_bar.layout.visibility = 'hidden'

    def on_save_csv(self, b):
        if self.controller.last_result:
            import os
            os.makedirs(self.cfg.export.results_dir, exist_ok=True)
            path = os.path.join(self.cfg.export.results_dir, "metrics.csv")
            self.controller.save_results_csv(self.controller.last_result, path)
            with self.output: print(f"Saved CSV to {path}")

    def show(self):
        display(self.layout)
