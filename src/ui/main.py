import ipywidgets as widgets
from IPython.display import display
from .widgets import InputPanel
from .plotters import MatplotlibPlotter
from ..app_logic import AnalysisController

class AnalysisUI:
    def __init__(self):
        self.controller = AnalysisController()
        self.panel = InputPanel()
        self.plotter = MatplotlibPlotter()
        
        # Action Buttons
        self.btn_run = widgets.Button(description='Run Analysis', button_style='primary', icon='play')
        self.btn_save_csv = widgets.Button(description='Save CSV', button_style='success', icon='file-text')
        self.btn_save_plot = widgets.Button(description='Save Plots', button_style='info', icon='image')
        
        self.btn_run.on_click(self.on_run)
        self.btn_save_csv.on_click(self.on_save_csv)
        self.btn_save_plot.on_click(self.on_save_plot)
        
        self.output = widgets.Output()
        self.prog_bar = widgets.IntProgress(value=0, min=0, max=100, layout=widgets.Layout(width='100%'))
        self.prog_bar.layout.visibility = 'hidden'

        self.layout = widgets.VBox([
            widgets.HTML("<h2>Advanced SAR Analysis Framework</h2>"),
            self.panel.widget,
            widgets.HBox([self.btn_run, self.prog_bar]),
            widgets.HBox([self.btn_save_csv, self.btn_save_plot]),
            self.output
        ])

    def on_run(self, b):
        self.output.clear_output()
        self.btn_run.disabled = True
        self.prog_bar.layout.visibility = 'visible'
        
        cfg = self.panel.get_config()
        
        try:
            with self.output:
                res = self.controller.run_analysis(
                    source_type=cfg['source_type'],
                    noise_level=cfg['noise_level'],
                    path_noised=cfg['path_noised'],
                    path_original=cfg['path_original'],
                    vst_a=cfg['vst_a'],
                    vst_b=cfg['vst_b'],
                    q_start=cfg['q_start'],
                    q_end=cfg['q_end'],
                    q_step=cfg['q_step']
                )
                
                # Display DataFrame
                display(res.metrics_df)
                
                # Plot
                self.plotter.plot_curves(res.curves, res.oop_points)
                
                # Auto-Save logic
                if cfg['save_oop']:
                    print("Saving OOP images...")
                    # Implementation pending in controller, but hooked up here
                    # self.controller.save_oop_image(res, 'vst')
        
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
            path = "results/metrics.csv"
            self.controller.save_results_csv(self.controller.last_result, path)
            with self.output: print(f"Saved CSV to {path}")

    def on_save_plot(self, b):
        with self.output: print("Saving plots feature requires capturing the plot object. For now, use Right Click -> Save Image on the plot.")

    def show(self):
        display(self.layout)
