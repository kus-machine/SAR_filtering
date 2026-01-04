import ipywidgets as widgets
from typing import Dict, Any

from ..config import AppConfig, VSTConfig, DataConfig, ExperimentConfig, ExportConfig

class InputPanel:
    def __init__(self, config: AppConfig):
        self.cfg = config
        s = {'description_width': 'initial'}
        layout_half = widgets.Layout(width='45%')
        
        # --- Tab 1: Setup ---
        self.w_source = widgets.Dropdown(
            options=[('Files', 'file'), ('Generator', 'gen')], 
            value=config.data.source_type, 
            description='Source:', 
            style=s
        )
        self.w_path_noised = widgets.Text(value=config.data.path_noised, description='Noised Path:', style=s, layout=layout_half)
        self.w_path_ref = widgets.Text(value=config.data.path_original, description='Original Path:', style=s, layout=layout_half)
        
        self.w_gen_noise = widgets.FloatSlider(value=config.data.gen_noise_level, min=0.01, max=0.5, step=0.01, description='Noise Level:', style=s)
        
        # Visibility Logic
        self.w_source.observe(self._on_source_change, names='value')
        
        box_files = widgets.VBox([widgets.Label("File Paths:"), self.w_path_noised, self.w_path_ref])
        box_gen = widgets.VBox([widgets.Label("Generator Params:"), self.w_gen_noise])
        
        self.container_setup = widgets.VBox([self.w_source, box_files, box_gen])
        self._on_source_change({'new': self.w_source.value}) # Init state

        # --- Tab 2: VST Params ---
        self.w_a = widgets.FloatSlider(value=config.vst.a, min=1.0, max=20.0, step=0.01, description='Param a:', style=s)
        self.w_b = widgets.FloatSlider(value=config.vst.b, min=1.05, max=5.0, step=0.05, description='Param b:', style=s)
        self.container_vst = widgets.VBox([self.w_a, self.w_b])

        # --- Tab 3: Experiment ---
        self.w_q_start = widgets.IntText(value=config.experiment.q_start, description='Q Start:', style=s, layout=widgets.Layout(width='150px'))
        self.w_q_end = widgets.IntText(value=config.experiment.q_end, description='Q End:', style=s, layout=widgets.Layout(width='150px'))
        self.w_q_step = widgets.IntText(value=config.experiment.q_step, description='Q Step:', style=s, layout=widgets.Layout(width='150px'))
        
        self.w_oop_metric = widgets.Dropdown(
            options=[('PSNR', 'psnr'), ('PSNR-HVS-M', 'psnr_hvsm')],
            value=config.experiment.oop_metric,
            description='OOP Metric:',
            style=s
        )
        
        self.container_exp = widgets.VBox([
            widgets.HBox([self.w_q_start, self.w_q_end, self.w_q_step]),
            self.w_oop_metric
        ])

        # --- Tab 4: Export ---
        self.w_save_plots = widgets.Checkbox(value=config.plotting.save_plots, description='Auto-Save Plots')
        self.w_save_oop_img = widgets.Checkbox(value=config.export.save_oop_images, description='Save OOP Compressed Image')
        self.container_export = widgets.VBox([self.w_save_plots, self.w_save_oop_img])

        # --- Main Tabs ---
        self.tabs = widgets.Tab(children=[self.container_setup, self.container_vst, self.container_exp, self.container_export])
        self.tabs.set_title(0, 'Setup')
        self.tabs.set_title(1, 'VST Params')
        self.tabs.set_title(2, 'Experiment')
        self.tabs.set_title(3, 'Export')

    def _on_source_change(self, change):
        val = change['new']
        # Logic to enable/disable or hide could go here
        pass

    @property
    def widget(self):
        return self.tabs
    
    def get_config_update(self) -> AppConfig:
        """Returns updated config object based on UI state."""
        # Create a deep copy or new instance preferably, but for now we update fields
        # Note: We are returning a new AppConfig with updated values
        
        # Update components
        self.cfg.data.source_type = self.w_source.value
        self.cfg.data.path_noised = self.w_path_noised.value
        self.cfg.data.path_original = self.w_path_ref.value
        self.cfg.data.gen_noise_level = self.w_gen_noise.value
        
        self.cfg.vst.a = self.w_a.value
        self.cfg.vst.b = self.w_b.value
        
        self.cfg.experiment.q_start = self.w_q_start.value
        self.cfg.experiment.q_end = self.w_q_end.value
        self.cfg.experiment.q_step = self.w_q_step.value
        self.cfg.experiment.oop_metric = self.w_oop_metric.value
        
        self.cfg.plotting.save_plots = self.w_save_plots.value
        self.cfg.export.save_oop_images = self.w_save_oop_img.value
        
        return self.cfg
