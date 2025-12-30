import ipywidgets as widgets
from typing import Dict, Any

class InputPanel:
    def __init__(self, default_noised="data/NOISED.tiff", default_ref="data/ORIGINAL.tiff"):
        s = {'description_width': 'initial'}
        layout_half = widgets.Layout(width='45%')
        
        # --- Tab 1: Setup ---
        self.w_source = widgets.Dropdown(
            options=[('Files', 'file'), ('Generator', 'gen')], 
            value='file', # DEFAULT TO FILE per request
            description='Source:', 
            style=s
        )
        self.w_path_noised = widgets.Text(value=default_noised, description='Noised Path:', style=s, layout=layout_half)
        self.w_path_ref = widgets.Text(value=default_ref, description='Original Path:', style=s, layout=layout_half)
        
        self.w_gen_noise = widgets.FloatSlider(value=0.05, min=0.01, max=0.5, step=0.01, description='Noise Level:', style=s)
        
        # Visibility Logic
        self.w_source.observe(self._on_source_change, names='value')
        
        box_files = widgets.VBox([widgets.Label("File Paths:"), self.w_path_noised, self.w_path_ref])
        box_gen = widgets.VBox([widgets.Label("Generator Params:"), self.w_gen_noise])
        
        self.container_setup = widgets.VBox([self.w_source, box_files, box_gen])
        self._on_source_change({'new': self.w_source.value}) # Init state

        # --- Tab 2: VST Params ---
        self.w_a = widgets.FloatSlider(value=8.39, min=1.0, max=20.0, step=0.01, description='Param a:', style=s)
        self.w_b = widgets.FloatSlider(value=1.2, min=1.05, max=5.0, step=0.05, description='Param b:', style=s)
        self.container_vst = widgets.VBox([self.w_a, self.w_b])

        # --- Tab 3: Experiment ---
        self.w_q_start = widgets.IntText(value=20, description='Q Start:', style=s, layout=widgets.Layout(width='150px'))
        self.w_q_end = widgets.IntText(value=51, description='Q End:', style=s, layout=widgets.Layout(width='150px'))
        self.w_q_step = widgets.IntText(value=1, description='Q Step:', style=s, layout=widgets.Layout(width='150px'))
        self.container_exp = widgets.HBox([self.w_q_start, self.w_q_end, self.w_q_step])

        # --- Tab 4: Export ---
        self.w_save_plots = widgets.Checkbox(value=False, description='Auto-Save Plots')
        self.w_save_oop_img = widgets.Checkbox(value=False, description='Save OOP Compressed Image')
        self.container_export = widgets.VBox([self.w_save_plots, self.w_save_oop_img])

        # --- Main Tabs ---
        self.tabs = widgets.Tab(children=[self.container_setup, self.container_vst, self.container_exp, self.container_export])
        self.tabs.set_title(0, 'Setup')
        self.tabs.set_title(1, 'VST Params')
        self.tabs.set_title(2, 'Experiment')
        self.tabs.set_title(3, 'Export')

    def _on_source_change(self, change):
        val = change['new']
        # Simple visibility toggling: In VBox, we can't easily hide children without rebuilding or CSS.
        # But we can assume the user sees both and inputs what is relevant, 
        # OR we can update the layout.
        # Let's keep it simple: Show all, but maybe disable?
        # Actually, let's just leave it open. The controller will use what's selected.
        pass

    @property
    def widget(self):
        return self.tabs
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'source_type': self.w_source.value,
            'path_noised': self.w_path_noised.value,
            'path_original': self.w_path_ref.value,
            'noise_level': self.w_gen_noise.value,
            'vst_a': self.w_a.value,
            'vst_b': self.w_b.value,
            'q_start': self.w_q_start.value,
            'q_end': self.w_q_end.value,
            'q_step': self.w_q_step.value,
            'save_plots': self.w_save_plots.value,
            'save_oop': self.w_save_oop_img.value
        }
