from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import platform

@dataclass
class VSTConfig:
    """Configuration for VST transformation parameters."""
    a: float = 8.39
    b: float = 1.2
    epsilon: float = 1.0

@dataclass
class DataConfig:
    """Configuration for data source."""
    source_type: str = 'file' # 'file' or 'gen'
    path_noised: str = 'data/NOISED.tiff'
    path_original: str = 'data/ORIGINAL.tiff'
    gen_noise_level: float = 0.05

@dataclass
class ExperimentConfig:
    """Configuration for encoding experiment."""
    q_start: int = 20
    q_end: int = 51
    q_step: int = 1
    metrics: List[str] = field(default_factory=lambda: ['psnr', 'psnr_hvsm', 'ssim', 'mse_codec'])
    oop_metric: str = 'psnr' # 'psnr' or 'psnr_hvsm'

@dataclass
class PlottingConfig:
    """Configuration for plotting and saving."""
    figsize: Tuple[int, int] = (21, 5)
    colors: Dict[str, str] = field(default_factory=lambda: {'linear': 'b', 'vst': 'r'})
    markers: Dict[str, str] = field(default_factory=lambda: {'linear': '--', 'vst': '-'})
    cmap: str = 'bwr'
    save_plots: bool = True
    save_format: str = 'png'
    dpi: int = 300
    show_plots: bool = True

@dataclass
class ExportConfig:
    """Configuration for exporting results."""
    save_csv: bool = False
    save_oop_images: bool = False
    results_dir: str = 'results'

@dataclass
class AppConfig:
    """Root configuration for the application."""
    bpg_path: str = field(default_factory=lambda: 'bpg-0.9.8-win64' if platform.system() == 'Windows' else 'libbpg')
    vst: VSTConfig = field(default_factory=VSTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """Helper to load from a dictionary (e.g. from notebook)."""
        # A simple recursive loader could be implemented here or external lib used.
        # For simplicity, we assume the user instantiates classes or we map manually if needed.
        # But providing a direct initializer is usually enough.
        pass