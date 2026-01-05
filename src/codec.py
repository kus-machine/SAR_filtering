import os
import subprocess
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from typing import Tuple, Optional
from shutil import which
from .interfaces import BaseCodec, EncodeResult

class BPGCodec(BaseCodec):
    def __init__(self, bpg_folder_path: str, temp_dir: str = 'temp'):
        is_windows = os.name == 'nt'
        
        # Construct full paths: Windows uses folder+program name, ARM uses program name only
        if is_windows:
            enc_name = 'bpgenc.exe'
            dec_name = 'bpgdec.exe'
            self.bpg_enc = Path(bpg_folder_path) / enc_name
            self.bpg_dec = Path(bpg_folder_path) / dec_name
        else:
            self.bpg_enc = Path('bpgenc')
            self.bpg_dec = Path('bpgdec')
        
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Check availability: Windows checks file existence, ARM checks PATH
        if is_windows:
            enc_available = self.bpg_enc.exists()
        else:
            enc_available = which('bpgenc') is not None
        
        if not enc_available:
            print(f"Warning: Encoder not found at {self.bpg_enc}")

    def _normalize_and_save_png(self, image: np.ndarray, png_path: Path) -> Tuple[float, float]:
        """Helper: Converts float image to 8-bit PNG."""
        d_min, d_max = image.min(), image.max()
        if d_max == d_min:
            norm_img = np.zeros_like(image, dtype=np.uint8)
        else:
            norm_img = ((image - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        
        iio.imwrite(png_path, norm_img)
        return d_min, d_max

    def save_to_file(self, image: np.ndarray, q: int, output_path: str) -> int:
        """Saves compressed BPG to file."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        t_input = self.temp_dir / f'temp_save_input_{os.getpid()}.png'
        
        try:
            self._normalize_and_save_png(image, t_input)
            
            cmd_enc = [str(self.bpg_enc), '-q', str(q), '-b', '8', '-o', str(out_path), str(t_input)]
            self._run_command(cmd_enc)
            
            return out_path.stat().st_size
        finally:
            if t_input.exists(): t_input.unlink()

    def compress_decompress(self, image: np.ndarray, q: int) -> EncodeResult:
        """Cycle: Float -> PNG -> BPG -> PNG -> Float"""
        pid = os.getpid()
        t_in = self.temp_dir / f'input_{pid}.png'
        t_bpg = self.temp_dir / f'output_{pid}.bpg'
        t_out = self.temp_dir / f'decoded_{pid}.png'
        
        try:
            # 1. Normalize
            d_min, d_max = self._normalize_and_save_png(image, t_in)
            
            # 2. Encode
            cmd_enc = [str(self.bpg_enc), '-q', str(q), '-b', '8', '-o', str(t_bpg), str(t_in)]
            self._run_command(cmd_enc)
            
            if not t_bpg.exists(): raise RuntimeError("BPG Enc failed")
            f_size = t_bpg.stat().st_size
            
            # 3. Decode
            cmd_dec = [str(self.bpg_dec), '-o', str(t_out), str(t_bpg)]
            self._run_command(cmd_dec)
            
            # 4. Restore
            dec_uint8 = iio.imread(t_out)
            # Handle grayscale issues (if saved as RGB)
            if image.ndim == 2 and dec_uint8.ndim == 3: 
                dec_uint8 = dec_uint8[:,:,0]
            
            # Crop padding if BPG added any (unlikely for 8x8 blocks but possible)
            if dec_uint8.shape != image.shape: 
                dec_uint8 = dec_uint8[:image.shape[0], :image.shape[1]]
                
            dec_float = (dec_uint8.astype(float) / 255.0) * (d_max - d_min) + d_min
            
            h, w = image.shape
            bpp = (f_size * 8) / (h * w)
            
            return EncodeResult(decoded_image=dec_float, file_size_bytes=f_size, bpp=bpp)
            
        finally:
            # Cleanup
            for p in [t_in, t_bpg, t_out]:
                if p.exists(): p.unlink()

    def _run_command(self, cmd):
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        res = subprocess.run(cmd, capture_output=True, startupinfo=startupinfo)
        if res.returncode != 0:
            raise RuntimeError(f"BPG Error: {res.stderr.decode(errors='ignore')}")

