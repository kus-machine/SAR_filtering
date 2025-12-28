import os
import subprocess
import numpy as np
import imageio.v3 as iio
import shutil

class BPGCodec:
    def __init__(self, bpg_folder_path: str, temp_dir='temp'):
        self.bpg_enc = os.path.join(bpg_folder_path, 'bpgenc.exe')
        self.bpg_dec = os.path.join(bpg_folder_path, 'bpgdec.exe')
        self.temp_dir = temp_dir
        
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        if not os.path.exists(self.bpg_enc):
            print(f"Warning: Encoder not found at {self.bpg_enc}")

    def _normalize_and_save_png(self, image: np.ndarray, png_path: str):
        """Helper: Converts float image to 8-bit PNG."""
        d_min, d_max = image.min(), image.max()
        if d_max == d_min:
            norm_img = np.zeros_like(image, dtype=np.uint8)
        else:
            norm_img = ((image - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        
        iio.imwrite(png_path, norm_img)
        return d_min, d_max

    def save_compressed_file(self, image: np.ndarray, q: int, output_path: str):
        """
        Saves the image as a BPG file to the specified output_path.
        Args:
            image: Float image (Linear or Log domain).
            q: Quantizer value.
            output_path: Where to save the result (e.g., 'results/best.bpg').
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        t_input = os.path.join(self.temp_dir, 'temp_save_input.png')
        
        # 1. Convert to 8-bit PNG
        self._normalize_and_save_png(image, t_input)
        
        # 2. Run Encoder
        # Direct output to final path
        cmd_enc = [self.bpg_enc, '-q', str(q), '-b', '8', '-o', output_path, t_input]
        
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        res = subprocess.run(cmd_enc, capture_output=True, startupinfo=startupinfo)
        
        if res.returncode != 0:
            raise RuntimeError(f"BPG Save Error: {res.stderr.decode(errors='ignore')}")
            
        # Clean temp png
        if os.path.exists(t_input): os.remove(t_input)
        
        return os.path.getsize(output_path)

    def compress_decompress(self, image_log: np.ndarray, q: int):
        """Cycle: Float -> PNG -> BPG -> PNG -> Float"""
        t_in = os.path.join(self.temp_dir, 'input.png')
        t_bpg = os.path.join(self.temp_dir, 'output.bpg')
        t_out = os.path.join(self.temp_dir, 'decoded.png')
        
        self._remove_safe(t_bpg)
        self._remove_safe(t_out)
        
        # 1. Normalize
        d_min, d_max = self._normalize_and_save_png(image_log, t_in)
        
        # 2. Encode
        cmd_enc = [self.bpg_enc, '-q', str(q), '-b', '8', '-o', t_bpg, t_in]
        
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        subprocess.run(cmd_enc, capture_output=True, startupinfo=startupinfo)
        
        if not os.path.exists(t_bpg): raise RuntimeError("BPG Enc failed")
        f_size = os.path.getsize(t_bpg)
        
        # 3. Decode
        cmd_dec = [self.bpg_dec, '-o', t_out, t_bpg]
        subprocess.run(cmd_dec, capture_output=True, startupinfo=startupinfo)
        
        # 4. Restore
        dec_uint8 = iio.imread(t_out)
        if len(image_log.shape) == 2 and dec_uint8.ndim == 3: dec_uint8 = dec_uint8[:,:,0]
        if dec_uint8.shape != image_log.shape: dec_uint8 = dec_uint8[:image_log.shape[0], :image_log.shape[1]]
            
        dec_float = (dec_uint8.astype(float) / 255.0) * (d_max - d_min) + d_min
        
        h, w = image_log.shape
        bpp = (f_size * 8) / (h * w)
        
        return dec_float, f_size, bpp

    def _remove_safe(self, path):
        try:
            if os.path.exists(path): os.remove(path)
        except: pass

