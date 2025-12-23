import os
import subprocess
import numpy as np
import imageio.v3 as iio

class BPGCodec:
    def __init__(self, bpg_folder_path: str, temp_dir='temp'):
        self.bpg_enc = os.path.join(bpg_folder_path, 'bpgenc.exe')
        self.bpg_dec = os.path.join(bpg_folder_path, 'bpgdec.exe')
        self.temp_dir = temp_dir
        
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Check executables (Optional warning/error logic can go here)
        if not os.path.exists(self.bpg_enc):
            print(f"Warning: Encoder not found at {self.bpg_enc}")

    def compress_decompress(self, image_log: np.ndarray, q: int):
        """
        Runs the full cycle: 
        Float Log Image -> 8bit PNG -> BPG Enc -> BPG Dec -> 8bit PNG -> Float Log Image
        """
        original_shape = image_log.shape
        
        # 1. Normalize Float to 8-bit [0, 255]
        d_min, d_max = image_log.min(), image_log.max()
        
        if d_max == d_min:
            norm_img = np.zeros_like(image_log, dtype=np.uint8)
        else:
            norm_img = ((image_log - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
            
        # 2. Save Input PNG
        t_input_png = os.path.join(self.temp_dir, 'input.png')
        t_output_bpg = os.path.join(self.temp_dir, 'output.bpg')
        t_decoded_png = os.path.join(self.temp_dir, 'decoded.png')
        
        # Clean previous runs
        self._remove_safe(t_output_bpg)
        self._remove_safe(t_decoded_png)
        
        iio.imwrite(t_input_png, norm_img)
        
        # 3. BPG Encode
        # -b 8: force 8-bit depth
        # -c ycbcr / -c rgb: specific color space can be forced, but auto usually works
        cmd_enc = [self.bpg_enc, '-q', str(q), '-b', '8', '-o', t_output_bpg, t_input_png]
        
        # Suppress console window on Windows (optional)
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        res_enc = subprocess.run(cmd_enc, capture_output=True, startupinfo=startupinfo)
        
        if res_enc.returncode != 0:
            raise RuntimeError(f"BPG Encode Failed: {res_enc.stderr.decode(errors='ignore')}")
            
        if not os.path.exists(t_output_bpg):
             raise RuntimeError("BPG Encoder did not produce an output file.")

        file_size = os.path.getsize(t_output_bpg)
        
        # 4. BPG Decode
        cmd_dec = [self.bpg_dec, '-o', t_decoded_png, t_output_bpg]
        res_dec = subprocess.run(cmd_dec, capture_output=True, startupinfo=startupinfo)
        
        if res_dec.returncode != 0:
            raise RuntimeError(f"BPG Decode Failed: {res_dec.stderr.decode(errors='ignore')}")
            
        # 5. Read back and Fix Shapes
        decoded_uint8 = iio.imread(t_decoded_png)
        
        # --- SHAPE FIX LOGIC ---
        # Випадок А: Оригінал 2D (H, W), а декодер повернув 3D (H, W, 3) (RGB)
        if len(original_shape) == 2 and len(decoded_uint8.shape) == 3:
            # Беремо тільки перший канал (вони однакові для Grayscale)
            decoded_uint8 = decoded_uint8[:, :, 0]
            
        # Випадок Б: Розміри не співпадають (наприклад, педдінг)
        if decoded_uint8.shape != original_shape:
            # Обрізаємо до оригінального розміру
            h, w = original_shape
            decoded_uint8 = decoded_uint8[:h, :w]
        # -----------------------

        # Map back to float range
        decoded_float = (decoded_uint8.astype(float) / 255.0) * (d_max - d_min) + d_min
        
        # Calculate bits per pixel
        h, w = original_shape
        bpp = (file_size * 8) / (h * w)
        
        return decoded_float, file_size, bpp

    def _remove_safe(self, path):
        try:
            if os.path.exists(path): os.remove(path)
        except: pass

    def cleanup(self):
        try:
            for f in ['input.png', 'output.bpg', 'decoded.png']:
                self._remove_safe(os.path.join(self.temp_dir, f))
        except: pass