
import sys
import os
import numpy as np

# Add repo root check removed as we now use relative imports

try:
    # Import the pure python backend from the VENDORED local library
    from .psnr_hvsm_lib.psnr_hvsm import psnr_hvs_hvsm as _lib_psnr_hvsm
except ImportError:
    print("WARNING: Could not import vendored psnr_hvsm_lib. Using fallback/stub.")
    _lib_psnr_hvsm = None

def psnr_hvs_hvsm(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """
    Wrapper for the local PSNR-HVS-M library integration.
    Handles normalization to [0, 1] and cropping to 8x8 multiples.
    
    Args:
        img1: Reference image (any range, will be normalized)
        img2: Distorted image (any range, will be normalized)
        
    Returns:
        (psnr_hvs, psnr_hvsm)
    """
    # 1. Determine Range and Normalize
    # The library hardcodes peak=1.0 in get_psnr().
    # So we MUST normalize to [0, 1].
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Heuristic for range: if max > 1.1, assume [0, 255]
    if img1.max() > 1.1:
        img1 /= 255.0
        img2 /= 255.0
        
    # 2. Crop to 8x8
    h, w = img1.shape
    h = (h // 8) * 8
    w = (w // 8) * 8
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    
    # 3. Call Library
    if _lib_psnr_hvsm is None:
        return 0.0, 0.0

    try:
        # Function signature: (images_a, images_b, batch=False)
        # It handles batching. We pass single images (H, W).
        # Internal to_blocks expects (..., H, W).
        # hvs_mse_tiles returns tiles.
        # psnr_hvs_hvsm returns scalar means if not batch.
        
        res_hvs, res_hvsm = _lib_psnr_hvsm(img1, img2, batch=False)
        
        # Ensure scalars
        if isinstance(res_hvs, np.ndarray) and res_hvs.size == 1:
            res_hvs = float(res_hvs)
        if isinstance(res_hvsm, np.ndarray) and res_hvsm.size == 1:
            res_hvsm = float(res_hvsm)
            
        return res_hvs, res_hvsm
        
    except Exception as e:
        print(f"Error calling internal PSNR-HVS-M library: {e}")
        return 0.0, 0.0
