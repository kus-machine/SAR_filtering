
import numpy as np
from scipy import fftpack

def dct_8x8(img):
    """
    Performs 8x8 block-wise DCT.
    """
    h, w = img.shape
    # Reshape to (h//8, 8, w//8, 8) -> (blocks_h, blocks_w, 8, 8)
    blocks = img.reshape(h//8, 8, w//8, 8).transpose(0, 2, 1, 3)
    
    # Run DCT on last two axes
    dct_blocks = fftpack.dctn(blocks, axes=(2, 3), type=2, norm='ortho')
    return dct_blocks

def get_csf_matrix():
    """
    Returns standard 8x8 CSF weighting matrix for HVS metrics.
    Based on standard approximations (e.g. Nill or Ahumada).
    """
    # This standard table resembles the inverse of the JPEG quantization matrix 
    # tailored for visual sensitivity.
    # Values approx from Ponomarenko et al. works.
    csf = np.array([
        [1.62, 2.72, 2.37, 1.48, 0.99, 0.70, 0.52, 0.40],
        [2.65, 3.42, 2.68, 1.57, 1.05, 0.73, 0.54, 0.41],
        [2.31, 2.76, 2.08, 1.25, 0.84, 0.59, 0.44, 0.33],
        [1.48, 1.63, 1.25, 0.78, 0.53, 0.38, 0.29, 0.22],
        [0.98, 1.05, 0.81, 0.52, 0.36, 0.26, 0.20, 0.15],
        [0.69, 0.72, 0.56, 0.37, 0.26, 0.19, 0.14, 0.11],
        [0.51, 0.54, 0.42, 0.28, 0.20, 0.14, 0.11, 0.08],
        [0.39, 0.41, 0.32, 0.21, 0.15, 0.11, 0.08, 0.06]
    ])
    return csf

def psnr_hvs_hvsm(img1, img2):
    """
    Computes PSNR-HVS and PSNR-HVS-M (approx).
    
    Args:
        img1: Reference image (grayscale, [0, 255] or [0, 1] -- will assume it matches img2 domain)
        img2: Distorted image
        
    Returns:
        (psnr_hvs, psnr_hvsm)
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Dimensions
    h, w = img1.shape
    h = (h // 8) * 8
    w = (w // 8) * 8
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    
    # 1. Block DCT
    coefs1 = dct_8x8(img1)
    coefs2 = dct_8x8(img2)
    
    # 2. CSF Weighting
    csf = get_csf_matrix()
    
    # Expand CSF to broadcast: (1, 1, 8, 8)
    csf_grid = csf.reshape(1, 1, 8, 8)
    
    # --- PSNR-HVS Calculation ---
    # Error in DCT domain weighted by CSF
    diff = (coefs1 - coefs2) * csf_grid
    mse_hvs = np.mean(diff ** 2)
    
    if mse_hvs == 0:
        psnr_hvs = 100.0
    else:
        # Dynamic range assumed 255 usually for standard metrics
        # If inputs are 0-1, this should be 1. But standard is usually 255.
        # We'll check max value or assume 255 if > 1.
        peak = 255.0
        if img1.max() <= 1.01: peak = 1.0
        
        psnr_hvs = 10 * np.log10(peak**2 / mse_hvs)
        
    # --- PSNR-HVS-M Calculation (Masking) ---
    # Masking effect: High energy in a frequency band masks errors in that band.
    # Simple model: Effective_Error = Error / max(1, w * |Coeff|)
    # Ponomarenko's model is more complex, involving regional contrast.
    # For this port, we will use a simplified coefficient masking.
    
    # Masking threshold based on reference coefficients
    # mask = 1 + k * |coefs1| * csf
    # This is a common form (Watson).
    # k usually around 0.1 ~ 0.3 for visual masking.
    k_mask = 0.2
    
    masking_factor = 1.0 + k_mask * (np.abs(coefs1) * csf_grid)
    
    # Apply masking to the difference
    diff_masked = diff / masking_factor
    
    mse_hvsm = np.mean(diff_masked ** 2)
    
    if mse_hvsm == 0:
        psnr_hvsm = 100.0
    else:
        psnr_hvsm = 10 * np.log10(peak**2 / mse_hvsm)
        
    return psnr_hvs, psnr_hvsm
