# File: error_measures.py
import numpy as np
import cv2 as cv
from skimage.metrics import structural_similarity

"""
    Computes the Mean Squared Error (MSE) between two images
    Identical images have MSE = 0
"""


def mse(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError('Images must have the same dimensions')

    difference = image1.astype(np.float64) - image2.astype(np.float64)

    return np.mean(difference ** 2)


"""
    Computes Peak Signal-to-Noise Ratio (PSNR) between two images
    The unit is decibels (dB)
    Identical images have PSNR = inf
    
      PSNR	    Interpretation
      < 20 dB	Poor
      20–30 dB	Noticeable differences
      30–40 dB	Usually good
      40–50 dB	Very good
      > 50 dB	Extremely similar
"""


def psnr(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError('Images must have the same dimensions')

    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    mse = np.mean((image1 - image2) ** 2)

    if mse == 0:
        return float('inf')

    max_value = 255.0

    return 10 * np.log10((max_value ** 2) / mse)


"""
    Computes SSIM - Structural Similarity Index (SSIM) between two images
    Identical images have SSIM = 1
    
    MSE and PSNR compare pixels independently
    SSIM tries to model something closer to how humans perceive images

    SSIM compares three properties in local regions:
        a) Luminance
        b) Contrast
        c) Structure
        
    Typically:
    -1 <= SSIM <= 1
    For ordinary nonnegative images the practical range is usually:
    0 <= SSIM <= 1

"""


def ssim(image1, image2):
    score = structural_similarity(image1, image2, data_range=255)
    return score

"""
    Computes Perceptual Color Difference (PCD) between two images
    
    PCD is a numerical measure of how distinct two colors appear to the human eye
"""
def pcd(image1, image2):
    image1_float = image1.astype(np.float32) / 255.0
    image2_float = image2.astype(np.float32) / 255.0

    lab1 = cv.cvtColor(image1_float, cv.COLOR_BGR2Lab)
    lab2 = cv.cvtColor(image2_float, cv.COLOR_BGR2Lab)

    delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))

    mean_delta_e = np.mean(delta_e)

    return mean_delta_e

