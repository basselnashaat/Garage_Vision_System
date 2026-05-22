"""
Image Enhancement Module

Applies preprocessing to improve OCR accuracy:
- Noise reduction (bilateral filter)
- Contrast enhancement (CLAHE)
- Optional upscaling for small plates
"""

import cv2 as cv
import numpy as np


def enhance(plate_img: np.ndarray) -> np.ndarray:
    """
    Enhance a plate image to improve OCR accuracy.

    Applies:
        1. Bilateral filtering for noise reduction while preserving edges
        2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        3. Morphological operations if needed

    Args:
        plate_img: BGR plate image as a numpy array.

    Returns:
        Enhanced plate image as a numpy array.

    Raises:
        ValueError: If the input image is None or empty.
    """
    if plate_img is None or plate_img.size == 0:
        raise ValueError("Invalid input plate image — None or empty.")

    # Convert to grayscale for processing
    if len(plate_img.shape) == 3 and plate_img.shape[2] == 3:
        gray = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)
    else:
        gray = plate_img

    # 1. Bilateral filtering — reduces noise while preserving edges
    filtered = cv.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)

    # 2. CLAHE — adaptive histogram equalization for better contrast
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)

    # 3. Optional morphological operations for small or degraded plates
    # Apply a slight morphological closing to fill small holes
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    enhanced = cv.morphologyEx(enhanced, cv.MORPH_CLOSE, kernel, iterations=1)

    # 4. Upscale if the plate is too small (less than 100 pixels wide)
    if enhanced.shape[1] < 100:
        scale_factor = max(1.5, 100 / enhanced.shape[1])
        new_width = int(enhanced.shape[1] * scale_factor)
        new_height = int(enhanced.shape[0] * scale_factor)
        enhanced = cv.resize(enhanced, (new_width, new_height), interpolation=cv.INTER_CUBIC)

    # 5. Thresholding for better character separation (optional, can be aggressive)
    # Using Otsu's thresholding for automatic threshold selection
    _, enhanced = cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Convert back to BGR to maintain consistency with pipeline
    if len(plate_img.shape) == 3:
        enhanced = cv.cvtColor(enhanced, cv.COLOR_GRAY2BGR)

    return enhanced
