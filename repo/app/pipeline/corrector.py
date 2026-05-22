import cv2 as cv
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]     # top-left     — smallest sum
    rect[2] = pts[np.argmax(s)]     # bottom-right — largest sum
    rect[1] = pts[np.argmin(diff)]  # top-right    — smallest difference
    rect[3] = pts[np.argmax(diff)]  # bottom-left  — largest difference

    return rect


def correct_perspective(image: np.ndarray, box_xyxy: list | np.ndarray) -> np.ndarray:
    """
    Apply perspective correction to a detected plate region.

    Takes the bounding box from model.predict() in xyxy format and returns
    a flat, front-facing crop of the plate. If the plate appears axis-aligned
    (no meaningful tilt), returns a simple rectangular crop instead.

    Args:
        image:    Full BGR image as a numpy array.
        box_xyxy: [x1, y1, x2, y2] pixel coordinates from YOLO detection.

    Returns:
        Corrected plate crop as a BGR numpy array.
    """
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]

    # Clamp to image bounds
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    plate_crop = image[y1:y2, x1:x2]

    if plate_crop.size == 0:
        return plate_crop

    # Try to find a quadrilateral contour in the crop for perspective warping
    quad = _find_plate_quadrilateral(plate_crop)

    if quad is None:
        # No tilt detected — return the rectangular crop as-is
        return plate_crop

    return _warp_to_rect(plate_crop, quad)


def _find_plate_quadrilateral(plate_crop: np.ndarray) -> np.ndarray | None:
    """
    Detect the four corners of the plate within the crop.

    Returns a (4, 2) float32 array of corner points if a clear quadrilateral
    is found, or None if the plate appears flat and needs no correction.
    """
    gray    = cv.cvtColor(plate_crop, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges   = cv.Canny(blurred, threshold1=50, threshold2=150)

    # Dilate edges slightly to close small gaps
    kernel  = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    edges   = cv.dilate(edges, kernel, iterations=1)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Take the largest contour by area
    largest = max(contours, key=cv.contourArea)
    peri    = cv.arcLength(largest, closed=True)
    approx  = cv.approxPolyDP(largest, epsilon=0.02 * peri, closed=True)

    plate_h, plate_w = plate_crop.shape[:2]
    plate_area       = plate_h * plate_w

    # Only proceed if we found a quadrilateral covering most of the crop
    if len(approx) != 4:
        return None

    if cv.contourArea(approx) < 0.3 * plate_area:
        return None

    return approx.reshape(4, 2).astype("float32")


def _warp_to_rect(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Warp the detected quadrilateral to a flat rectangle.

    Computes output dimensions from the detected corner points so the
    aspect ratio of the corrected plate is preserved.
    """
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # Output width — max of top and bottom edge lengths
    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    out_w        = int(max(width_top, width_bottom))

    # Output height — max of left and right edge lengths
    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    out_h        = int(max(height_left, height_right))

    if out_w == 0 or out_h == 0:
        return image

    dst = np.array([
        [0,         0        ],
        [out_w - 1, 0        ],
        [out_w - 1, out_h - 1],
        [0,         out_h - 1],
    ], dtype="float32")

    M       = cv.getPerspectiveTransform(rect, dst)
    warped  = cv.warpPerspective(image, M, (out_w, out_h))

    return warped