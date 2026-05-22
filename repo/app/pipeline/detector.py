"""
Plate Region Detection Module

Detects license plate regions in video frames using a fine-tuned YOLOv11n model.
Loads the model once at initialization and reuses it for all subsequent frames.
Cropping and perspective correction are handled downstream by corrector.py.
"""

import numpy as np
from ultralytics import YOLO
from pathlib import Path
from huggingface_hub import hf_hub_download

from ..config import PLATE_DETECTOR_MODEL_PATH, CONF_THRESHOLD


class PlateDetector:
    """
    Detects license plate bounding boxes in a frame using a YOLO model.

    Attributes:
        model:          Loaded YOLO model, initialized once.
        conf_threshold: Minimum confidence score to accept a detection.
    """

    def __init__(
        self,
        model_path: str = None,
        hf_repo: str = None,
        conf_threshold: float = CONF_THRESHOLD,
    ):
        """
        Initialize the PlateDetector.

        Args:
            model_path:     Path to local YOLO model file.
            hf_repo:        Hugging Face repo ID (e.g., 'username/model_name').
            conf_threshold: Confidence threshold for detections (0.0 to 1.0).

        Raises:
            FileNotFoundError: If the model file does not exist at model_path.
        """
        if hf_repo:
            # Download from Hugging Face
            model_path = hf_hub_download(
                repo_id=hf_repo,
                filename="best.pt",
                token=True,  # Uses HF_TOKEN from environment
            )
        elif model_path is None:
            model_path = PLATE_DETECTOR_MODEL_PATH

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Plate detector model not found: {model_path}")

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        """
        Detect license plate regions in a frame.

        Does not crop or correct perspective — returns raw bounding boxes only.
        Pass each box to corrector.correct_perspective() in the pipeline.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            List of bounding boxes, each as [x1, y1, x2, y2] in pixel coordinates.
            Empty list if no plates are detected or frame fails quality check.

        Raises:
            ValueError: If the frame is None or has zero size.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame — frame is None or empty.")

        if not self._is_frame_usable(frame):
            return []

        results   = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        boxes_xyxy = results[0].boxes.xyxy

        return [box.tolist() for box in boxes_xyxy]

    @staticmethod
    def _is_frame_usable(
        frame: np.ndarray,
        variance_threshold: float = 100.0,
    ) -> bool:
        """
        Return True if the frame has enough contrast and detail to process.

        Rejects frames that are too dark, too bright, or heavily occluded.

        Args:
            frame:              BGR image as a numpy array.
            variance_threshold: Minimum pixel variance to accept the frame.

        Returns:
            True if the frame is usable, False otherwise.
        """
        import cv2 as cv
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return float(np.var(gray)) >= variance_threshold