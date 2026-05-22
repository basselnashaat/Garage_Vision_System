"""
Video capture module — OpenCV VideoCapture + YOLOv11 vehicle detection.

Reads a video file or camera stream, selects sharp frames, crops one vehicle
per trigger, and yields crops for the LPR coordinator (no HTTP).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np

from .. import config as app_config

logger = logging.getLogger(__name__)

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck


@dataclass
class CaptureSettings:
    """Runtime settings for VideoCapturer (defaults from app.config)."""

    process_every_n_frames: int = 3
    frame_buffer_size: int = 20
    motion_threshold: int = 1500
    mog2_history: int = 500
    mog2_var_threshold: int = 50
    cooldown_seconds: float = 2.0
    trigger_mode: str = "yolo"
    vehicle_model: str = "yolo11n.pt"
    vehicle_conf: float = 0.40
    vehicle_min_box_area: int = 3000
    vehicle_crop_padding: int = 50
    vehicle_filter_enabled: bool = True
    max_vehicles_per_trigger: int = 1
    min_sharpness_score: float = 5.0
    min_brightness: int = 30
    max_brightness: int = 220
    dedup_iou: float = 0.45
    max_triggers: int | None = None  # cap API runs on long videos

    @classmethod
    def from_app_config(cls) -> CaptureSettings:
        return cls(
            process_every_n_frames=getattr(app_config, "CAPTURE_PROCESS_EVERY_N_FRAMES", 3),
            frame_buffer_size=getattr(app_config, "CAPTURE_FRAME_BUFFER_SIZE", 20),
            motion_threshold=getattr(app_config, "CAPTURE_MOTION_THRESHOLD", 1500),
            mog2_history=getattr(app_config, "CAPTURE_MOG2_HISTORY", 500),
            mog2_var_threshold=getattr(app_config, "CAPTURE_MOG2_VAR_THRESHOLD", 50),
            cooldown_seconds=getattr(app_config, "CAPTURE_COOLDOWN_SECONDS", 2.0),
            trigger_mode=getattr(app_config, "CAPTURE_TRIGGER_MODE", "yolo"),
            vehicle_model=getattr(app_config, "CAPTURE_VEHICLE_MODEL", "yolo11n.pt"),
            vehicle_conf=getattr(app_config, "CAPTURE_VEHICLE_CONF", 0.40),
            vehicle_min_box_area=getattr(app_config, "CAPTURE_VEHICLE_MIN_BOX_AREA", 3000),
            vehicle_crop_padding=getattr(app_config, "CAPTURE_VEHICLE_CROP_PADDING", 50),
            vehicle_filter_enabled=getattr(app_config, "CAPTURE_VEHICLE_FILTER_ENABLED", True),
            max_vehicles_per_trigger=getattr(app_config, "CAPTURE_MAX_VEHICLES_PER_TRIGGER", 1),
            min_sharpness_score=getattr(app_config, "CAPTURE_MIN_SHARPNESS_SCORE", 5.0),
            min_brightness=getattr(app_config, "CAPTURE_MIN_BRIGHTNESS", 30),
            max_brightness=getattr(app_config, "CAPTURE_MAX_BRIGHTNESS", 220),
            dedup_iou=getattr(app_config, "CAPTURE_DEDUP_IOU", 0.45),
            max_triggers=getattr(app_config, "CAPTURE_MAX_TRIGGERS", None),
        )


@dataclass
class VehicleCropEvent:
    """One vehicle crop ready for the LPR pipeline."""

    crop: np.ndarray
    bbox: list[int]
    label: str
    conf: float
    frame_index: int
    trigger_index: int
    sharpness: float = 0.0


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


class VideoCapturer:
    """
    VideoCapture → YOLOv11 → one vehicle crop per trigger.

    Use iter_vehicle_crops() to drive the LPR pipeline without HTTP.
    """

    def __init__(self, settings: CaptureSettings | None = None):
        self.settings = settings or CaptureSettings.from_app_config()
        self._vehicle_model = None

    def _load_vehicle_model(self):
        if self._vehicle_model is None:
            from ultralytics import YOLO
            self._vehicle_model = YOLO(self.settings.vehicle_model)
            logger.info("Vehicle detector loaded (%s)", self.settings.vehicle_model)
        return self._vehicle_model

    def detect_vehicles(self, frame: np.ndarray) -> list[dict]:
        if not self.settings.vehicle_filter_enabled:
            h, w = frame.shape[:2]
            return [{"label": "vehicle", "conf": 1.0, "bbox": [0, 0, w, h], "area": w * h}]

        model = self._load_vehicle_model()
        results = model.predict(
            frame,
            conf=self.settings.vehicle_conf,
            verbose=False,
            classes=list(VEHICLE_CLASS_IDS),
        )
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area < self.settings.vehicle_min_box_area:
                continue
            detections.append({
                "label": results[0].names[cls_id],
                "conf": float(box.conf),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "area": area,
            })
        return detections

    @staticmethod
    def sharpness_score(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def frame_is_usable(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_b = gray.mean()
        if not (self.settings.min_brightness <= mean_b <= self.settings.max_brightness):
            return False
        return self.sharpness_score(frame) >= self.settings.min_sharpness_score

    def best_frame_from_buffer(self, buffer: deque) -> np.ndarray | None:
        scored = []
        for frame in buffer:
            if self.frame_is_usable(frame):
                scored.append((self.sharpness_score(frame), frame))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def crop_vehicle(self, frame: np.ndarray, bbox: list[int]) -> np.ndarray:
        pad = self.settings.vehicle_crop_padding
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        return frame[y1:y2, x1:x2]

    def select_vehicles(
        self,
        detections: list[dict],
        exclude_bbox: list[int] | None = None,
    ) -> list[dict]:
        if not detections:
            return []
        ranked = sorted(
            detections,
            key=lambda d: d["conf"] * d.get("area", 1),
            reverse=True,
        )
        if exclude_bbox is not None:
            ranked = [
                d for d in ranked
                if _bbox_iou(d["bbox"], exclude_bbox) < self.settings.dedup_iou
            ]
        return ranked[: self.settings.max_vehicles_per_trigger]

    def iter_vehicle_crops(self, source: str | int) -> Iterator[VehicleCropEvent]:
        """
        Scan video/camera source and yield one vehicle crop per trigger.
        """
        s = self.settings
        mode = (s.trigger_mode or "yolo").lower()
        if s.vehicle_filter_enabled:
            self._load_vehicle_model()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        fgbg = None
        if mode == "motion":
            fgbg = cv2.createBackgroundSubtractorMOG2(
                history=s.mog2_history,
                varThreshold=s.mog2_var_threshold,
                detectShadows=True,
            )

        buffer: deque = deque(maxlen=s.frame_buffer_size)
        frame_count = 0
        trigger_index = 0
        last_trigger_ts = 0.0
        last_sent_bbox = None

        logger.info(
            "VideoCapturer started source=%s mode=%s max_triggers=%s",
            source, mode, s.max_triggers,
        )

        try:
            while True:
                if s.max_triggers is not None and trigger_index >= s.max_triggers:
                    logger.info("Reached max_triggers=%s", s.max_triggers)
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                buffer.append(frame.copy())

                if frame_count % s.process_every_n_frames != 0:
                    continue

                now = time.time()
                if (now - last_trigger_ts) < s.cooldown_seconds:
                    continue

                should_scan = False
                if mode == "yolo":
                    should_scan = True
                elif fgbg is not None:
                    fgmask = fgbg.apply(frame)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    cleaned = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
                    if cv2.countNonZero(cleaned) > s.motion_threshold:
                        should_scan = True

                if not should_scan:
                    continue

                best = self.best_frame_from_buffer(buffer)
                if best is None:
                    continue

                vehicles = self.detect_vehicles(best)
                vehicles = self.select_vehicles(vehicles, exclude_bbox=last_sent_bbox)
                if not vehicles:
                    continue

                score = self.sharpness_score(best)
                for vehicle in vehicles:
                    crop = self.crop_vehicle(best, vehicle["bbox"])
                    if crop is None or crop.size == 0:
                        continue

                    trigger_index += 1
                    last_trigger_ts = now
                    last_sent_bbox = vehicle["bbox"]

                    yield VehicleCropEvent(
                        crop=crop,
                        bbox=vehicle["bbox"],
                        label=vehicle["label"],
                        conf=vehicle["conf"],
                        frame_index=frame_count,
                        trigger_index=trigger_index,
                        sharpness=score,
                    )
        finally:
            cap.release()
