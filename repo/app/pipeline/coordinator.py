"""
LPR Pipeline Coordinator
"""
import logging
from datetime import datetime
from typing import Any

import cv2 as cv
import numpy as np

from .detector        import PlateDetector
from .ocr             import OCR
from .classifier      import Classifier
from .scorer          import Scorer
from .special_plate   import SpecialPlateAnalyzer
from .video_capture   import VideoCapturer, CaptureSettings
from ..config    import (
    PLATE_DETECTOR_MODEL_PATH,
    CHAR_OCR_MODEL_PATH,
    CONF_THRESHOLD,
    HF_TOKEN,
    USE_HUGGING_FACE,
    HF_PLATE_DETECTOR_REPO,
    HF_CHAR_OCR_REPO,
    HF_CAR_CLASSIFIER_REPO,
)

logger = logging.getLogger(__name__)

class LPRPipeline:
    def __init__(self, use_hugging_face: bool = True):
        logger.info("Initializing LPR pipeline...")
        self.detector = None
        self.ocr = None
        self.classifier = None
        self.scorer = Scorer()
        self.special_plate = SpecialPlateAnalyzer()
        self.capturer = VideoCapturer()
        use_hf = USE_HUGGING_FACE and use_hugging_face

        try:
            self.detector = PlateDetector(
                hf_repo=HF_PLATE_DETECTOR_REPO if use_hf else None,
                model_path=None if use_hf else PLATE_DETECTOR_MODEL_PATH,
                conf_threshold=CONF_THRESHOLD,
            )
        except Exception as e:
            logger.exception(f"Failed to load PlateDetector: {e}")

        try:
            self.ocr = OCR(
                hf_repo=HF_CHAR_OCR_REPO if use_hf else None,
                model_path=None if use_hf else CHAR_OCR_MODEL_PATH,
                conf_threshold=CONF_THRESHOLD,
            )
        except Exception as e:
            logger.exception(f"Failed to load OCR model: {e}")

        try:
            self.classifier = Classifier(
                hf_repo=HF_CAR_CLASSIFIER_REPO,
                token=HF_TOKEN,  # Fixed kwarg name based on your classifier class
            )
        except Exception as e:
            logger.exception(f"Failed to load Classifier: {e}")
            self.classifier = None

        if not self.detector or not self.ocr:
            raise RuntimeError("Required pipeline components failed to load")

        logger.info("✓ Pipeline ready")

    def process(self, frame: np.ndarray) -> dict[str, Any]:
        if frame is None or frame.size == 0:
            return self._error("Invalid frame — None or empty.")

        try:
            bboxes = self.detector.detect(frame)

            if not bboxes:
                return {"success": True, "num_plates": 0, "detections": []}

            class_info    = self._classify_vehicle(frame)
            segment       = class_info["segment"]
            segment_score = self.scorer.score_segment(segment)
            visit_time    = datetime.now()

            detections = []

            for idx, bbox in enumerate(bboxes):
                try:
                    x1, y1, x2, y2 = map(int, bbox)
                    cropped = frame[y1:y2, x1:x2]

                    if cropped is None or cropped.size == 0:
                        continue

                    # OCR (Check if using read_with_details or get_text)
                    ocr_details = self.ocr.get_text(cropped) if hasattr(self.ocr, 'get_text') else self.ocr.read_with_details(cropped)
                    plate_string = ocr_details["plate_string"] or "UNKNOWN"

                    digits      = ocr_details.get("numbers", [])
                    digit_str   = "".join(digits)
                    plate_score = self.scorer.score_plate(digit_str) if digit_str else 0.0
                    
                    # Special plate analysis
                    special_info = self.special_plate.analyze(plate_string, digits)
                    
                    power_data  = self.scorer.calculate_purchasing_power(
                        plate_score=plate_score,
                        segment_score=segment_score,
                        visit_time=visit_time,
                        estimated_price=class_info.get("estimated_price", 0.0),
                        is_special_plate=special_info["is_special_plate"],
                        segment_name=segment 
                    )

                    # ── ADD THIS OVERRIDE ──
                    # Update the dictionary so the Gradio UI receives the fallback price
                    class_info["estimated_price"] = power_data["final_price"]

                    detections.append({
                        "special_plate_score": special_info.get("special_plate_score", 0.0),
                        "special_plate_level": special_info.get("special_plate_level", "normal"),
                        "is_special_plate":    special_info.get("is_special_plate", False),
                        "plate_string":     plate_string,
                        "digits":           digit_str,
                        "letters":          "".join(ocr_details.get("letters", [])),
                        "plate_score":      plate_score,
                        "segment":          segment,
                        "segment_score":    segment_score,
                        "car_brand":        class_info.get("car_brand", "unknown"),
                        "car_model":        class_info.get("car_model", "unknown"),
                        "estimated_price":  class_info.get("estimated_price", 0.0),
                        "purchasing_power": power_data["pp_score"],  # The numeric score
                        "pp_tier":          power_data["pp_tier"],   # The Elite/High tier
                        "bbox":             bbox,
                        "visit_time":       visit_time.isoformat(),
                        "visit_id":         None,
                        "num_detections":   ocr_details.get("num_detections", 1),
                    })

                except Exception as e:
                    logger.error(f"Error processing plate {idx}: {e}")
                    continue

            return {
                "success":    True,
                "num_plates": len(detections),
                "detections": detections,
            }

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            return self._error(str(e))

    def process_video(
        self,
        source: str | int,
        capture_settings: CaptureSettings | None = None,
    ) -> dict[str, Any]:
        
        # Native safe fallback since VideoCapturer method signature changed previously
        try:
            logger.info(f"Opening video source natively: {source}")
            cap = cv.VideoCapture(source)
            if not cap.isOpened():
                return self._error(f"Failed to open video file: {source}")

            all_detections = []
            frame_count = 0
            processed_count = 0
            step_frames = 12

            while True:
                ret, frame = cap.read()
                if not ret: break

                if frame_count % step_frames == 0:
                    frame_result = self.process(frame)
                    plates = frame_result.get("detections", [])

                    for p in plates:
                        p.update({
                            "visit_time":    datetime.now().isoformat(),
                            "vehicle_index": processed_count,
                            "frame_index":   frame_count,
                        })
                        all_detections.append(p)
                    processed_count += 1
                frame_count += 1
            cap.release()

            return {
                "success":                True,
                "num_vehicles_processed": processed_count,
                "num_plates":             len(all_detections),
                "detections":             all_detections,
                "vehicle_events":         [],
            }
        except Exception as e:
            logger.exception(f"process_video native error: {e}")
            return self._error(str(e))

    def _classify_vehicle(self, frame: np.ndarray) -> dict:
        """
        Run car classifier safely. Ensures estimated_price is 0.0 rather than missing.
        """
        fallback = {"segment": "unknown", "car_brand": "unknown", "car_model": "unknown", "estimated_price": 0.0}
        
        if self.classifier is None:
            return fallback

        try:
            # Color conversion applied for accurate model representation
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = self.classifier.classify(frame_rgb)
            # Ensure price key exists
            if "estimated_price" not in result or result["estimated_price"] is None:
                result["estimated_price"] = 0.0
            return result
        except Exception as e:
            logger.warning(f"Classifier failed, defaulting to unknown: {e}")
            return fallback

    @staticmethod
    def _error(message: str) -> dict:
        return {"success": False, "num_plates": 0, "detections": [], "error": message}