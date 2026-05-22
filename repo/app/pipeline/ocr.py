import logging
import os
import re
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)

_ARABIC_INDIC_TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def _load_hf_asset(repo_id: str, filename: str, token: Optional[str]) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for HF model downloads. "
            "Install it with `pip install huggingface_hub`."
        ) from exc
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, token=token))


class OCR:
    """
    Handles Egyptian license plate character recognition (OCR) using YOLO.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hf_repo: Optional[str] = None,
        conf_threshold: float = 0.25,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        
        logger.info(f"OCR Engine initializing on device: {self.device}")

        # 1. Handle Hugging Face download if configured
        if hf_repo:
            logger.info(f"Downloading OCR model weights from HF Hub: {hf_repo}")
            try:
                model_path = str(_load_hf_asset(hf_repo, "best.pt", None))
            except Exception as e:
                logger.warning(f"Could not download from HF Hub, falling back to local weights: {e}")

        # 2. Fall back to local path if needed
        if not model_path or not os.path.exists(model_path):
            from ..config import CHAR_OCR_MODEL_PATH
            model_path = CHAR_OCR_MODEL_PATH

        logger.info(f"Loading OCR YOLO weights from: {model_path}")
        
        # Load the underlying ultralytics YOLO model safely
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def get_text(self, plate_crop: np.ndarray) -> dict[str, Any]:
        """
        Interface method required by coordinator.py line 84.
        Processes a cropped license plate image and extracts characters.
        """
        if plate_crop is None or plate_crop.size == 0:
            return {"plate_string": "", "letters": [], "numbers": []}

        letters_list = []
        numbers_list = []
        
        # Run inference using the underlying YOLO engine
        results = self.model.predict(plate_crop, conf=self.conf_threshold, verbose=False)
        
        detected_chars = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = self.model.names[cls_id]
                # Track the top-left x-coordinate to properly sequence text flow
                x_min = float(box.xyxy[0][0].item())
                detected_chars.append((x_min, label))
        
        # Arabic license plates flow from right-to-left. 
        # Sorting reverse=True ensures text builds seamlessly in the correct reading order.
        detected_chars.sort(key=lambda x: x[0], reverse=True)
        
        # Segment arrays into dedicated letter and numerical streams
        for _, char in detected_chars:
            char_clean = char.strip()
            # Check if digit (translating Arabic Indic characters to standard ascii digits safely)
            if re.match(r"^\d+$", char_clean.translate(_ARABIC_INDIC_TO_ASCII)):
                numbers_list.append(char_clean)
            else:
                letters_list.append(char_clean)
        
        # Combine character slices into complete output string arrays
        combined_letters = " ".join(letters_list)
        combined_numbers = " ".join(numbers_list)
        
        if combined_letters and combined_numbers:
            plate_string = f"{combined_letters} {combined_numbers}"
        else:
            plate_string = combined_letters if combined_letters else combined_numbers

        # Return the exact keys your coordinator.py needs to process scoring
        return {
            "plate_string": plate_string.strip(),
            "letters": letters_list,
            "numbers": numbers_list,
        }