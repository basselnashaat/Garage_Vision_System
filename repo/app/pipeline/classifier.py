"""
Vehicle Classifier

Loads the EfficientNetV2-S car generation classifier and returns vehicle
segment predictions in the same style as app/predict.py.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s

from ..config import PROJECT_ROOT

logger = logging.getLogger(__name__)

IMG_SIZE = 300
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_PRICE_CSV = PROJECT_ROOT / "datasets" / "car_recognition" / "hatla2ee_cars_unique.csv"


def _load_hf_asset(repo_id: str, filename: str, token: Optional[str]) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for HF model downloads. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, token=token))


class Classifier:
    """
    Classifies a vehicle image into one of 1240 classes using EfficientNetV2-S.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        class_names_path: Optional[str] = None,
        price_csv_path: Path = DEFAULT_PRICE_CSV,
        hf_repo: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Classifier initializing on device: {self.device}")

        if hf_repo:
            logger.info(f"Downloading classifier assets from HF Hub: {hf_repo}")
            model_path = str(_load_hf_asset(hf_repo, "best.pt", token))
            class_names_path = str(_load_hf_asset(hf_repo, "class_names.json", token))

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Classifier model weights not found at: {model_path}")
        if not class_names_path or not os.path.exists(class_names_path):
            raise FileNotFoundError(f"Class names JSON file not found at: {class_names_path}")

        with open(class_names_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

        num_classes = len(self.class_names)

        self.model = efficientnet_v2_s()
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Check if the state_dict is nested inside a metadata dictionary wrapper
        if isinstance(state_dict, dict) and "model_state" in state_dict:
            logger.info("Unpacking nested 'model_state' from model weights file wrapper.")
            state_dict = state_dict["model_state"]
            
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.price_lookup = {}
        if price_csv_path.exists():
            with open(price_csv_path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    c_name = row.get("class_name", "").strip()
                    try:
                        price_val = float(row.get("price", 0))
                    except ValueError:
                        price_val = 0.0
                    if c_name:
                        self.price_lookup[c_name] = price_val
            logger.info(f"Loaded price lookup for {len(self.price_lookup)} classes")
        else:
            logger.warning(f"Price lookup table CSV not found at: {price_csv_path}")

    @staticmethod
    def _parse_class_name(class_name: str) -> tuple[str, str]:
        parts = class_name.split("_")
        if len(parts) >= 4:
            brand = parts[0]
            model_name = "_".join(parts[1:-2])
        elif len(parts) == 3:
            brand, model_name = parts[0], parts[1]
        else:
            brand, model_name = class_name, "unknown"
        return brand, model_name

    def classify(self, frame: Any) -> dict[str, Any]:
        """
        Expects a numpy ndarray image in standard RGB format.
        Returns segment name, confidence score, car brand, and car model.
        """
        if frame is None:
            raise ValueError("Frame must not be None")
        if self.model is None:
            raise RuntimeError("Classifier model is not loaded")
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy ndarray")

        # Core Fix: Accept explicit RGB matrix frame arrays directly
        pil = Image.fromarray(frame)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = probs.topk(1, dim=1)

        idx = int(top_idx[0, 0].item())
        confidence = float(top_prob[0, 0].item())
        segment = self.class_names[idx]
        brand, model_name = self._parse_class_name(segment)
        estimated_price = self.price_lookup.get(segment, None)

        return {
            "segment": segment,
            "confidence": confidence,
            "car_brand": brand,
            "car_model": model_name,
            "estimated_price": estimated_price,
        }