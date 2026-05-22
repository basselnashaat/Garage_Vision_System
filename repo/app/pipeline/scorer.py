import pandas as pd
import numpy as np
import logging
from pathlib import Path
from ..config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Price is the dominant signal
TIER_THRESHOLDS = {
    "Elite":    0.90,   # top 10% by price
    "High":     0.70,   # next 20%
    "Mid":      0.40,   # next 30%
    "Standard": 0.0,    # bottom 40%
}

TIER_ORDER = ["Standard", "Mid", "High", "Elite"]
SPECIAL_PLATE_BUMP = 1

class Scorer:
    def __init__(self):
        self.saved_prices = np.array([])
        
        csv_path = PROJECT_ROOT / "datasets" / "purchasing_power_results.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "estimated_price" in df.columns:
                    self.saved_prices = df["estimated_price"].dropna().values
                    logger.info(f"Loaded {len(self.saved_prices)} historical car prices for tier ranking.")
            except Exception as e:
                logger.warning(f"Could not load prices for Scorer: {e}")
        else:
            logger.warning(f"Purchasing power CSV not found at {csv_path}. Using fallback distribution.")
        
        if len(self.saved_prices) == 0:
            self.saved_prices = np.array([500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000])

    def score_plate(self, digits: str) -> float:
        return 0.0

    def score_segment(self, segment: str) -> float:
        return 0.0

    def calculate_purchasing_power(
        self,
        plate_score: float,
        segment_score: float,
        visit_time,
        estimated_price: float,
        is_special_plate: bool,
        segment_name: str = "", # ADDED: Pass the segment string here
    ) -> dict:
        
        # SMART FALLBACK: If price is 0, check the car's name before assigning the median!
        if estimated_price is None or pd.isna(estimated_price) or estimated_price <= 0:
            name_lower = str(segment_name).lower()
            
            # If it's an obvious luxury car, give it a 15 Million EGP default
            if any(vip in name_lower for vip in ["g63", "mercedes", "porsche", "rover", "bmw_x", "lexus"]):
                estimated_price = 15_000_000.0
            # Otherwise, use the standard median average
            else:
                estimated_price = float(np.median(self.saved_prices))

        price_pct = float(np.mean(self.saved_prices <= estimated_price))
        
        if price_pct >= TIER_THRESHOLDS["Elite"]: price_tier = "Elite"
        elif price_pct >= TIER_THRESHOLDS["High"]: price_tier = "High"
        elif price_pct >= TIER_THRESHOLDS["Mid"]: price_tier = "Mid"
        else: price_tier = "Standard"

        if is_special_plate:
            idx = TIER_ORDER.index(price_tier)
            final_tier = TIER_ORDER[min(idx + SPECIAL_PLATE_BUMP, len(TIER_ORDER) - 1)]
        else:
            final_tier = price_tier

        return {
            "pp_score": round(price_pct, 4),
            "pp_tier": final_tier,
            "final_price": estimated_price
        }