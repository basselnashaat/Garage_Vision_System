import csv
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ARABIC_INDIC_TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_OCR_NORMALISE = str.maketrans({
    "ا": "أ",
    "إ": "أ",
    "آ": "أ",
    "ى": "ي",
    "ه": "ة",
})

SPECIAL_THRESHOLD = 6
ARABIC_NAMES_CSV = r"D:\Uni y3s2\ML\Project_Main_ML\vehicle-intelligence-dashboard\datasets\special_plates\arabic_names.csv"
PRESTIGE_WORDS = {
    "ملك", "أمر", "سيد", "شيخ",
    "أسد", "نمر", "صقر", "فهد", "ذئب", "نسر",
    "عز", "مجد", "ذهب", "نور", "هيب",
    "حب", "أمل", "روح", "حلم"
}


def normalize(text: str) -> str:
    return str(text).translate(_OCR_NORMALISE)

def extract_letters(plate: str) -> str:
    letters = re.sub(r"[0-9٠١٢٣٤٥٦٧٨٩\s]", "", str(plate))
    return normalize(letters).strip()

def extract_digits_from_list(nums: list[Any]) -> list[int]:
    digits: list[int] = []
    for value in nums or []:
        if value is None:
            continue
        if isinstance(value, int):
            digits.append(value)
            continue
        text = str(value).translate(_ARABIC_INDIC_TO_ASCII).strip()
        if not text:
            continue
        for ch in text:
            if ch.isdigit():
                digits.append(int(ch))
    return digits

def load_arabic_names(path: str = ARABIC_NAMES_CSV, name_col: str = "arabic_name") -> set[str]:
    location = Path(path)
    if not location.exists():
        logger.warning(f"arabic names csv not found: {location} — continuing with empty name set")
        return set()

    names: set[str] = set()
    with location.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = str(row.get(name_col, "")).strip()
            if 2 <= len(name) <= 3:
                names.add(name)
    return names

def get_level(score: float) -> str:
    if score <= 4:
        return "normal"
    if score <= 6:
        return "special"
    if score <= 9:
        return "very_special"
    return "VIP"

def is_sequential(digits: list[int]) -> bool:
    if len(digits) < 3:
        return False
    return all(digits[i] + 1 == digits[i + 1] for i in range(len(digits) - 1)) or \
           all(digits[i] - 1 == digits[i + 1] for i in range(len(digits) - 1))

def is_three_plus_one(digits: list[int]) -> bool:
    if len(digits) != 4:
        return False
    return sorted(Counter(digits).values(), reverse=True) == [3, 1]

def contains_year(digits: list[int]) -> bool:
    s = "".join(map(str, digits))
    for i in range(len(s) - 3):
        try:
            value = int(s[i:i + 4])
            if 1970 <= value <= 2026:
                return True
        except ValueError:
            continue
    return False

def _score_plate(full_plate: str, digits_list: list[int]) -> dict[str, Any]:
    letters = extract_letters(full_plate)
    digits = digits_list

    arabic_names = load_arabic_names()
    prestige_words_norm = {normalize(w) for w in PRESTIGE_WORDS}

    score = 0
    if letters in arabic_names:
        score += 3
    if letters in prestige_words_norm:
        score += 5
    if len(letters) == 1:
        score += 6
    if len(digits) == 1:
        score += 6
    if len(set(letters)) == 1 and letters:
        score += 4
    if len(set(digits)) == 1 and digits:
        score += 6
    if len(digits) == 4 and digits[:2] * 2 == digits:
        score += 4
    if len(digits) >= 3 and is_sequential(digits):
        score += 3
    if contains_year(digits):
        score += 2
    if is_three_plus_one(digits):
        score += 2

    return {
        "plate": full_plate,
        "score": round(score, 2),
        "is_special": int(score >= SPECIAL_THRESHOLD),
        "level": get_level(score),
    }


class SpecialPlateAnalyzer:
    """Analyzes a detected plate using the special plate scoring rules."""

    def __init__(self):
        self.arabic_names = None

    def analyze_plate(self, plate_string: str) -> dict[str, Any]:
        """
        Intercepts pipeline requests from coordinator.py cleanly.
        """
        digits = []
        if plate_string:
            clean_str = plate_string.replace(" ", "").translate(_ARABIC_INDIC_TO_ASCII)
            digits = list(re.sub(r"\D", "", clean_str))
        return self.analyze(plate_string, digits)

    def analyze(self, plate_string: str, digits: list[Any]) -> dict[str, Any]:
        """Main internal pipeline logic."""
        digits_list = extract_digits_from_list(digits)

        if not plate_string or not digits_list:
            return {
                "score": 0.0,
                "level": "normal",
                "is_special": False,
                "special_plate_score": 0.0,
                "special_plate_level": "normal",
                "is_special_plate": False,
            }

        try:
            result = _score_plate(plate_string, digits_list)
            scr = float(result.get("score", 0.0))
            lvl = result.get("level", "normal")
            is_spc = bool(result.get("is_special", 0))

            return {
                "score": scr,
                "level": lvl,
                "is_special": is_spc,
                "special_plate_score": scr,
                "special_plate_level": lvl,
                "is_special_plate": is_spc,
            }
        except Exception as exc:
            logger.warning(f"Special plate scoring failed: {exc}")
            return {
                "score": 0.0,
                "level": "normal",
                "is_special": False,
                "special_plate_score": 0.0,
                "special_plate_level": "normal",
                "is_special_plate": False,
            }