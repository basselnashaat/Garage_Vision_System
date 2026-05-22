"""
Visit Logger Module

Logs detection results and scores to the database.
Records include: plate detection, character recognition, and scoring information.
"""

from datetime import datetime
import hashlib

from .connection import Database


class VisitLogger:
    """
    Logs visit records to the database.

    Records are written to multiple tables:
    - visits: Main visit record
    - plate_scores: License plate prestige score
    - segment_scores: Vehicle segment/class score
    - purchasing_power: Final purchasing power metric
    """

    def __init__(self, database: Database):
        self.db = database

    def log(self, plate_digits: str, plate_letters: str, plate_score: float,
            segment: str, segment_score: float, purchasing_power: float,
            car_brand: str = None, car_model: str = None,
            timestamp: datetime = None,
            camera_id: str = None) -> str:

        if not plate_digits:
            raise ValueError("plate_digits cannot be empty")
        if not camera_id:
            raise ValueError(
                "camera_id is required — pass a valid camera UUID from the cameras table."
            )
        if not (0.0 <= plate_score <= 1.0):
            raise ValueError(f"plate_score must be in [0.0, 1.0], got {plate_score}")
        if not (0.0 <= segment_score <= 1.0):
            raise ValueError(f"segment_score must be in [0.0, 1.0], got {segment_score}")
        if not (0.0 <= purchasing_power <= 1.0):
            raise ValueError(f"purchasing_power must be in [0.0, 1.0], got {purchasing_power}")

        timestamp = timestamp or datetime.now()

        try:
            visit_id = self._insert_visit(timestamp, plate_digits, plate_letters, camera_id)
            self._insert_plate_score(visit_id, plate_score)
            self._insert_segment_score(visit_id, segment, segment_score)
            self._insert_purchasing_power_score(visit_id, purchasing_power)
            return visit_id
        except Exception as e:
            raise RuntimeError(f"Failed to log visit: {e}")

    def _insert_visit(self, timestamp: datetime, plate_digits: str,
                      plate_letters: str, camera_id: str) -> str:

        plate_hash = hashlib.sha256(
            f"{plate_letters}{plate_digits}".encode("utf-8")
        ).hexdigest()

        data = {
            "camera_id":    camera_id,
            "plate_hash":   plate_hash,
            "plate_digits": plate_digits,
            "plate_letters": plate_letters,
            "visited_at":   timestamp.isoformat()
        }

        response = self.db.insert("visits", data)
        if not response or len(response) == 0:
            raise RuntimeError("Failed to insert visit: no data returned")

        return response[0]["id"]

    def _insert_plate_score(self, visit_id: str, score: float) -> None:
        data = {
            "visit_id":           visit_id,
            "digit_length_score": score,
            "repeat_score":       0,
            "sequential_score":   0,
            "low_number_score":   0,
            "total_score":        score
        }
        response = self.db.insert("plate_scores", data)
        if not response or len(response) == 0:
            raise RuntimeError("Failed to insert plate score")

    def _insert_segment_score(self, visit_id: str, segment: str,
                              score: float) -> None:
        data = {
            "visit_id":      visit_id,
            "segment":       segment,
            "confidence":    score,
            "segment_score": score
        }
        response = self.db.insert("segment_scores", data)
        if not response or len(response) == 0:
            raise RuntimeError("Failed to insert segment score")

    def _insert_purchasing_power_score(self, visit_id: str, score: float) -> None:
        data = {
            "visit_id":       visit_id,
            "plate_weight":   0,
            "segment_weight": 0,
            "final_score":    score
        }
        response = self.db.insert("purchasing_power", data)
        if not response or len(response) == 0:
            raise RuntimeError("Failed to insert purchasing power score")