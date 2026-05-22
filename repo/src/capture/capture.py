"""
CLI for video capture — uploads video to POST /api/detect (pipeline runs server-side).

For local processing without HTTP:
    python capture.py --local

Legacy standalone loop with preview (uses app pipeline in-process):
    python capture.py --local --preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root on path so `app` package resolves
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as capture_config  # noqa: E402


def post_video_to_api(video_path: str, camera_id: str, api_url: str, max_triggers: int | None):
    import requests

    with open(video_path, "rb") as f:
        data = {"camera_id": camera_id}
        if max_triggers is not None:
            data["max_triggers"] = str(max_triggers)
        resp = requests.post(
            api_url,
            files={"file": (Path(video_path).name, f, "video/mp4")},
            data=data,
            timeout=getattr(capture_config, "API_TIMEOUT", 600),
        )
    print(f"Status: {resp.status_code}")
    print(resp.text[:4000])
    return resp.status_code == 200


def run_local(video_source, trigger_mode: str | None, max_triggers: int | None, preview: bool):
    from app.pipeline.coordinator import LPRPipeline
    from app.pipeline.video_capture import CaptureSettings

    settings = CaptureSettings(
        trigger_mode=trigger_mode or getattr(capture_config, "TRIGGER_MODE", "yolo"),
        process_every_n_frames=getattr(capture_config, "PROCESS_EVERY_N_FRAMES", 3),
        frame_buffer_size=getattr(capture_config, "FRAME_BUFFER_SIZE", 20),
        motion_threshold=getattr(capture_config, "MOTION_THRESHOLD", 1500),
        mog2_history=getattr(capture_config, "MOG2_HISTORY", 500),
        mog2_var_threshold=getattr(capture_config, "MOG2_VAR_THRESHOLD", 50),
        cooldown_seconds=getattr(capture_config, "COOLDOWN_SECONDS", 2.0),
        vehicle_model=getattr(capture_config, "VEHICLE_MODEL", "yolo11n.pt"),
        vehicle_conf=getattr(capture_config, "VEHICLE_CONF", 0.40),
        vehicle_min_box_area=getattr(capture_config, "VEHICLE_MIN_BOX_AREA", 3000),
        vehicle_crop_padding=getattr(capture_config, "VEHICLE_CROP_PADDING", 50),
        vehicle_filter_enabled=getattr(capture_config, "VEHICLE_FILTER_ENABLED", True),
        max_vehicles_per_trigger=getattr(capture_config, "MAX_VEHICLES_PER_TRIGGER", 1),
        min_sharpness_score=getattr(capture_config, "MIN_SHARPNESS_SCORE", 5.0),
        min_brightness=getattr(capture_config, "MIN_BRIGHTNESS", 30),
        max_brightness=getattr(capture_config, "MAX_BRIGHTNESS", 220),
        max_triggers=max_triggers,
    )

    print("Loading LPR pipeline (models may take a minute)...")
    pipeline = LPRPipeline()
    source = video_source if video_source is not None else capture_config.CAMERA_URL
    print(f"Processing video: {source}")
    result = pipeline.process_video(source, capture_settings=settings)

    print(f"success: {result.get('success')}")
    print(f"vehicles processed: {result.get('num_vehicles_processed')}")
    print(f"plates found: {result.get('num_plates')}")
    for ev in result.get("vehicle_events", []):
        print(
            f"  trigger #{ev['trigger_index']} frame {ev['frame_index']} "
            f"{ev['vehicle_label']} {ev['vehicle_conf']:.0%} → {ev['num_plates']} plate(s)"
        )

    if preview:
        print("Preview not available in --local mode; use API upload or extend VideoCapturer.")


def main():
    parser = argparse.ArgumentParser(description="EALPR video capture client")
    parser.add_argument("--camera", default=None, help="Video path, RTSP URL, or 0 for webcam")
    parser.add_argument("--camera-id", default=None, help="Camera UUID for /api/detect")
    parser.add_argument("--local", action="store_true", help="Run pipeline in-process (no HTTP)")
    parser.add_argument("--mode", choices=["yolo", "motion"], default=None)
    parser.add_argument("--max-triggers", type=int, default=None, help="Max vehicles to process")
    parser.add_argument(
        "--endpoint",
        default=getattr(capture_config, "API_ENDPOINT", "http://localhost:8000/api/detect"),
    )
    args = parser.parse_args()

    source = args.camera
    if source is not None and str(source).isdigit():
        source = int(source)

    camera_id = args.camera_id or getattr(capture_config, "CAMERA_ID", "")
    video_path = source if source is not None else capture_config.CAMERA_URL

    if args.local:
        run_local(video_path, args.mode, args.max_triggers, preview=False)
        return

    if not camera_id:
        print("Set CAMERA_ID in config.py or pass --camera-id")
        sys.exit(1)

    if not post_video_to_api(str(video_path), camera_id, args.endpoint, args.max_triggers):
        sys.exit(1)


if __name__ == "__main__":
    main()
