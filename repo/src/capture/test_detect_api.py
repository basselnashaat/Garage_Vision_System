"""
POST a video (or image) to /api/detect.

Usage (FastAPI running):
    python test_detect_api.py
    python test_detect_api.py --video "d:\\path\\to\\file.mp4"
"""

import argparse
import glob
import os
import sys

import requests

import config


def main():
    parser = argparse.ArgumentParser(description="Test POST /api/detect with video or image")
    parser.add_argument("--video", default=None, help="Path to MP4/video file")
    parser.add_argument("--image", default=None, help="Path to a single crop image (legacy)")
    parser.add_argument("--camera-id", default=None)
    parser.add_argument(
        "--endpoint",
        default=getattr(config, "API_ENDPOINT", "http://localhost:8000/api/detect"),
    )
    parser.add_argument("--max-triggers", type=int, default=5, help="Limit vehicles for test")
    args = parser.parse_args()

    camera_id = args.camera_id or getattr(config, "CAMERA_ID", "")
    if not camera_id:
        print("CAMERA_ID empty. Run: python register_camera.py --write-config")
        sys.exit(1)

    path = args.video
    content_type = "video/mp4"
    if not path:
        if args.image:
            path = args.image
            content_type = "image/jpeg"
        else:
            path = getattr(config, "CAMERA_URL", None)
            if path and os.path.isfile(str(path)) and str(path).lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv")
            ):
                content_type = "video/mp4"
            else:
                candidates = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "plate_*.jpg")))
                if candidates:
                    path = candidates[-1]
                    content_type = "image/jpeg"
                else:
                    print("Provide --video or set CAMERA_URL to an mp4 in config.py")
                    sys.exit(1)

    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"POST {args.endpoint}")
    print(f"  file       : {path}")
    print(f"  type       : {content_type}")
    print(f"  camera_id  : {camera_id}")
    print(f"  max_triggers: {args.max_triggers}")

    with open(path, "rb") as f:
        resp = requests.post(
            args.endpoint,
            files={"file": (os.path.basename(path), f, content_type)},
            data={"camera_id": camera_id, "max_triggers": str(args.max_triggers)},
            timeout=getattr(config, "API_TIMEOUT", 600),
        )

    print(f"\nStatus: {resp.status_code}")
    print(resp.text[:5000])
    sys.exit(0 if resp.status_code == 200 else 1)


if __name__ == "__main__":
    main()
