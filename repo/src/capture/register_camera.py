"""
Register a camera via POST /api/cameras and optionally save CAMERA_ID to config.py.

Usage (FastAPI must be running on port 8000):
    python register_camera.py
    python register_camera.py --name "Gate A" --type entrance --write-config
"""

import argparse
import re
import sys

import requests

import config

DEFAULT_API = "http://localhost:8000"


def create_camera(api_base: str, location_name: str, location_type: str) -> dict:
    url = f"{api_base.rstrip('/')}/api/cameras"
    resp = requests.post(
        url,
        json={"location_name": location_name, "location_type": location_type},
        timeout=15,
    )
    if resp.status_code != 201:
        print(f"Failed ({resp.status_code}): {resp.text}")
        sys.exit(1)
    return resp.json()


def write_camera_id_to_config(camera_id: str) -> None:
    config_path = config.__file__
    with open(config_path, encoding="utf-8") as f:
        text = f.read()

    if re.search(r'^CAMERA_ID\s*=', text, flags=re.MULTILINE):
        text = re.sub(
            r'^CAMERA_ID\s*=.*$',
            f'CAMERA_ID = "{camera_id}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text += f'\nCAMERA_ID = "{camera_id}"\n'

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser(description="Register a camera for capture → API integration")
    parser.add_argument("--api", default=DEFAULT_API, help="FastAPI base URL")
    parser.add_argument("--name", default="Video test camera", help="location_name")
    parser.add_argument("--type", default="entrance", help="location_type")
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write returned UUID into config.py as CAMERA_ID",
    )
    args = parser.parse_args()

    print(f"Creating camera at {args.api}/api/cameras ...")
    try:
        row = create_camera(args.api, args.name, args.type)
    except requests.exceptions.ConnectionError:
        print("Cannot reach API. Start the server first:")
        print('  cd vehicle-intelligence-dashboard')
        print("  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        sys.exit(1)

    camera_id = row["id"]
    print(f"\nCamera created:")
    print(f"  id             : {camera_id}")
    print(f"  location_name  : {row['location_name']}")
    print(f"  location_type  : {row['location_type']}")

    if args.write_config:
        write_camera_id_to_config(camera_id)
        print(f"\nWrote CAMERA_ID to {config.__file__}")
    else:
        print(f"\nAdd to config.py:  CAMERA_ID = \"{camera_id}\"")
        print(f"Or run capture with:  --camera-id {camera_id}")

    print("\nTest detect with a saved crop:")
    print(
        f'  curl -X POST "{args.api}/api/detect" '
        f'-F "file=@captured_frames/plate_....jpg" '
        f'-F "camera_id={camera_id}"'
    )


if __name__ == "__main__":
    main()
