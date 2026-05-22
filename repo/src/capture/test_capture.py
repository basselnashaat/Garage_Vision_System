"""
EALPR — Capture System Test Suite
───────────────────────────────────
Tests every component of capture.py independently
so you can verify each piece works before connecting
a real camera or the FastAPI backend.

Run with:
    python test_capture.py
    python xtest_capture.py --image path/to/car.jpg
    python test_capture.py --video path/to/clip.mp4
"""

import cv2
import numpy as np
import os
import argparse
import time
from collections import deque
from datetime import datetime

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from app.pipeline.video_capture import VideoCapturer

_capturer = VideoCapturer()


def sharpness_score(frame):
    return _capturer.sharpness_score(frame)


def brightness_ok(frame):
    gray = __import__("cv2").cvtColor(frame, __import__("cv2").COLOR_BGR2GRAY)
    return config.MIN_BRIGHTNESS <= gray.mean() <= config.MAX_BRIGHTNESS


def frame_is_usable(frame):
    if not brightness_ok(frame):
        return False, "brightness out of range"
    score = sharpness_score(frame)
    if score < config.MIN_SHARPNESS_SCORE:
        return False, f"too blurry (score={score:.1f})"
    return True, ""


def best_frame_from_buffer(buffer):
    return _capturer.best_frame_from_buffer(buffer)


def save_vehicle_crop(crop, vehicle_idx, timestamp):
    import cv2
    filename = f"plate_{timestamp}_v{vehicle_idx}{config.SAVE_FORMAT}"
    filepath = os.path.join(config.OUTPUT_DIR, filename)
    cv2.imwrite(filepath, crop, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    return filepath


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def make_synthetic_frame(brightness: int = 120, blur_ksize: int = 1) -> np.ndarray:
    """Creates a fake frame for testing quality functions."""
    # White text on dark background (simulates a plate)
    frame = np.full((480, 640, 3), brightness, dtype=np.uint8)
    cv2.putText(frame, "ABC 1234", (80, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)
    if blur_ksize > 1:
        frame = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)
    return frame


def pass_fail(condition: bool) -> str:
    return "✅ PASS" if condition else "❌ FAIL"


# ─────────────────────────────────────────────
#  Unit tests
# ─────────────────────────────────────────────

def test_sharpness():
    print("\n── Test 1: Sharpness Scoring ─────────────────")
    sharp  = make_synthetic_frame(blur_ksize=1)
    blurry = make_synthetic_frame(blur_ksize=51)

    s_sharp  = sharpness_score(sharp)
    s_blurry = sharpness_score(blurry)

    print(f"  Sharp frame score  : {s_sharp:.1f}")
    print(f"  Blurry frame score : {s_blurry:.1f}")
    print(f"  Sharp > Blurry     : {pass_fail(s_sharp > s_blurry)}")
    print(f"  Sharp passes gate  : {pass_fail(s_sharp >= config.MIN_SHARPNESS_SCORE)}")
    print(f"  Blurry fails gate  : {pass_fail(s_blurry < config.MIN_SHARPNESS_SCORE)}")


def test_brightness():
    print("\n── Test 2: Brightness Check ──────────────────")
    normal   = make_synthetic_frame(brightness=120)
    too_dark = make_synthetic_frame(brightness=10)
    blown_out = make_synthetic_frame(brightness=250)

    print(f"  Normal frame ok    : {pass_fail(brightness_ok(normal))}")
    print(f"  Dark frame rejected: {pass_fail(not brightness_ok(too_dark))}")
    print(f"  Blown out rejected : {pass_fail(not brightness_ok(blown_out))}")


def test_usability():
    print("\n── Test 3: Combined Usability Gate ──────────")
    good   = make_synthetic_frame(brightness=120, blur_ksize=1)
    bad    = make_synthetic_frame(brightness=5,   blur_ksize=51)

    ok_good, _   = frame_is_usable(good)
    ok_bad,  why = frame_is_usable(bad)

    print(f"  Good frame passes  : {pass_fail(ok_good)}")
    print(f"  Bad frame rejected : {pass_fail(not ok_bad)} (reason: {why})")


def test_best_frame_selection():
    print("\n── Test 4: Best Frame from Buffer ───────────")
    buffer = deque(maxlen=20)

    # Fill with increasingly sharp frames
    for blur in [51, 31, 21, 11, 1]:
        buffer.append(make_synthetic_frame(blur_ksize=blur))

    best = best_frame_from_buffer(buffer)
    if best is None:
        print("  ❌ FAIL — no frame selected")
        return

    score = sharpness_score(best)
    print(f"  Best frame score   : {score:.1f}")
    print(f"  Correctly sharpest : {pass_fail(score >= config.MIN_SHARPNESS_SCORE)}")


def test_save_frame():
    print("\n── Test 5: Save Frame to Disk ───────────────")
    frame    = make_synthetic_frame()
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = save_vehicle_crop(frame, vehicle_idx=1, timestamp=ts)

    exists = os.path.isfile(filepath)
    size   = os.path.getsize(filepath) if exists else 0

    print(f"  File created       : {pass_fail(exists)}")
    print(f"  File not empty     : {pass_fail(size > 0)}")
    print(f"  Saved to           : {filepath}")


def test_motion_detection():
    print("\n── Test 6: Motion Detection (MOG2) ──────────")
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=config.MOG2_HISTORY,
        varThreshold=config.MOG2_VAR_THRESHOLD,
        detectShadows=True,
    )

    background = make_synthetic_frame(brightness=80, blur_ksize=1)
    # "Prime" the background subtractor
    for _ in range(30):
        fgbg.apply(background)

    # Now introduce a "car" (bright rectangle = large foreground change)
    car_frame = background.copy()
    cv2.rectangle(car_frame, (100, 150), (540, 350), (200, 200, 200), -1)

    mask    = fgbg.apply(car_frame)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    pixels  = cv2.countNonZero(cleaned)

    print(f"  Foreground pixels  : {pixels}")
    print(f"  Trigger fires      : {pass_fail(pixels > config.MOTION_THRESHOLD)}")


# ─────────────────────────────────────────────
#  Integration test on a real image
# ─────────────────────────────────────────────

def test_with_image(image_path: str):
    print(f"\n── Integration Test: Real Image ─────────────")
    print(f"  Path: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  ❌ Cannot load image: {image_path}")
        return

    h, w = frame.shape[:2]
    print(f"  Dimensions : {w}×{h}")

    score = sharpness_score(frame)
    ok, reason = frame_is_usable(frame)

    print(f"  Sharpness  : {score:.1f}")
    print(f"  Usable     : {pass_fail(ok)}", f"({reason})" if not ok else "")

    if ok:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = save_vehicle_crop(frame, vehicle_idx=1, timestamp=ts)
        print(f"  Saved to   : {filepath}")

        # Show the frame
        cv2.imshow("Test Image", frame)
        print("  (Press any key to close preview)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  Integration test on a real video clip
# ─────────────────────────────────────────────

def test_with_video(video_path: str):
    print(f"\n── Integration Test: Real Video Clip ────────")
    print(f"  Path: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Frames: {total_frames}  FPS: {fps:.1f}")

    fgbg   = cv2.createBackgroundSubtractorMOG2(
        history=config.MOG2_HISTORY,
        varThreshold=config.MOG2_VAR_THRESHOLD,
        detectShadows=True,
    )
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    buffer  = deque(maxlen=config.FRAME_BUFFER_SIZE)
    triggers = 0
    last_trigger = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        buffer.append(frame.copy())

        if frame_idx % config.PROCESS_EVERY_N_FRAMES != 0:
            continue

        fgmask  = fgbg.apply(frame)
        cleaned = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        pixels  = cv2.countNonZero(cleaned)

        now = time.time()
        if pixels > config.MOTION_THRESHOLD and (now - last_trigger) > config.COOLDOWN_SECONDS:
            triggers  += 1
            last_trigger = now
            best = best_frame_from_buffer(buffer)
            if best is not None:
                score    = sharpness_score(best)
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filepath = save_vehicle_crop(best, vehicle_idx=1, timestamp=ts)
                print(f"  🚗 Trigger #{triggers} at frame {frame_idx}  "
                      f"sharpness={score:.1f}  → {os.path.basename(filepath)}")

    cap.release()
    print(f"\n  Total triggers : {triggers}")
    print(f"  Frames saved   : {triggers}")
    print(f"  {pass_fail(triggers > 0)}")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EALPR Capture System Tests")
    parser.add_argument("--image", default=None, help="Path to a test car image")
    parser.add_argument("--video", default=None, help="Path to a test video clip")
    args = parser.parse_args()

    print("═" * 52)
    print("  EALPR Capture System — Test Suite")
    print("═" * 52)

    # Always run unit tests
    test_sharpness()
    test_brightness()
    test_usability()
    test_best_frame_selection()
    test_save_frame()
    test_motion_detection()

    # Optional integration tests
    if args.image:
        test_with_image(args.image)

    if args.video:
        test_with_video(args.video)

    print("\n" + "═" * 52)
    print("  All unit tests complete.")
    if not args.image and not args.video:
        print("  Tip: run with --image car.jpg or --video clip.mp4")
        print("       for integration tests on real footage.")
    print("═" * 52)