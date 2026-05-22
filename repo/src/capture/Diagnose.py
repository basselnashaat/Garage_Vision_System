"""
EALPR — Camera Diagnostic
──────────────────────────
Reads 5 seconds of frames from your webcam and prints the
actual sharpness + brightness scores so you can set the
right thresholds in config.py.

Run with:
    python diagnose.py
    python diagnose.py --camera 0
"""

import cv2
import numpy as np
import argparse
import time

def diagnose(camera_source=0):
    print("═" * 52)
    print("  EALPR — Camera Diagnostic")
    print("  Reading 5 seconds of frames...")
    print("  Move around naturally in front of the camera.")
    print("═" * 52)

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"❌ Cannot open camera: {camera_source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nCamera: {w}×{h} @ {fps:.1f} FPS\n")

    sharpness_scores = []
    brightness_scores = []
    start = time.time()

    while time.time() - start < 5.0:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness  = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = gray.mean()

        sharpness_scores.append(sharpness)
        brightness_scores.append(brightness)

        # Live preview
        cv2.putText(frame, f"Sharpness : {sharpness:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Brightness: {brightness:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Diagnostic — press Q to stop early", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not sharpness_scores:
        print("No frames captured.")
        return

    s_min  = min(sharpness_scores)
    s_max  = max(sharpness_scores)
    s_mean = np.mean(sharpness_scores)
    s_med  = np.median(sharpness_scores)

    b_min  = min(brightness_scores)
    b_max  = max(brightness_scores)
    b_mean = np.mean(brightness_scores)

    print(f"── Sharpness Results ({'frames':>6}) ──────────────")
    print(f"  Min    : {s_min:.1f}")
    print(f"  Max    : {s_max:.1f}")
    print(f"  Mean   : {s_mean:.1f}")
    print(f"  Median : {s_med:.1f}")
    print()
    print(f"── Brightness Results ───────────────────────")
    print(f"  Min    : {b_min:.1f}")
    print(f"  Max    : {b_max:.1f}")
    print(f"  Mean   : {b_mean:.1f}")
    print()
    print(f"── Recommended config.py values ─────────────")

    # Recommend threshold at 70% of median sharpness
    recommended_sharpness = max(10.0, round(s_med * 0.7, 1))
    recommended_min_brightness = max(10, int(b_min * 0.8))
    recommended_max_brightness = min(245, int(b_max * 1.1))

    print(f"  MIN_SHARPNESS_SCORE = {recommended_sharpness}")
    print(f"  MIN_BRIGHTNESS      = {recommended_min_brightness}")
    print(f"  MAX_BRIGHTNESS      = {recommended_max_brightness}")
    print()
    print("  Copy these values into config.py and re-run capture.py")
    print("═" * 52)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=0, type=int)
    args = parser.parse_args()
    diagnose(args.camera)