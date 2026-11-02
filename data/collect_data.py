"""
Dataset Collection Utility
Captures hand landmarks from the camera, normalizes them, and appends them
to a training CSV dataset for machine learning classification.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import cv2
import numpy as np

# Ensure root dir is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.capture import ThreadedCamera
from src.normalizer import LandmarkNormalizer
from src.tracker import HandTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Hand Gesture Landmark Dataset")
    parser.add_argument("--label", type=str, required=True, help="Name of the gesture (e.g. Peace, Fist, Custom)")
    parser.add_argument("--samples", type=int, default=150, help="Number of landmark samples to record")
    parser.add_argument("--output", type=str, default="data/gesture_dataset.csv", help="Path to CSV dataset")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Initialize CSV header if file doesn't exist
    if not os.path.exists(args.output):
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"{axis}_{i}" for i in range(21) for axis in ("x", "y", "z")]
            writer.writerow(header)

    normalizer = LandmarkNormalizer()
    tracker = HandTracker()
    camera = ThreadedCamera().start()

    print(f"\n[INFO] Collecting dataset for gesture: '{args.label}'")
    print("[INFO] Press 'SPACE' to start/pause recording, 'q' to quit.")

    recording = False
    collected_count = 0

    try:
        while collected_count < args.samples:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame)

            h, w, _ = frame.shape
            status_text = f"Recorded: {collected_count}/{args.samples} (Space to toggle)"
            status_color = (0, 255, 0) if recording else (0, 165, 255)

            if hands:
                hand = hands[0]
                # Draw skeleton
                for pt in hand.landmarks_pixel:
                    cv2.circle(frame, tuple(pt), 4, (0, 255, 0), -1)

                if recording:
                    # Normalize landmarks and append to CSV
                    features = normalizer.normalize(hand.landmarks_normalized)
                    with open(args.output, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([args.label] + features.tolist())
                    collected_count += 1
                    time.sleep(0.04)  # ~25 samples per sec

            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Gesture: {args.label}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Dataset Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                recording = not recording
                print(f"[STATUS] Recording {'STARTED' if recording else 'PAUSED'}")
            elif key in (ord("q"), 27):
                break

    finally:
        camera.stop()
        tracker.close()
        cv2.destroyAllWindows()
        print(f"\n[FINISHED] Recorded {collected_count} samples for '{args.label}' into {args.output}")


if __name__ == "__main__":
    main()

