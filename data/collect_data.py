"""
Real-World ASL Landmark Data Collector
Captures live webcam hand landmarks with 3D canonical normalization,
allowing users to quickly record real training data for any ASL letter (A-Z).
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import os
import sys
import time
import cv2
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.capture import ThreadedCamera
from src.normalizer import LandmarkNormalizer
from src.tracker import HandTracker


def load_dataset_stats(csv_path: str) -> Counter:
    """Returns sample count per letter in existing CSV."""
    counts = Counter()
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row.get("label", "").strip().upper()
                    if label:
                        counts[label] += 1
        except Exception as e:
            print(f"[WARN] Could not read existing dataset: {e}")
    return counts


def create_csv_if_missing(csv_path: str) -> None:
    """Initializes CSV header if file does not exist."""
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # Header: label, handedness, 63 canonical landmarks (c0..c62), 63 raw normalized landmarks (n0..n62)
        canonical_cols = [f"c{i}" for i in range(63)]
        norm_cols = [f"n{i}" for i in range(63)]
        header = ["label", "handedness"] + canonical_cols + norm_cols
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"[INFO] Initialized new dataset file: {csv_path}")


def append_samples(csv_path: str, rows: list) -> None:
    """Appends recorded rows to CSV."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def run_collection(
    letter: str,
    target_samples: int = 100,
    output_csv: str = "data/real_world_landmarks.csv",
    countdown_sec: int = 3,
    device_id: int = 0,
) -> None:
    """Runs live interactive webcam recording for the specified letter."""
    letter = letter.strip().upper()
    print(f"\n=======================================================")
    print(f" Real-World ASL Data Collector: Target Letter '{letter}'")
    print(f" Samples to collect: {target_samples} -> {output_csv}")
    print(f"=======================================================")

    create_csv_if_missing(output_csv)
    stats = load_dataset_stats(output_csv)
    print(f"[INFO] Current samples for '{letter}': {stats.get(letter, 0)}")

    normalizer = LandmarkNormalizer()
    tracker = HandTracker(running_mode="video", enable_clahe=True)

    camera = ThreadedCamera(device_id=device_id, width=1280, height=720, fps=30)
    try:
        camera.start()
    except Exception as e:
        print(f"[ERROR] Failed to start camera: {e}")
        return

    window_name = f"ASL Data Collector - Recording '{letter}'"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[INSTRUCTIONS]")
    print(f"1. Form the sign for '{letter}' in front of the camera.")
    print("2. Press [SPACE] to start the 3-second countdown and record.")
    print("3. Press [Q] or [ESC] to cancel/exit.")

    state = "READY"  # "READY", "COUNTDOWN", "RECORDING", "DONE"
    countdown_start = 0.0
    recorded_rows = []

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            frame_display = cv2.flip(frame, 1)  # mirror for intuitive UI
            h, w, _ = frame_display.shape

            hands = tracker.process(frame)
            hand_present = len(hands) > 0

            # UI overlays
            cv2.rectangle(frame_display, (20, 20), (500, 160), (20, 20, 20), -1)
            cv2.putText(
                frame_display,
                f"TARGET SIGN: [ {letter} ]",
                (35, 60),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (0, 255, 120),
                2,
            )

            status_color = (0, 255, 0) if hand_present else (0, 0, 255)
            status_text = "Hand Tracked (OK)" if hand_present else "NO HAND DETECTED"
            cv2.putText(frame_display, status_text, (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

            now = time.time()

            if state == "READY":
                cv2.putText(
                    frame_display,
                    "Press [SPACE] to Record | [Q] Quit",
                    (35, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 220, 220),
                    1,
                )

            elif state == "COUNTDOWN":
                elapsed = now - countdown_start
                remaining = max(1, countdown_sec - int(elapsed))
                cv2.putText(
                    frame_display,
                    f"Get Ready: {remaining}s...",
                    (35, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 215, 255),
                    2,
                )
                cv2.putText(
                    frame_display,
                    str(remaining),
                    (w // 2 - 40, h // 2 + 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    3.5,
                    (0, 215, 255),
                    6,
                )
                if elapsed >= countdown_sec:
                    state = "RECORDING"
                    recorded_rows.clear()
                    print(f"[INFO] Recording {target_samples} frames for '{letter}'...")

            elif state == "RECORDING":
                progress = len(recorded_rows)
                cv2.putText(
                    frame_display,
                    f"RECORDING: {progress}/{target_samples}",
                    (35, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # Progress bar
                bar_width = int(440 * (progress / target_samples))
                cv2.rectangle(frame_display, (35, 145), (35 + bar_width, 152), (0, 0, 255), -1)

                if hand_present:
                    primary_hand = hands[0]
                    lm_raw = primary_hand.landmarks_normalized  # (21, 3)

                    # 3D Canonical Alignment
                    lm_canonical, _ = normalizer.canonical_align(lm_raw, handedness=primary_hand.handedness)
                    canonical_flat = lm_canonical.flatten().tolist()
                    norm_flat = normalizer.normalize(lm_raw).tolist()

                    row = [letter, primary_hand.handedness] + canonical_flat + norm_flat
                    recorded_rows.append(row)

                if len(recorded_rows) >= target_samples:
                    append_samples(output_csv, recorded_rows)
                    state = "DONE"
                    print(f"[SUCCESS] Saved {len(recorded_rows)} samples for '{letter}' into {output_csv}")

            elif state == "DONE":
                cv2.putText(
                    frame_display,
                    f"COMPLETED! Press [SPACE] for more, [Q] to Exit",
                    (35, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

            # Draw skeleton
            if hand_present:
                for pt in hands[0].landmarks_pixel:
                    px = w - pt[0]
                    py = pt[1]
                    cv2.circle(frame_display, (px, py), 4, (0, 255, 0), -1)

            cv2.imshow(window_name, frame_display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                if state in ("READY", "DONE"):
                    state = "COUNTDOWN"
                    countdown_start = time.time()

    finally:
        camera.stop()
        tracker.close()
        cv2.destroyAllWindows()
        final_counts = load_dataset_stats(output_csv)
        print(f"\n[SUMMARY] Current Total Dataset Samples in {output_csv}:")
        for k in sorted(final_counts.keys()):
            print(f"  {k}: {final_counts[k]} samples")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-World ASL Landmark Data Collector")
    parser.add_argument("--letter", "--label", type=str, default="A", dest="letter", help="ASL letter to collect (A-Z)")
    parser.add_argument("--samples", type=int, default=100, help="Number of frames to record")
    parser.add_argument("--output", type=str, default="data/real_world_landmarks.csv", help="Output CSV path")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before recording")
    parser.add_argument("--device-id", type=int, default=0, help="Webcam device ID")
    args = parser.parse_args()

    run_collection(
        letter=args.letter,
        target_samples=args.samples,
        output_csv=args.output,
        countdown_sec=args.countdown,
        device_id=args.device_id,
    )


if __name__ == "__main__":
    main()
