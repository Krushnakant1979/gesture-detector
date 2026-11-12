"""
Real-World ASL Landmark & Dynamic Sequence Data Collector
Supports:
1. Static ASL gesture collection (A-Y) with 3D canonical normalization.
2. Dynamic ASL sequence collection (J, Z) with trajectory tracking, start-to-finish recording,
   live motion trail preview, and kinetic validation before saving.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import os
import sys
import time
import uuid
import cv2
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.capture import ThreadedCamera
from src.normalizer import LandmarkNormalizer
from src.tracker import HandTracker

STATIC_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
]
DYNAMIC_CLASSES = ["J", "Z"]


def load_dataset_stats(csv_path: str) -> Counter:
    """Returns sample count per letter in existing static CSV."""
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
            print(f"[WARN] Could not read dataset {csv_path}: {e}")
    return counts


def load_dynamic_stats(csv_path: str) -> Counter:
    """Returns sequence count per dynamic gesture in dynamic sequences CSV."""
    counts = Counter()
    if os.path.exists(csv_path):
        try:
            seen_seqs = set()
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = row.get("seq_id", "")
                    lbl = row.get("label", "").strip().upper()
                    if sid and sid not in seen_seqs:
                        seen_seqs.add(sid)
                        counts[lbl] += 1
        except Exception as e:
            print(f"[WARN] Could not read dynamic dataset {csv_path}: {e}")
    return counts


def init_static_csv(csv_path: str) -> None:
    """Initializes static CSV header if missing."""
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        canonical_cols = [f"c{i}" for i in range(63)]
        norm_cols = [f"n{i}" for i in range(63)]
        header = ["label", "handedness"] + canonical_cols + norm_cols
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"[INFO] Initialized new static dataset file: {csv_path}")


def init_dynamic_csv(csv_path: str) -> None:
    """Initializes dynamic sequence CSV header if missing."""
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        canonical_cols = [f"c{i}" for i in range(63)]
        header = ["seq_id", "frame_idx", "label", "handedness"] + canonical_cols + ["tip_x", "tip_y", "tip_z"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"[INFO] Initialized new dynamic sequence dataset file: {csv_path}")


def run_static_collection(
    letter: str,
    target_samples: int = 100,
    output_csv: str = "data/real_world_landmarks.csv",
    countdown_sec: int = 3,
    device_id: int = 0,
) -> None:
    """Runs live interactive webcam recording for a static ASL letter."""
    letter = letter.strip().upper()
    print(f"\n=======================================================")
    print(f" ASL Static Data Collector: Target Letter '{letter}'")
    print(f" Samples to collect: {target_samples} -> {output_csv}")
    print(f"=======================================================")

    init_static_csv(output_csv)
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

    window_name = f"ASL Static Collector - Recording '{letter}'"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    state = "READY"
    countdown_start = 0.0
    recorded_rows = []

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            frame_display = cv2.flip(frame, 1)
            h, w, _ = frame_display.shape

            hands = tracker.process(frame)
            hand_present = len(hands) > 0

            # UI banner
            cv2.rectangle(frame_display, (20, 20), (520, 160), (20, 20, 20), -1)
            cv2.putText(
                frame_display,
                f"STATIC SIGN: [ {letter} ]",
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
                bar_width = int(440 * (progress / target_samples))
                cv2.rectangle(frame_display, (35, 145), (35 + bar_width, 152), (0, 0, 255), -1)

                if hand_present:
                    primary_hand = hands[0]
                    lm_raw = primary_hand.landmarks_normalized
                    lm_canonical, _ = normalizer.canonical_align(lm_raw, handedness=primary_hand.handedness)
                    canonical_flat = lm_canonical.flatten().tolist()
                    norm_flat = normalizer.normalize(lm_raw).tolist()

                    row = [letter, primary_hand.handedness] + canonical_flat + norm_flat
                    recorded_rows.append(row)

                if len(recorded_rows) >= target_samples:
                    with open(output_csv, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerows(recorded_rows)
                    state = "DONE"
                    print(f"[SUCCESS] Saved {len(recorded_rows)} samples for '{letter}' into {output_csv}")
            elif state == "DONE":
                cv2.putText(
                    frame_display,
                    f"COMPLETED! Press [SPACE] to record more, [Q] to Exit",
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


def run_dynamic_collection(
    gesture: str,
    target_sequences: int = 15,
    seq_len: int = 20,
    output_csv: str = "data/dynamic_sequences.csv",
    countdown_sec: int = 3,
    device_id: int = 0,
) -> None:
    """Runs live interactive webcam recording for dynamic sequence gestures ('J' or 'Z')."""
    gesture = gesture.strip().upper()
    if gesture not in ("J", "Z"):
        print(f"[ERROR] Dynamic gesture must be 'J' or 'Z', got '{gesture}'")
        return

    print(f"\n=======================================================")
    print(f" ASL Dynamic Sequence Collector: Target Gesture '{gesture}'")
    print(f" Target sequences: {target_sequences} (each {seq_len} frames) -> {output_csv}")
    print(f"=======================================================")

    init_dynamic_csv(output_csv)
    stats = load_dynamic_stats(output_csv)
    print(f"[INFO] Current sequences for '{gesture}': {stats.get(gesture, 0)}")

    normalizer = LandmarkNormalizer()
    tracker = HandTracker(running_mode="video", enable_clahe=True)

    camera = ThreadedCamera(device_id=device_id, width=1280, height=720, fps=30)
    try:
        camera.start()
    except Exception as e:
        print(f"[ERROR] Failed to start camera: {e}")
        return

    window_name = f"ASL Dynamic Collector - Recording '{gesture}'"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    state = "READY"
    countdown_start = 0.0
    recorded_frames = []
    trail_points = []
    completed_seq_count = 0
    feedback_message = ""
    feedback_color = (0, 255, 0)
    feedback_time = 0.0

    tip_idx = 20 if gesture == "J" else 8  # Pinky tip for J, Index tip for Z

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            frame_display = cv2.flip(frame, 1)
            h, w, _ = frame_display.shape

            hands = tracker.process(frame)
            hand_present = len(hands) > 0

            # UI banner
            cv2.rectangle(frame_display, (20, 20), (550, 175), (20, 20, 20), -1)
            cv2.putText(
                frame_display,
                f"DYNAMIC GESTURE: [ {gesture} ]",
                (35, 55),
                cv2.FONT_HERSHEY_DUPLEX,
                0.95,
                (0, 215, 255),
                2,
            )

            status_color = (0, 255, 0) if hand_present else (0, 0, 255)
            status_text = f"Hand Tracked | Seq {completed_seq_count}/{target_sequences}" if hand_present else "NO HAND DETECTED"
            cv2.putText(frame_display, status_text, (35, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)

            now = time.time()

            if state == "READY":
                instruction = "Hold hand ready -> Press [SPACE] -> Trace stroke"
                cv2.putText(frame_display, instruction, (35, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                cv2.putText(frame_display, "Controls: [SPACE] Record | [Q] Quit", (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            elif state == "COUNTDOWN":
                elapsed = now - countdown_start
                remaining = max(1, countdown_sec - int(elapsed))
                cv2.putText(
                    frame_display,
                    f"Get Ready to Trace '{gesture}': {remaining}s...",
                    (35, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
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
                    recorded_frames.clear()
                    trail_points.clear()
                    print(f"\n[RECORDING] Tracing sequence {completed_seq_count + 1}...")

            elif state == "RECORDING":
                progress = len(recorded_frames)
                cv2.putText(
                    frame_display,
                    f"TRACING: Frame {progress}/{seq_len} - PERFORM MOTION!",
                    (35, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )
                bar_width = int(480 * (progress / seq_len))
                cv2.rectangle(frame_display, (35, 140), (35 + bar_width, 148), (0, 0, 255), -1)

                if hand_present:
                    primary_hand = hands[0]
                    lm_raw = primary_hand.landmarks_normalized  # (21, 3)
                    lm_canonical, _ = normalizer.canonical_align(lm_raw, handedness=primary_hand.handedness)

                    tip_pt = lm_raw[tip_idx]
                    recorded_frames.append({
                        "canonical": lm_canonical.flatten().tolist(),
                        "tip": [float(tip_pt[0]), float(tip_pt[1]), float(tip_pt[2])],
                        "handedness": primary_hand.handedness,
                    })
                    trail_points.append((int((1.0 - tip_pt[0]) * w), int(tip_pt[1] * h)))

                if len(recorded_frames) >= seq_len:
                    # Validate dynamic sequence: check kinetic motion
                    tips = np.array([f["tip"] for f in recorded_frames])
                    path_len = float(np.sum(np.linalg.norm(np.diff(tips[:, :2], axis=0), axis=1)))
                    disp = float(np.linalg.norm(tips[-1, :2] - tips[0, :2]))

                    min_path = 0.10 if gesture == "J" else 0.14
                    min_disp = 0.04

                    if path_len < min_path or disp < min_disp:
                        feedback_message = f"REJECTED: Motion too small ({path_len:.2f} < {min_path})! Try again."
                        feedback_color = (0, 0, 255)
                        print(f"[REJECTED] Sequence motion too small (path={path_len:.3f}, disp={disp:.3f})")
                    else:
                        # Valid sequence: save to CSV
                        seq_id = str(uuid.uuid4())[:8]
                        rows = []
                        for f_idx, frame_data in enumerate(recorded_frames):
                            row = (
                                [seq_id, f_idx, gesture, frame_data["handedness"]]
                                + frame_data["canonical"]
                                + frame_data["tip"]
                            )
                            rows.append(row)
                        with open(output_csv, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerows(rows)

                        completed_seq_count += 1
                        feedback_message = f"Sequence {completed_seq_count} Saved! (Path: {path_len:.2f})"
                        feedback_color = (0, 255, 0)
                        print(f"[SUCCESS] Saved sequence {seq_id} for '{gesture}' (path={path_len:.3f})")

                    feedback_time = time.time()
                    if completed_seq_count >= target_sequences:
                        state = "DONE"
                    else:
                        state = "READY"

            elif state == "DONE":
                cv2.putText(
                    frame_display,
                    f"COLLECTION COMPLETE! {completed_seq_count} Sequences Saved.",
                    (35, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(frame_display, "Press [Q] to Exit", (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            # Display validation feedback banner
            if feedback_message and now - feedback_time < 3.0:
                cv2.putText(
                    frame_display,
                    feedback_message,
                    (35, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    feedback_color,
                    2,
                )

            # Draw motion trail
            if len(trail_points) > 1:
                for i in range(1, len(trail_points)):
                    cv2.line(frame_display, trail_points[i - 1], trail_points[i], (0, 255, 255), 3)

            # Draw hand skeleton
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
        final_counts = load_dynamic_stats(output_csv)
        print(f"\n[SUMMARY] Current Total Dynamic Sequences in {output_csv}:")
        for k in sorted(final_counts.keys()):
            print(f"  {k}: {final_counts[k]} sequences")


def main() -> None:
    parser = argparse.ArgumentParser(description="ASL Landmark & Sequence Data Collector")
    parser.add_argument("--mode", type=str, choices=["static", "dynamic"], default="static", help="Collection mode")
    parser.add_argument("--letter", "--gesture", "--label", type=str, default="A", dest="letter", help="ASL letter/gesture (A-Z)")
    parser.add_argument("--samples", type=int, default=100, help="Samples to collect (frames for static, sequences for dynamic)")
    parser.add_argument("--seq-len", type=int, default=20, help="Frames per sequence for dynamic collection")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before recording")
    parser.add_argument("--device-id", type=int, default=0, help="Webcam device ID")
    args = parser.parse_args()

    letter = args.letter.strip().upper()

    if args.mode == "dynamic" or letter in ("J", "Z"):
        out_csv = args.output or "data/dynamic_sequences.csv"
        target_seqs = args.samples if args.samples <= 50 else 15
        run_dynamic_collection(
            gesture=letter,
            target_sequences=target_seqs,
            seq_len=args.seq_len,
            output_csv=out_csv,
            countdown_sec=args.countdown,
            device_id=args.device_id,
        )
    else:
        out_csv = args.output or "data/real_world_landmarks.csv"
        run_static_collection(
            letter=letter,
            target_samples=args.samples,
            output_csv=out_csv,
            countdown_sec=args.countdown,
            device_id=args.device_id,
        )


if __name__ == "__main__":
    main()
