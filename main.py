"""
Main Application Entry Point for Real-Time ASL Fingerspelling Recognition.
Orchestrates high-fps camera capture with auto-reconnect, MediaPipe video tracking,
landmark smoothing, dynamic stroke recognition (J/Z), hybrid classification,
debouncing, HUD rendering, auto-complete, and non-blocking Text-To-Speech.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List
import cv2
import yaml

from src.capture import ThreadedCamera
from src.classifier import GestureClassifier
from src.filters import (
    ExponentialMovingAverage,
    GestureDebouncer,
    StrokeTrajectoryTracker,
    TextToSpeech,
    WordPredictor,
)
from src.normalizer import LandmarkNormalizer
from src.tracker import HandTracker
from src.visualizer import ModernVisualizer


def setup_logger() -> logging.Logger:
    """Configures formatted standard logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("ASLApp")


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration with sane fallbacks."""
    default_config: Dict[str, Any] = {
        "camera": {"device_id": 0, "width": 1280, "height": 720, "fps": 30, "reconnect_timeout_sec": 2.0},
        "tracker": {
            "model_path": "models/hand_landmarker.task",
            "running_mode": "video",
            "max_num_hands": 1,
            "min_detection_confidence": 0.65,
            "min_presence_confidence": 0.55,
            "min_tracking_confidence": 0.55,
            "box_padding": 20,
        },
        "filter": {
            "smoothing_alpha": 0.65,
            "enable_smoothing": true,
            "enable_trajectory": true,
        },
        "debouncer": {"consecutive_frames": 4, "history_length": 8},
        "visualizer": {
            "mirror": True,
            "draw_landmarks": True,
            "show_fps": True,
            "show_bounding_box": True,
        },
        "classifier": {
            "mode": "hybrid",
            "model_path": "models/gesture_classifier.pkl",
            "min_confidence": 0.70,
            "ml_fallback_min_confidence": 0.75,
        },
        "speller": {
            "enable_tts": True,
            "enable_autocomplete": True,
        },
        "heuristics": {},
    }

    if not os.path.exists(config_path):
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg or default_config
    except Exception:
        return default_config


def main() -> None:
    logger = setup_logger()
    logger.info("Initializing Real-Time ASL Alphabet Fingerspelling Studio...")

    config = load_config()

    # 1. Initialize Camera with Watchdog
    cam_cfg = config.get("camera", {})
    camera = ThreadedCamera(
        device_id=cam_cfg.get("device_id", 0),
        width=cam_cfg.get("width", 1280),
        height=cam_cfg.get("height", 720),
        fps=cam_cfg.get("fps", 30),
        reconnect_timeout_sec=cam_cfg.get("reconnect_timeout_sec", 2.0),
    )

    # 2. Initialize Hand Tracker with Video Mode
    track_cfg = config.get("tracker", {})
    tracker = HandTracker(
        model_path=track_cfg.get("model_path", "models/hand_landmarker.task"),
        num_hands=track_cfg.get("max_num_hands", 1),
        min_detection_confidence=track_cfg.get("min_detection_confidence", 0.65),
        min_presence_confidence=track_cfg.get("min_presence_confidence", 0.55),
        min_tracking_confidence=track_cfg.get("min_tracking_confidence", 0.55),
        box_padding=track_cfg.get("box_padding", 20),
        running_mode=track_cfg.get("running_mode", "video"),
    )

    # 3. Filters, Smoothers & Trajectory Recognizers
    filter_cfg = config.get("filter", {})
    enable_smoothing = filter_cfg.get("enable_smoothing", True)
    landmark_smoother = ExponentialMovingAverage(alpha=filter_cfg.get("smoothing_alpha", 0.65))
    stroke_tracker = StrokeTrajectoryTracker() if filter_cfg.get("enable_trajectory", True) else None

    # 4. Initialize ASL Classifier & Normalizer with Config Heuristics
    heuristics_cfg = config.get("heuristics", {})
    cls_cfg = config.get("classifier", {})
    classifier = GestureClassifier(
        mode=cls_cfg.get("mode", "hybrid"),
        model_path=cls_cfg.get("model_path", "models/gesture_classifier.pkl"),
        min_confidence=cls_cfg.get("min_confidence", 0.70),
        ml_fallback_min_confidence=cls_cfg.get("ml_fallback_min_confidence", 0.75),
        heuristics=heuristics_cfg,
    )
    normalizer = LandmarkNormalizer(heuristics=heuristics_cfg)

    # 5. Debouncer & Visualizer
    deb_cfg = config.get("debouncer", {})
    debouncer = GestureDebouncer(
        consecutive_frames=deb_cfg.get("consecutive_frames", 4),
        history_length=deb_cfg.get("history_length", 8),
    )

    vis_cfg = config.get("visualizer", {})
    visualizer = ModernVisualizer(
        draw_landmarks=vis_cfg.get("draw_landmarks", True),
        show_fps=vis_cfg.get("show_fps", True),
        show_bounding_box=vis_cfg.get("show_bounding_box", True),
    )
    mirror_mode = vis_cfg.get("mirror", True)

    # 6. Speller, Auto-complete & Text-To-Speech
    speller_cfg = config.get("speller", {})
    word_predictor = WordPredictor() if speller_cfg.get("enable_autocomplete", True) else None
    tts = TextToSpeech(enabled=speller_cfg.get("enable_tts", True))

    window_name = "ASL Alphabet Recognition Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        camera.start()
    except Exception as e:
        logger.error("Failed to initialize camera: %s", e)
        logger.info("Check if another application is using the webcam, or adjust camera device_id in config.yaml.")
        return

    logger.info("ASL Studio running. Show ASL signs (A-Z) in front of the camera.")
    logger.info("Controls: [SPACE] Add Letter | [TAB] Auto-Complete | [C] Clear | [BACKSPACE] Delete | [Q/ESC] Quit")

    spelled_text = ""

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            if mirror_mode:
                frame = cv2.flip(frame, 1)

            # Detect hand with Video Mode
            hands = tracker.process(frame)

            raw_letter = "None"
            confidence = 0.0
            debug_info = None

            if hands:
                primary_hand = hands[0]

                # Handedness parity adjustment under camera mirroring
                effective_handedness = primary_hand.handedness
                if mirror_mode:
                    effective_handedness = "Left" if effective_handedness == "Right" else "Right"

                # Apply landmark EMA smoothing to eliminate high-frequency jitter
                if enable_smoothing:
                    smoothed_lm = landmark_smoother.update(primary_hand.landmarks_normalized)
                    primary_hand.landmarks_normalized = smoothed_lm

                # Extract geometric metrics
                debug_info = normalizer.analyze_hand(
                    primary_hand.landmarks_normalized,
                    handedness=effective_handedness,
                )

                # Update trajectory tracker for dynamic signs ('J' and 'Z')
                dynamic_stroke = None
                if stroke_tracker:
                    stroke_res = stroke_tracker.update(primary_hand.landmarks_normalized, debug_info)
                    if stroke_res:
                        dynamic_stroke, _ = stroke_res

                # Hybrid / Heuristic Classification
                raw_letter, confidence = classifier.classify(
                    primary_hand.landmarks_normalized,
                    handedness=effective_handedness,
                    dynamic_gesture=dynamic_stroke,
                )
            else:
                landmark_smoother.reset()
                if stroke_tracker:
                    stroke_tracker.reset()

            # Debounce prediction to stabilize letter display
            active_letter, active_conf = debouncer.update(raw_letter, confidence)

            # Calculate word auto-completion suggestions for current prefix
            suggestions: List[str] = []
            if word_predictor and spelled_text:
                current_word = spelled_text.split()[-1] if not spelled_text.endswith(" ") else ""
                if current_word:
                    suggestions = word_predictor.predict(current_word, max_results=3)

            # Render ASL HUD
            rendered = visualizer.render(
                frame=frame,
                hands=hands,
                active_letter=active_letter,
                confidence=active_conf,
                spelled_text=spelled_text,
                debug_info=debug_info,
                suggestions=suggestions,
            )

            cv2.imshow(window_name, rendered)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord(" "):  # Space: append current letter or space
                if active_letter not in ("None", "-", "Unknown"):
                    spelled_text += active_letter
                else:
                    if spelled_text and not spelled_text.endswith(" "):
                        last_word = spelled_text.split()[-1]
                        tts.speak(last_word)
                    spelled_text += " "
            elif key == 9:  # TAB key: accept top auto-complete suggestion
                if suggestions:
                    top_word = suggestions[0]
                    words = spelled_text.rstrip().split()
                    if words:
                        words[-1] = top_word
                        spelled_text = " ".join(words) + " "
                    else:
                        spelled_text = top_word + " "
                    tts.speak(top_word)
            elif key in (ord("c"), ord("C")):  # 'C': Clear
                spelled_text = ""
            elif key in (8, 127):  # Backspace
                spelled_text = spelled_text[:-1]

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Shutting down camera and closing windows...")
        camera.stop()
        tracker.close()
        cv2.destroyAllWindows()
        logger.info("Application cleanly closed.")


if __name__ == "__main__":
    main()
