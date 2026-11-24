"""
Main Application Entry Point for Real-Time ASL Fingerspelling Studio.
Orchestrates high-fps camera capture with auto-reconnect, MediaPipe video tracking,
landmark smoothing, dynamic stroke recognition (J/Z), hybrid classification,
debouncing, Gesture Event Management, Word Formation, Thumbs-Up Confirmation,
HUD rendering, auto-complete, and non-blocking Text-To-Speech (TTS).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List
import cv2
import numpy as np
import yaml

from src.capture import ThreadedCamera
from src.classifier import GestureClassifier
from src.filters import (
    ExponentialMovingAverage,
    GestureDebouncer,
    HandMotionDetector,
    OneEuroFilter,
    StrokeTrajectoryTracker,
    WordPredictor,
)
from src.gesture_commands import ControlGestureRecognizer, ControlGestureType
from src.gesture_event_manager import EventType, GestureEventManager, SystemState
from src.normalizer import LandmarkNormalizer
from src.tracker import HandTracker
from src.tts_engine import SpeechManager
from src.visualizer import ModernVisualizer
from src.word_builder import WordBuilder


def setup_logger() -> logging.Logger:
    """Configures formatted standard logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("ASLApp")


def _load_env_file(env_path: str = ".env") -> None:
    """Safely loads local .env file if present, setting environment variables without external dependencies."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration with sane fallbacks and secure environment variable overrides."""
    _load_env_file()

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
            "enable_smoothing": True,
            "enable_trajectory": True,
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
            "model_path": "models/lightgbm_classifier.pkl",
            "min_confidence": 0.92,
            "ml_fallback_min_confidence": 0.92,
        },
        "speller": {
            "enable_tts": True,
            "enable_autocomplete": True,
        },
        "word_builder": {
            "hold_frames": 18,
            "confirm_hold_frames": 14,
            "cooldown_frames": 12,
            "neutral_frames": 8,
            "min_confidence": 0.92,
            "enable_tts": True,
            "voice_rate": 150,
        },
        "heuristics": {},
    }

    cfg = default_config.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    cfg.update(loaded)
        except Exception:
            pass

    # Secure environment variable overrides for multi-environment deployments
    if "ASL_CAMERA_DEVICE_ID" in os.environ:
        try:
            cfg.setdefault("camera", {})["device_id"] = int(os.environ["ASL_CAMERA_DEVICE_ID"])
        except ValueError:
            pass
    if "ASL_MIN_CONFIDENCE" in os.environ:
        try:
            val = float(os.environ["ASL_MIN_CONFIDENCE"])
            cfg.setdefault("classifier", {})["min_confidence"] = val
            cfg.setdefault("word_builder", {})["min_confidence"] = val
        except ValueError:
            pass
    if "ASL_ENABLE_TTS" in os.environ:
        tts_flag = os.environ["ASL_ENABLE_TTS"].lower() in ("true", "1", "yes")
        cfg.setdefault("speller", {})["enable_tts"] = tts_flag
        cfg.setdefault("word_builder", {})["enable_tts"] = tts_flag

    return cfg


def main() -> None:
    logger = setup_logger()
    logger.info("Initializing Real-Time ASL Alphabet Fingerspelling Studio with TTS & Word Formation...")

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

    # 2. Initialize Hand Tracker with Video Mode and CLAHE Low-Light Preprocessing
    track_cfg = config.get("tracker", {})
    tracker = HandTracker(
        model_path=track_cfg.get("model_path", "models/hand_landmarker.task"),
        num_hands=track_cfg.get("max_num_hands", 1),
        min_detection_confidence=track_cfg.get("min_detection_confidence", 0.65),
        min_presence_confidence=track_cfg.get("min_presence_confidence", 0.55),
        min_tracking_confidence=track_cfg.get("min_tracking_confidence", 0.55),
        box_padding=track_cfg.get("box_padding", 20),
        running_mode=track_cfg.get("running_mode", "video"),
        enable_clahe=track_cfg.get("enable_clahe", True),
        clahe_clip_limit=track_cfg.get("clahe_clip_limit", 2.0),
    )

    # 3. Filters, Smoothers & Trajectory Recognizers
    filter_cfg = config.get("filter", {})
    enable_smoothing = filter_cfg.get("enable_smoothing", True)
    filter_type = filter_cfg.get("type", "one_euro").lower()
    if filter_type == "one_euro":
        landmark_smoother = OneEuroFilter(
            min_cutoff=filter_cfg.get("min_cutoff", 1.0),
            beta=filter_cfg.get("beta", 0.007),
            d_cutoff=filter_cfg.get("d_cutoff", 1.0),
        )
    else:
        landmark_smoother = ExponentialMovingAverage(alpha=filter_cfg.get("smoothing_alpha", 0.60))

    stroke_tracker = StrokeTrajectoryTracker() if filter_cfg.get("enable_trajectory", True) else None

    # 4. Initialize ASL Classifier & Normalizer
    heuristics_cfg = config.get("heuristics", {})
    cls_cfg = config.get("classifier", {})
    classifier = GestureClassifier(
        mode=cls_cfg.get("mode", "hybrid"),
        model_path=cls_cfg.get("model_path", "models/lightgbm_classifier.pkl"),
        min_confidence=cls_cfg.get("min_confidence", 0.92),
        ml_fallback_min_confidence=cls_cfg.get("ml_fallback_min_confidence", 0.92),
        heuristics=heuristics_cfg,
        enable_lstm=cls_cfg.get("enable_lstm", True),
        lstm_model_path=cls_cfg.get("lstm_model_path", "models/lstm_gesture_model.npz"),
    )
    normalizer = LandmarkNormalizer(heuristics=heuristics_cfg)

    # 5. Debouncer, Motion Detector & Visualizer
    deb_cfg = config.get("debouncer", {})
    debouncer = GestureDebouncer(
        consecutive_frames=deb_cfg.get("consecutive_frames", 3),
        history_length=deb_cfg.get("history_length", 8),
    )

    motion_cfg = config.get("motion_detector", {})
    motion_detector = None
    if motion_cfg.get("enable_steady_gate", True):
        motion_detector = HandMotionDetector(
            motion_threshold=motion_cfg.get("motion_threshold", 0.07),
            min_steady_frames=motion_cfg.get("min_steady_frames", 3),
            window_size=motion_cfg.get("window_size", 5),
        )

    vis_cfg = config.get("visualizer", {})
    visualizer = ModernVisualizer(
        draw_landmarks=vis_cfg.get("draw_landmarks", True),
        show_fps=vis_cfg.get("show_fps", True),
        show_bounding_box=vis_cfg.get("show_bounding_box", True),
    )
    mirror_mode = vis_cfg.get("mirror", True)

    # 6. Word Builder & Gesture Event Manager
    wb_cfg = config.get("word_builder", {})
    word_builder = WordBuilder()
    event_manager = GestureEventManager(
        hold_frames=wb_cfg.get("hold_frames", 10),
        confirm_hold_frames=wb_cfg.get("confirm_hold_frames", 8),
        cooldown_frames=wb_cfg.get("cooldown_frames", 6),
        neutral_frames_required=wb_cfg.get("neutral_frames", 3),
        min_confidence=wb_cfg.get("min_confidence", 0.92),
    )

    # 7. Auto-complete & Text-To-Speech Speech Manager
    speller_cfg = config.get("speller", {})
    word_predictor = WordPredictor() if speller_cfg.get("enable_autocomplete", True) else None
    enable_tts = wb_cfg.get("enable_tts", True) or speller_cfg.get("enable_tts", True)
    speech_manager = SpeechManager(enabled=enable_tts, voice_rate=wb_cfg.get("voice_rate", 150))

    window_name = "ASL Alphabet Recognition Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    spelled_text = ""

    # Interactive UI mouse click handler for on-screen buttons
    def on_mouse_click(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal spelled_text
        if event == cv2.EVENT_LBUTTONDOWN:
            btn = visualizer.get_button_at_pos(x, y)
            if btn == "DELETE":
                deleted = word_builder.delete_last()
                if deleted:
                    logger.info("Button [DEL] clicked -> Deleted '%s', Word: '%s'", deleted, word_builder.current_word)
                elif spelled_text:
                    spelled_text = spelled_text[:-1].rstrip()
                    logger.info("Button [DEL] clicked -> Backspaced spelled text to '%s'", spelled_text)
            elif btn == "CLEAR":
                cleared = word_builder.clear()
                spelled_text = ""
                logger.info("Button [CLEAR] clicked -> Cleared word '%s' and reset display", cleared)
            elif btn == "REPEAT":
                logger.info("Button [REPEAT] clicked -> Speaking last confirmed word '%s'", speech_manager.last_spoken_word)
                speech_manager.repeat_last()
            elif btn == "CONFIRM":
                confirmed = word_builder.confirm()
                if confirmed:
                    logger.info("Button [CONFIRM] clicked -> Confirmed '%s'! Speaking aloud...", confirmed)
                    speech_manager.speak(confirmed)
                    if spelled_text:
                        spelled_text += " " + confirmed
                    else:
                        spelled_text = confirmed

    cv2.setMouseCallback(window_name, on_mouse_click)

    try:
        camera.start()
    except Exception as e:
        logger.error("Failed to initialize camera: %s", e)
        logger.info("Check if another application is using the webcam, or adjust camera device_id in config.yaml.")
        return

    logger.info("ASL Studio running.")
    logger.info("Workflow: Hold signs to spell -> Show 👍 Thumbs-Up to Confirm & Speak word.")
    logger.info("Controls: [Hold Sign]: Add  |  [👍 Thumbs-Up]: Speak  |  [Del]: Backspace  |  [C]: Clear  |  [R]: Repeat  |  [Q]: Quit")

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
            effective_handedness = "Right"
            dynamic_stroke = None

            is_steady = True
            motion_score = 0.0

            if hands:
                primary_hand = hands[0]

                # Handedness parity adjustment under camera mirroring
                effective_handedness = primary_hand.handedness
                if mirror_mode:
                    effective_handedness = "Left" if effective_handedness == "Right" else "Right"

                # Apply landmark smoothing to eliminate high-frequency jitter
                if enable_smoothing:
                    smoothed_lm = landmark_smoother.update(primary_hand.landmarks_normalized)
                    primary_hand.landmarks_normalized = smoothed_lm

                # Extract geometric metrics
                debug_info = normalizer.analyze_hand(
                    primary_hand.landmarks_normalized,
                    handedness=effective_handedness,
                )

                # Track hand motion & steadiness
                if motion_detector:
                    is_steady, motion_score = motion_detector.update(
                        primary_hand.landmarks_normalized,
                        ref_scale=debug_info.get("ref_scale") if debug_info else None,
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
                    precomputed_m=debug_info,
                )
            else:
                landmark_smoother.reset()
                if stroke_tracker:
                    stroke_tracker.reset()
                if motion_detector:
                    motion_detector.reset()
                is_steady = False

            # When the hand is moving (transitioning between gestures or shaking),
            # suppress letter output so the model does not show transient random letters.
            # Once the hand gesture is held steady (or completing a dynamic stroke like J/Z),
            # debounce and display the correct alphabet sign.
            if motion_detector and not is_steady and dynamic_stroke is None:
                active_letter, active_conf = debouncer.update("None", 0.0)
            else:
                # Debounce prediction to stabilize letter display
                active_letter, active_conf = debouncer.update(raw_letter, confidence)

            # Process frame through Gesture Event Manager
            hand_lm = hands[0].landmarks_normalized if hands else None
            gesture_event = event_manager.process_frame(
                raw_letter=active_letter,
                confidence=active_conf,
                landmarks=hand_lm,
                debug_info=debug_info,
                handedness=effective_handedness,
                is_tts_speaking=speech_manager.is_speaking,
                is_steady=(is_steady or dynamic_stroke is not None),
            )

            # Handle discrete gesture events
            if gesture_event.event_type == EventType.LETTER:
                word_builder.add_letter(gesture_event.value)
                logger.info("Committed letter '%s' -> Formed word: '%s'", gesture_event.value, word_builder.current_word)
            elif gesture_event.event_type == EventType.CONFIRM:
                confirmed_word = word_builder.confirm()
                if confirmed_word:
                    logger.info("Confirmed word '%s' via 👍 Thumbs-Up! Pronouncing aloud...", confirmed_word)
                    speech_manager.speak(confirmed_word)
                    if spelled_text:
                        spelled_text += " " + confirmed_word
                    else:
                        spelled_text = confirmed_word
            elif gesture_event.event_type == EventType.DELETE:
                deleted = word_builder.delete_last()
                if deleted:
                    logger.info("Deleted letter '%s' -> Current word: '%s'", deleted, word_builder.current_word)

            # Calculate word auto-completion suggestions for current formed word
            suggestions: List[str] = []
            if word_predictor and word_builder.current_word:
                suggestions = word_predictor.predict(word_builder.current_word, max_results=3)

            # Render ASL HUD with word formation, buttons, status, and hold progress
            rendered = visualizer.render(
                frame=frame,
                hands=hands,
                active_letter=active_letter,
                confidence=active_conf,
                spelled_text=spelled_text,
                debug_info=debug_info,
                suggestions=suggestions,
                system_state=gesture_event.state.value,
                current_word=word_builder.current_word,
                letter_sequence=word_builder.letter_sequence,
                hold_progress=gesture_event.progress,
                last_spoken_word=speech_manager.last_spoken_word,
                tts_status=speech_manager.status_text,
                active_control="THUMBS_UP" if gesture_event.state == SystemState.CONFIRMING or gesture_event.value == "THUMBS_UP" else "NONE",
                is_steady=is_steady,
            )

            cv2.imshow(window_name, rendered)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key in (10, 13):  # ENTER: Manual Confirm & Speak
                confirmed = word_builder.confirm()
                if confirmed:
                    logger.info("Manual Confirm [ENTER] -> Speaking '%s'", confirmed)
                    speech_manager.speak(confirmed)
            elif key == ord(" "):  # Space: Manual Add current letter or confirm
                if active_letter not in ("None", "-", "Unknown") and active_conf >= 0.92:
                    word_builder.add_letter(active_letter)
                elif not word_builder.is_empty():
                    confirmed = word_builder.confirm()
                    if confirmed:
                        speech_manager.speak(confirmed)
            elif key == 9:  # TAB key: accept top auto-complete suggestion
                if suggestions:
                    top_word = suggestions[0]
                    word_builder.clear()
                    for ch in top_word:
                        word_builder.add_letter(ch)
                    confirmed = word_builder.confirm()
                    if confirmed:
                        speech_manager.speak(confirmed)
            elif key in (ord("c"), ord("C")):  # 'C': Clear word buffer and display
                word_builder.clear()
                spelled_text = ""
                logger.info("Key [C] pressed -> Cleared word buffer and reset display")
            elif key in (ord("r"), ord("R")):  # 'R': Repeat last spoken word
                speech_manager.repeat_last()
            elif key in (8, 127):  # Backspace: Delete last letter
                deleted = word_builder.delete_last()
                if not deleted and spelled_text:
                    spelled_text = spelled_text[:-1].rstrip()
                    logger.info("Key [Backspace] pressed -> Backspaced spelled text to '%s'", spelled_text)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Shutting down camera, TTS engine, and closing windows...")
        camera.stop()
        tracker.close()
        speech_manager.shutdown()
        cv2.destroyAllWindows()
        logger.info("Application cleanly closed.")


if __name__ == "__main__":
    main()
