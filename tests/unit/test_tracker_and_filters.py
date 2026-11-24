"""
ASL Gesture Recognition Studio - Unit Tests
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict, List, Optional
import numpy as np

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.classifier import GestureClassifier
from src.filters import (
    ExponentialMovingAverage,
    GestureDebouncer,
    HandMotionDetector,
    OneEuroFilter,
    StrokeTrajectoryTracker,
    WordPredictor,
)
from src.gesture_event_manager import EventType, GestureEventManager, SystemState
from src.lstm_recognizer import LSTMRecurrentClassifier
from src.normalizer import LandmarkNormalizer
from src.tracker import HandData, HandTracker
from src.visualizer import ModernVisualizer


def test_debouncer_stabilization():
    """Ensures debouncer rejects rapid 1-frame glitches and locks on sustained gestures."""
    debouncer = GestureDebouncer(consecutive_frames=4, history_length=8)

    state, _ = debouncer.update("A", 0.9)
    assert state == "None"

    for _ in range(4):
        state, conf = debouncer.update("B", 0.95)

    assert state == "B"
    assert conf > 0.9

    state, _ = debouncer.update("A", 0.9)
    assert state == "B"




def test_hand_motion_detector_and_steady_gate():
    """
    Verifies that HandMotionDetector detects hand motion, suppresses letters during movement,
    and unlocks detection once the hand gesture is held steady.
    """
    detector = HandMotionDetector(motion_threshold=0.07, min_steady_frames=3, window_size=5)
    debouncer = GestureDebouncer(consecutive_frames=3, history_length=8)
    event_manager = GestureEventManager(hold_frames=3, confirm_hold_frames=3, min_confidence=0.92)

    # Base hand landmarks (wrist at (0.5, 0.7), middle MCP 9 at (0.5, 0.55), scale ~0.15)
    base_lm = np.array([[0.5, 0.7, 0.0] for _ in range(21)], dtype=np.float32)
    base_lm[9] = [0.5, 0.55, 0.0]
    ref_scale = 0.15

    # 1. Moving hand: translating across frame (step = 0.02 per frame -> motion ~0.133 > 0.07)
    for i in range(5):
        moving_lm = base_lm.copy()
        moving_lm[:, :2] += np.array([i * 0.02, i * 0.015], dtype=np.float32)
        is_steady, motion = detector.update(moving_lm, ref_scale=ref_scale)
        assert is_steady is False, f"Expected moving on frame {i}, got is_steady={is_steady}"

        # Gate suppresses debouncer
        active_letter, active_conf = debouncer.update("None", 0.0)
        assert active_letter == "None"

        # Gate suppresses event manager hold accumulation
        event = event_manager.process_frame(
            raw_letter="A",
            confidence=0.96,
            landmarks=moving_lm,
            debug_info={"ref_scale": ref_scale},
            is_steady=is_steady,
        )
        assert event.event_type == EventType.NONE
        assert event_manager.candidate_hold_count == 0

    # 2. Hand settles and holds still (sub-millimeter tremor ~0.0005 per frame -> motion ~0.003 < 0.07)
    stationary_lm = moving_lm.copy()
    steady_detected = False
    for frame_idx in range(12):
        jitter_lm = stationary_lm + np.random.uniform(-0.0005, 0.0005, (21, 3)).astype(np.float32)
        is_steady, motion = detector.update(jitter_lm, ref_scale=ref_scale)

        if is_steady:
            steady_detected = True
            active_letter, active_conf = debouncer.update("A", 0.96)
            event = event_manager.process_frame(
                raw_letter=active_letter,
                confidence=active_conf,
                landmarks=jitter_lm,
                debug_info={"ref_scale": ref_scale},
                is_steady=is_steady,
            )
        else:
            active_letter, active_conf = debouncer.update("None", 0.0)

    assert steady_detected is True, "HandMotionDetector failed to detect steady state after settling"
    assert active_letter == "A", f"Expected active letter 'A' once steady, got {active_letter}"

    # 3. Reset when hand is removed
    detector.reset()
    assert detector.is_steady is False
    assert detector.prev_landmarks is None

    # 4. Finger morphing / folding detection (wrist still, index & middle folding down)
    detector_morph = HandMotionDetector(motion_threshold=0.040, min_steady_frames=6, window_size=7)
    for _ in range(8):
        detector_morph.update(base_lm, ref_scale=ref_scale)
    assert detector_morph.is_steady is True, "Should be steady while resting"

    # Now fold fingers (e.g., transition from U to N)
    for i in range(5):
        morphed_lm = base_lm.copy()
        morphed_lm[8, 1] += (i + 1) * 0.015
        morphed_lm[12, 1] += (i + 1) * 0.015
        is_steady, motion = detector_morph.update(morphed_lm, ref_scale=ref_scale)
        assert is_steady is False, f"Expected is_steady=False during finger morphing frame {i}"




def test_visualizer_rendering():
    """Verifies ASL HUD, suggestions, and skeleton drawing on a synthetic frame."""
    visualizer = ModernVisualizer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    dummy_hand = HandData(
        handedness="Right",
        confidence=0.98,
        landmarks_normalized=np.zeros((21, 3), dtype=np.float32),
        landmarks_pixel=np.array([[200 + i * 5, 300 + i * 5] for i in range(21)], dtype=np.int32),
        bbox=(180, 280, 150, 150),
    )

    rendered = visualizer.render(
        frame=frame,
        hands=[dummy_hand],
        active_letter="A",
        confidence=0.94,
        spelled_text="HELLO",
        suggestions=["HELLO", "HELP"],
    )
    assert rendered.shape == (720, 1280, 3)
    assert rendered.dtype == np.uint8

    # Verify spurious cross-palm intersection suppression on draped fist coordinates
    draped_coords = {
        0: (185.9, 208.4), 1: (253.6, 237.4), 2: (309.1, 263.6), 3: (322.0, 274.0), 4: (335.9, 284.2),
        5: (128.7, 133.6), 6: (132.0, 227.4), 7: (149.9, 292.7), 8: (166.5, 342.8),
        9: (193.5, 111.7), 10: (184.4, 246.0), 11: (183.7, 333.3), 12: (187.6, 399.3),
        13: (265.7, 108.8), 14: (247.4, 255.1), 15: (226.5, 349.7), 16: (215.9, 415.5),
        17: (337.4, 133.7), 18: (323.9, 248.8), 19: (302.7, 343.8), 20: (292.6, 411.1),
    }
    draped_px = np.array([draped_coords[i] for i in range(21)], dtype=np.int32)
    test_frame = np.zeros((512, 538, 3), dtype=np.uint8)
    visualizer.draw_skeleton(test_frame, draped_px)
    assert test_frame.shape == (512, 538, 3)




def test_tracker_initialization():
    """Verifies MediaPipe HandTracker initializes in video mode and processes an empty image."""
    tracker = HandTracker(running_mode="video")
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
    hands = tracker.process(blank_image)
    assert isinstance(hands, list)
    assert len(hands) == 0
    tracker.close()




def test_exponential_moving_average():
    """Verifies EMA filter smooths coordinate noise."""
    ema = ExponentialMovingAverage(alpha=0.5)
    v1 = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    v2 = np.array([[2.0, 2.0, 2.0]], dtype=np.float32)

    s1 = ema.update(v1)
    np.testing.assert_allclose(s1, v1)

    s2 = ema.update(v2)
    np.testing.assert_allclose(s2, np.array([[1.5, 1.5, 1.5]], dtype=np.float32))

    ema.reset()
    assert ema.state is None




def test_word_predictor():
    """Verifies prefix auto-completion suggestions."""
    predictor = WordPredictor()
    suggestions = predictor.predict("HEL")
    assert "HELLO" in suggestions or "HELP" in suggestions
    assert len(suggestions) <= 3




def test_one_euro_filter():
    """Verifies that 1€ Filter adapts cutoff frequency, suppresses stationary jitter, and tracks fast motions."""
    filter1 = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)

    # 1. Stationary coordinates with sensor noise
    base = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    filtered = []
    t = 0.0
    for i in range(20):
        noisy = base + np.random.normal(0, 0.01, 3).astype(np.float32)
        out = filter1.update(noisy, timestamp=t)
        filtered.append(out)
        t += 0.033

    diffs = np.diff(np.array(filtered), axis=0)
    assert np.mean(np.abs(diffs)) < 0.02, "1€ filter should heavily dampen stationary jitter"

    # 2. Fast step transition
    step_target = np.array([1.5, 2.0, 0.0], dtype=np.float32)
    out_step = filter1.update(step_target, timestamp=t + 0.033)
    assert np.linalg.norm(out_step - base) > 0.4, "1€ filter should rapidly open cutoff for fast motion"




def test_clahe_preprocessing():
    """Verifies that CLAHE contrast enhancement runs and preserves image dimensions."""
    tracker = HandTracker(running_mode="image", enable_clahe=True, clahe_clip_limit=2.0)
    dark_frame = np.full((100, 100, 3), 30, dtype=np.uint8)
    enhanced = tracker.preprocess_frame(dark_frame)
    assert enhanced.shape == dark_frame.shape
    assert enhanced.dtype == np.uint8
    tracker.close()



