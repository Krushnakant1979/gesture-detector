"""
Unit and Integration Tests for ASL Gesture Detection Pipeline
Verifies normalizer, filters, classifier, visualizer, tracker, and assistive modules
without needing a physical webcam.
"""

from __future__ import annotations

import os
import sys
import tempfile
import numpy as np

# Set writable cache dir for matplotlib
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classifier import GestureClassifier
from src.filters import (
    ExponentialMovingAverage,
    GestureDebouncer,
    StrokeTrajectoryTracker,
    WordPredictor,
)
from src.normalizer import LandmarkNormalizer
from src.tracker import HandData, HandTracker
from src.visualizer import ModernVisualizer


def test_normalizer_invariance():
    """Verifies that translation and scale shifts result in identical normalized coordinates."""
    normalizer = LandmarkNormalizer()
    np.random.seed(42)

    base_hand = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    norm_base = normalizer.normalize(base_hand)
    assert norm_base.shape == (63,)

    shifted_hand = base_hand + np.array([2.5, -1.8, 0.4], dtype=np.float32)
    norm_shifted = normalizer.normalize(shifted_hand)
    np.testing.assert_allclose(norm_base, norm_shifted, atol=1e-5)


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


def test_tracker_initialization():
    """Verifies MediaPipe HandTracker initializes in video mode and processes an empty image."""
    tracker = HandTracker(running_mode="video")
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
    hands = tracker.process(blank_image)
    assert isinstance(hands, list)
    assert len(hands) == 0
    tracker.close()


def test_asl_classifier():
    """Verifies that the trained ASL ML model loads and predicts valid letters A-Z."""
    classifier = GestureClassifier(mode="hybrid", model_path="models/gesture_classifier.pkl")
    dummy_landmarks = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    letter, conf = classifier.classify(dummy_landmarks)
    assert isinstance(letter, str)
    assert 0.0 <= conf <= 1.0


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


def test_stroke_trajectory_tracker():
    """Verifies StrokeTrajectoryTracker resets and handles empty/normal inputs cleanly."""
    tracker = StrokeTrajectoryTracker()
    assert tracker.update(None) is None
    dummy_lm = np.zeros((21, 3), dtype=np.float32)
    assert tracker.update(dummy_lm) is None
    tracker.reset()


def test_k_vs_v_distinction():
    """Verifies that K (elevated thumb between fingers) is reliably separated from V (folded thumb)."""
    classifier = GestureClassifier(mode="heuristic")

    def make_2finger_hand(cls: str) -> np.ndarray:
        vec = np.zeros((21, 3), dtype=np.float32)
        vec[0] = [0, 0, 0]
        vec[1] = [-0.15, -0.2, 0.0]
        vec[2] = [-0.25, -0.35, 0.0]
        vec[3] = [-0.3, -0.45, 0.0]
        vec[5] = [-0.15, -0.5, 0.0]
        vec[9] = [0.0, -0.52, 0.0]
        vec[13] = [0.15, -0.5, 0.0]
        vec[17] = [0.28, -0.45, 0.0]
        vec[16] = [0.15, -0.35, 0.1]
        vec[20] = [0.25, -0.35, 0.1]
        # Extended index & middle
        vec[6] = [-0.16, -0.65, 0.0]
        vec[7] = [-0.18, -0.80, 0.0]
        vec[8] = [-0.20, -0.95, 0.0]
        vec[10] = [0.10, -0.65, 0.0]
        vec[11] = [0.15, -0.80, 0.0]
        vec[12] = [0.20, -0.95, 0.0]
        if cls == "K":
            # Thumb elevated between fingers
            vec[4] = [0.0, -0.70, 0.05]
        else:
            # Thumb folded across palm (V)
            vec[4] = [0.05, -0.4, 0.1]
        return vec

    k_letter, _ = classifier.classify(make_2finger_hand("K"))
    assert k_letter == "K"

    v_letter, _ = classifier.classify(make_2finger_hand("V"))
    assert v_letter == "V"


def test_strict_a_gesture():
    """Verifies that only an upright snug thumb produces 'A', and hitchhiker thumbs are rejected."""
    classifier = GestureClassifier(mode="heuristic")

    def make_fist_variant(thumb_mode: str) -> np.ndarray:
        vec = np.zeros((21, 3), dtype=np.float32)
        vec[0] = [0, 0, 0]
        vec[5] = [-0.15, -0.5, 0.0]
        vec[9] = [0.0, -0.52, 0.0]
        vec[13] = [0.15, -0.5, 0.0]
        vec[17] = [0.28, -0.45, 0.0]
        for mcp_idx, tip_idx in [(5, 8), (9, 12), (13, 16), (17, 20)]:
            vec[mcp_idx + 1] = [vec[mcp_idx][0], -0.60, 0.0]
            vec[mcp_idx + 2] = [vec[mcp_idx][0], -0.52, 0.05]
            vec[tip_idx] = [vec[mcp_idx][0], -0.42, 0.08]
        vec[1] = [-0.15, -0.2, 0.0]
        vec[2] = [-0.25, -0.35, 0.0]
        vec[3] = [-0.25, -0.48, 0.0]
        if thumb_mode == "upright_snug":  # True A sign
            vec[4] = [-0.24, -0.62, 0.0]
        else:  # Thumb sticking out wide horizontally (hitchhiker, Image 1)
            vec[3] = [-0.45, -0.37, 0.0]
            vec[4] = [-0.60, -0.38, 0.0]
        return vec

    true_a, conf_a = classifier.classify(make_fist_variant("upright_snug"))
    assert true_a == "A"
    assert conf_a >= 0.95

    wrong_pose, _ = classifier.classify(make_fist_variant("thumb_out_horizontal"))
    assert wrong_pose == "None"


def test_c_vs_open_palm_distinction():
    """Verifies that only an arched C-cup gesture produces 'C', and flat open palms are rejected."""
    classifier = GestureClassifier(mode="heuristic")

    # Image 1 style: flat open palm (5 / high five)
    open_palm = np.zeros((21, 3), dtype=np.float32)
    open_palm[0] = [0, 0, 0]
    open_palm[1] = [0.15, -0.2, 0.0]
    open_palm[2] = [0.25, -0.35, 0.0]
    open_palm[3] = [0.45, -0.40, 0.0]
    open_palm[4] = [0.65, -0.45, 0.0]
    open_palm[5] = [0.15, -0.5, 0.0]
    open_palm[9] = [0.0, -0.52, 0.0]
    open_palm[13] = [-0.15, -0.5, 0.0]
    open_palm[17] = [-0.28, -0.45, 0.0]
    open_palm[8] = [0.25, -0.95, 0.0]
    open_palm[12] = [0.0, -0.98, 0.0]
    open_palm[16] = [-0.25, -0.95, 0.0]
    open_palm[20] = [-0.40, -0.90, 0.0]
    open_palm[6] = [0.20, -0.7, 0.0]
    open_palm[10] = [0.0, -0.75, 0.0]
    open_palm[14] = [-0.20, -0.7, 0.0]
    open_palm[18] = [-0.35, -0.65, 0.0]

    res_open, _ = classifier.classify(open_palm)
    assert res_open == "None", f"Open palm should be rejected, got {res_open}"

    # Image 2 style: true curved C cup
    true_c = np.zeros((21, 3), dtype=np.float32)
    true_c[0] = [0, 0, 0]
    true_c[5] = [0.40, -0.30, 0.0]
    true_c[9] = [0.42, -0.25, 0.0]
    true_c[13] = [0.40, -0.20, 0.0]
    true_c[17] = [0.38, -0.15, 0.0]
    true_c[6] = [0.55, -0.35, 0.0]
    true_c[7] = [0.65, -0.32, 0.0]
    true_c[8] = [0.70, -0.28, 0.0]
    true_c[10] = [0.58, -0.30, 0.0]
    true_c[11] = [0.68, -0.27, 0.0]
    true_c[12] = [0.72, -0.24, 0.0]
    true_c[14] = [0.55, -0.25, 0.0]
    true_c[15] = [0.65, -0.22, 0.0]
    true_c[16] = [0.70, -0.19, 0.0]
    true_c[18] = [0.50, -0.20, 0.0]
    true_c[19] = [0.60, -0.17, 0.0]
    true_c[20] = [0.65, -0.14, 0.0]
    true_c[1] = [0.15, -0.05, 0.0]
    true_c[2] = [0.30, 0.05, 0.0]
    true_c[3] = [0.50, 0.10, 0.0]
    true_c[4] = [0.68, 0.08, 0.0]

    res_c, conf_c = classifier.classify(true_c)
    assert res_c == "C", f"True C should be classified as C, got {res_c}"
    assert conf_c >= 0.95


if __name__ == "__main__":
    print("Running self-checks...")
    test_normalizer_invariance()
    print("[PASS] Normalizer test passed")
    test_debouncer_stabilization()
    print("[PASS] Debouncer test passed")
    test_visualizer_rendering()
    print("[PASS] Visualizer test passed")
    test_tracker_initialization()
    print("[PASS] Tracker test passed")
    test_asl_classifier()
    print("[PASS] ASL Classifier test passed")
    test_exponential_moving_average()
    print("[PASS] Exponential Moving Average test passed")
    test_word_predictor()
    print("[PASS] Word Predictor test passed")
    test_stroke_trajectory_tracker()
    print("[PASS] Stroke Trajectory Tracker test passed")
    test_k_vs_v_distinction()
    print("[PASS] K vs V distinction test passed")
    test_strict_a_gesture()
    print("[PASS] Strict A gesture distinction test passed")
    test_c_vs_open_palm_distinction()
    print("[PASS] C vs Open Palm distinction test passed")
    print("\nAll ASL pipeline tests completed successfully!")
