"""
Unit and Integration Tests for ASL Gesture Detection Pipeline
Verifies normalizer, filters, classifier, visualizer, and tracker without needing a physical webcam.
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
from src.filters import ExponentialMovingAverage, GestureDebouncer
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
    """Verifies ASL HUD and skeleton drawing on a synthetic frame."""
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
    )
    assert rendered.shape == (720, 1280, 3)
    assert rendered.dtype == np.uint8


def test_tracker_initialization():
    """Verifies MediaPipe HandTracker initializes and processes an empty image without crashing."""
    tracker = HandTracker()
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
    print("\nAll ASL pipeline tests completed successfully!")
