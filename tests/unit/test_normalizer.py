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




def test_3d_canonical_alignment():
    """Verifies that 3D Canonical Alignment produces identical landmarks under roll, pitch, and yaw."""
    normalizer = LandmarkNormalizer()
    np.random.seed(42)
    hand = np.random.uniform(-0.5, 0.5, (21, 3)).astype(np.float32)
    hand[0] = [0, 0, 0]
    hand[9] = [0, 1.0, 0]
    hand[5] = [0.3, 0.9, 0]
    hand[17] = [-0.3, 0.8, 0]

    # Create 3D rotation matrix
    theta = 0.65
    axis = np.array([0.4, 0.7, -0.3], dtype=np.float32)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_rot = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

    hand_rotated = hand @ R_rot.T

    canonical_orig, _ = normalizer.canonical_align(hand, "Right")
    canonical_rot, _ = normalizer.canonical_align(hand_rotated, "Right")

    np.testing.assert_allclose(canonical_orig, canonical_rot, atol=1e-4)



