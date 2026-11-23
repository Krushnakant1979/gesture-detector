"""
Real-Time ASL Fingerspelling & Assistive Speech Studio
=====================================================

Modular Architecture:
- Capture & Perception:
    ThreadedCamera, HandTracker, HandData
- Geometry & Normalization:
    LandmarkNormalizer
- Temporal Filtering & Motion Gating:
    OneEuroFilter, ExponentialMovingAverage, GestureDebouncer,
    HandMotionDetector, StrokeTrajectoryTracker
- Classification & Control Recognition:
    GestureClassifier, LSTMRecurrentClassifier,
    ControlGestureRecognizer, ControlGestureType
- Interaction & State Machine:
    GestureEventManager, GestureEvent, EventType, SystemState,
    WordBuilder, WordPredictor
- Voice Synthesis:
    SpeechManager
- UI & Rendering:
    ModernVisualizer
"""

from __future__ import annotations

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
from src.gesture_event_manager import EventType, GestureEvent, GestureEventManager, SystemState
from src.lstm_recognizer import LSTMRecurrentClassifier
from src.normalizer import LandmarkNormalizer
from src.tracker import HandData, HandTracker
from src.tts_engine import SpeechManager
from src.visualizer import ModernVisualizer
from src.word_builder import WordBuilder

__version__ = "2.0.0"

__all__ = [
    # Perception
    "ThreadedCamera",
    "HandTracker",
    "HandData",
    # Geometry & Transforms
    "LandmarkNormalizer",
    # Classification
    "GestureClassifier",
    "LSTMRecurrentClassifier",
    "ControlGestureRecognizer",
    "ControlGestureType",
    # Filtering & Motion Detection
    "OneEuroFilter",
    "ExponentialMovingAverage",
    "GestureDebouncer",
    "HandMotionDetector",
    "StrokeTrajectoryTracker",
    # Assistive Interaction & State Machine
    "GestureEventManager",
    "GestureEvent",
    "EventType",
    "SystemState",
    "WordBuilder",
    "WordPredictor",
    "SpeechManager",
    # Visualization
    "ModernVisualizer",
]
