"""
Temporal Filtering and Stabilization Module
Includes debouncing for gesture state transitions and smoothing filters for landmarks.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional, Tuple
import numpy as np


class GestureDebouncer:
    """
    Suppresses single-frame misclassifications by requiring a gesture
    to appear consistently across consecutive frames before confirming state change.
    """

    def __init__(self, consecutive_frames: int = 6, history_length: int = 12) -> None:
        self.consecutive_frames = max(1, consecutive_frames)
        self.history: deque[Tuple[str, float]] = deque(maxlen=max(self.consecutive_frames, history_length))
        self.current_gesture: str = "None"
        self.current_confidence: float = 0.0

    def update(self, raw_gesture: str, confidence: float) -> Tuple[str, float]:
        """
        Updates filter with newest raw prediction and returns debounced gesture state.
        
        Args:
            raw_gesture: The predicted gesture string for the current frame.
            confidence: The prediction confidence score [0.0, 1.0].
            
        Returns:
            Tuple of (debounced_gesture_name, average_confidence)
        """
        self.history.append((raw_gesture, confidence))

        if len(self.history) < self.consecutive_frames:
            return self.current_gesture, self.current_confidence

        # Check the most recent N frames
        recent = list(self.history)[-self.consecutive_frames:]
        recent_labels = [label for label, _ in recent]

        # If all N recent frames match the same gesture
        if len(set(recent_labels)) == 1:
            candidate = recent_labels[0]
            avg_conf = sum(conf for _, conf in recent) / len(recent)
            self.current_gesture = candidate
            self.current_confidence = avg_conf
        else:
            # If current gesture is still dominant in the whole history buffer, maintain it
            counts = Counter([label for label, _ in self.history])
            most_common, freq = counts.most_common(1)[0]
            if most_common == self.current_gesture and freq >= self.consecutive_frames:
                pass  # Keep current
            elif raw_gesture == "None":
                # Quickly decay to None if nothing is detected
                self.current_gesture = "None"
                self.current_confidence = 0.0

        return self.current_gesture, self.current_confidence

    def reset(self) -> None:
        """Clears the buffer and resets state."""
        self.history.clear()
        self.current_gesture = "None"
        self.current_confidence = 0.0


class ExponentialMovingAverage:
    """Smooths multi-dimensional coordinate arrays (e.g. landmarks) over time."""

    def __init__(self, alpha: float = 0.6) -> None:
        """
        Args:
            alpha: Smoothing factor between 0.0 and 1.0.
                   Higher alpha gives more weight to current frame (faster, more responsive).
                   Lower alpha gives more weight to previous history (smoother, slightly more lag).
        """
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.state: Optional[np.ndarray] = None

    def update(self, value: np.ndarray) -> np.ndarray:
        """Applies EMA smoothing to input array."""
        if self.state is None or self.state.shape != value.shape:
            self.state = value.copy().astype(np.float32)
            return self.state

        self.state = self.alpha * value + (1.0 - self.alpha) * self.state
        return self.state

    def reset(self) -> None:
        self.state = None

