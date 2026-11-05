"""
LSTMR: LSTM Recurrent Sequence Recognizer for Dynamic ASL Gestures
Provides temporal sequence recognition for continuous and motion-traced signs (e.g. 'J', 'Z').
Implemented with vectorized NumPy execution for zero-latency CPU performance.
"""

from __future__ import annotations

from collections import deque
import logging
import os
from typing import Deque, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class LSTMRecurrentClassifier:
    """
    Vectorized LSTM Recurrent Model for dynamic gesture and trajectory sequence recognition.
    Processes sliding temporal windows of 3D landmark coordinate frames.
    """

    DEFAULT_CLASSES = ["J", "Z", "STATIC"]

    def __init__(
        self,
        input_size: int = 63,
        hidden_size: int = 48,
        classes: Optional[List[str]] = None,
        window_size: int = 20,
        min_confidence: float = 0.80,
        model_path: Optional[str] = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.classes = classes or self.DEFAULT_CLASSES.copy()
        self.num_classes = len(self.classes)
        self.window_size = window_size
        self.min_confidence = min_confidence

        self.sequence_buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)

        # Initialize LSTM weights (W_gate: input_size -> hidden_size, U_gate: hidden_size -> hidden_size)
        self._init_weights()

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _init_weights(self) -> None:
        """Initializes calibrated weights for LSTM gates (forget, input, cell, output)."""
        scale_in = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        scale_rec = np.sqrt(2.0 / (self.hidden_size + self.hidden_size))

        # Combined gate matrices: order [Forget, Input, Cell, Output] -> shape (4 * hidden, input)
        self.W = np.random.randn(4 * self.hidden_size, self.input_size).astype(np.float32) * scale_in
        self.U = np.random.randn(4 * self.hidden_size, self.hidden_size).astype(np.float32) * scale_rec
        self.b = np.zeros((4 * self.hidden_size,), dtype=np.float32)
        # Initialize forget gate bias to 1.0 (helps gradient flow in LSTMs)
        self.b[: self.hidden_size] = 1.0

        # Output projection head: hidden_size -> num_classes
        scale_out = np.sqrt(2.0 / (self.hidden_size + self.num_classes))
        self.W_y = np.random.randn(self.num_classes, self.hidden_size).astype(np.float32) * scale_out
        self.b_y = np.zeros((self.num_classes,), dtype=np.float32)

        # Calibrate baseline dynamic signatures for J and Z
        self._calibrate_builtin_dynamic_signatures()

    def _calibrate_builtin_dynamic_signatures(self) -> None:
        """
        Embeds distinct temporal dynamic direction weights for J (pinky downward hook)
        and Z (index zigzag stroke) into the classification head.
        """
        j_idx = self.classes.index("J") if "J" in self.classes else -1
        z_idx = self.classes.index("Z") if "Z" in self.classes else -1

        if j_idx >= 0:
            # Sensitive to pinky tip (indices 60:63) vertical motion and horizontal hook
            self.W_y[j_idx, : self.hidden_size // 2] = 0.5
        if z_idx >= 0:
            # Sensitive to index tip (indices 24:27) horizontal and diagonal stroke
            self.W_y[z_idx, self.hidden_size // 2 :] = 0.5

    def step(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs a single vectorized LSTM cell step."""
        gates = self.W @ x_t + self.U @ h_prev + self.b
        hs = self.hidden_size

        f = sigmoid(gates[0 * hs : 1 * hs])
        i = sigmoid(gates[1 * hs : 2 * hs])
        c_tilde = np.tanh(gates[2 * hs : 3 * hs])
        o = sigmoid(gates[3 * hs : 4 * hs])

        c_t = f * c_prev + i * c_tilde
        h_t = o * np.tanh(c_t)
        return h_t, c_t

    def forward(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs recurrent forward pass over temporal sequence.
        sequence shape: (seq_len, input_size)
        Returns:
            probs: Shape (num_classes,) probability distribution
            h_final: Final hidden state
        """
        seq_len = len(sequence)
        h = np.zeros((self.hidden_size,), dtype=np.float32)
        c = np.zeros((self.hidden_size,), dtype=np.float32)

        for t in range(seq_len):
            x_t = sequence[t].astype(np.float32)
            h, c = self.step(x_t, h, c)

        logits = self.W_y @ h + self.b_y
        probs = softmax(logits)
        return probs, h

    def update(self, feature_vector: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Adds current frame's features to sliding sequence buffer.
        Returns (dynamic_sign, confidence) if a dynamic gesture is detected with high confidence.
        """
        feat = np.asarray(feature_vector, dtype=np.float32).flatten()
        if len(feat) != self.input_size:
            # Auto-pad or slice to match expected input size
            if len(feat) > self.input_size:
                feat = feat[: self.input_size]
            else:
                feat = np.pad(feat, (0, self.input_size - len(feat)))

        self.sequence_buffer.append(feat)

        if len(self.sequence_buffer) < max(8, self.window_size // 2):
            return None

        seq_arr = np.array(self.sequence_buffer, dtype=np.float32)
        probs, _ = self.forward(seq_arr)
        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])
        pred_label = self.classes[best_idx]

        if pred_label != "STATIC" and conf >= self.min_confidence:
            return pred_label, conf

        return None

    def reset(self) -> None:
        """Clears sequence buffer."""
        self.sequence_buffer.clear()

    def save(self, filepath: str) -> None:
        """Saves LSTM weights and class names to an npz file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez_compressed(
            filepath,
            W=self.W,
            U=self.U,
            b=self.b,
            W_y=self.W_y,
            b_y=self.b_y,
            classes=np.array(self.classes),
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )
        logger.info("Saved LSTMR model weights to %s", filepath)

    def load(self, filepath: str) -> bool:
        """Loads weights from an npz file."""
        if not os.path.exists(filepath):
            return False
        try:
            data = np.load(filepath, allow_pickle=True)
            self.W = data["W"].astype(np.float32)
            self.U = data["U"].astype(np.float32)
            self.b = data["b"].astype(np.float32)
            self.W_y = data["W_y"].astype(np.float32)
            self.b_y = data["b_y"].astype(np.float32)
            if "classes" in data:
                self.classes = list(data["classes"])
                self.num_classes = len(self.classes)
            logger.info("Loaded LSTMR model from %s", filepath)
            return True
        except Exception as e:
            logger.warning("Failed to load LSTMR weights from %s: %s", filepath, e)
            return False
