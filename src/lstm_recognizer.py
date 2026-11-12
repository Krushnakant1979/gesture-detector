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
        self.tip_buffer: Deque[Dict[str, np.ndarray]] = deque(maxlen=self.window_size)

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

    def fit(
        self,
        X_seqs: List[np.ndarray],
        y_labels: List[str],
        epochs: int = 30,
        lr: float = 0.005,
        verbose: bool = True,
    ) -> float:
        """
        Trains the LSTM weights using vectorized Backpropagation Through Time (BPTT) and Adam.
        X_seqs: List of (T, 63) arrays
        y_labels: List of label strings ('J', 'Z', 'STATIC')
        """
        K = self.num_classes
        N = len(X_seqs)
        H, D = self.hidden_size, self.input_size

        y_onehot = np.zeros((N, K), dtype=np.float32)
        for i, lbl in enumerate(y_labels):
            idx = self.classes.index(lbl) if lbl in self.classes else 0
            y_onehot[i, idx] = 1.0

        mW, vW = np.zeros_like(self.W), np.zeros_like(self.W)
        mU, vU = np.zeros_like(self.U), np.zeros_like(self.U)
        mb, vb = np.zeros_like(self.b), np.zeros_like(self.b)
        mWy, vWy = np.zeros_like(self.W_y), np.zeros_like(self.W_y)
        mby, vby = np.zeros_like(self.b_y), np.zeros_like(self.b_y)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step_cnt = 0

        final_acc = 0.0
        for ep in range(epochs):
            indices = np.random.permutation(N)
            total_loss = 0.0
            correct = 0

            for idx in indices:
                step_cnt += 1
                x_seq = X_seqs[idx].astype(np.float32)
                y_target = y_onehot[idx]
                T = len(x_seq)

                hs = np.zeros((T + 1, H), dtype=np.float32)
                cs = np.zeros((T + 1, H), dtype=np.float32)
                fs = np.zeros((T, H), dtype=np.float32)
                ins = np.zeros((T, H), dtype=np.float32)
                c_tildes = np.zeros((T, H), dtype=np.float32)
                os_gate = np.zeros((T, H), dtype=np.float32)
                tanh_cs = np.zeros((T, H), dtype=np.float32)

                for t in range(T):
                    g = self.W @ x_seq[t] + self.U @ hs[t] + self.b
                    f = sigmoid(g[0:H])
                    i_gate = sigmoid(g[H : 2 * H])
                    c_tilde = np.tanh(g[2 * H : 3 * H])
                    o = sigmoid(g[3 * H : 4 * H])

                    fs[t], ins[t], c_tildes[t], os_gate[t] = f, i_gate, c_tilde, o
                    cs[t + 1] = f * cs[t] + i_gate * c_tilde
                    tanh_cs[t] = np.tanh(cs[t + 1])
                    hs[t + 1] = o * tanh_cs[t]

                logits = self.W_y @ hs[T] + self.b_y
                probs = softmax(logits)

                loss = -np.sum(y_target * np.log(np.maximum(probs, 1e-12)))
                total_loss += loss
                if np.argmax(probs) == np.argmax(y_target):
                    correct += 1

                # Backward pass
                dlogits = probs - y_target
                dWy = np.outer(dlogits, hs[T])
                dby = dlogits

                dh = self.W_y.T @ dlogits
                dW = np.zeros_like(self.W)
                dU = np.zeros_like(self.U)
                db = np.zeros_like(self.b)

                dg_prev = np.zeros(4 * H, dtype=np.float32)
                dc_next = np.zeros(H, dtype=np.float32)

                for t in reversed(range(T)):
                    if t < T - 1:
                        dh = self.U.T @ dg_prev
                        dc = dc_next * fs[t + 1]
                    else:
                        dc = np.zeros(H, dtype=np.float32)

                    do = dh * tanh_cs[t]
                    dg_o = do * os_gate[t] * (1.0 - os_gate[t])

                    dc += dh * os_gate[t] * (1.0 - tanh_cs[t] ** 2)

                    df = dc * cs[t]
                    dg_f = df * fs[t] * (1.0 - fs[t])

                    di = dc * c_tildes[t]
                    dg_i = di * ins[t] * (1.0 - ins[t])

                    dc_tilde = dc * ins[t]
                    dg_c = dc_tilde * (1.0 - c_tildes[t] ** 2)

                    dg = np.concatenate([dg_f, dg_i, dg_c, dg_o])
                    dg_prev = dg
                    dc_next = dc

                    dW += np.outer(dg, x_seq[t])
                    dU += np.outer(dg, hs[t])
                    db += dg

                # Adam parameter update
                for p, dp, mp_v, vp_v in [
                    (self.W, dW, mW, vW),
                    (self.U, dU, mU, vU),
                    (self.b, db, mb, vb),
                    (self.W_y, dWy, mWy, vWy),
                    (self.b_y, dby, mby, vby),
                ]:
                    mp_v[:] = beta1 * mp_v + (1.0 - beta1) * dp
                    vp_v[:] = beta2 * vp_v + (1.0 - beta2) * (dp ** 2)
                    m_hat = mp_v / (1.0 - beta1 ** step_cnt)
                    v_hat = vp_v / (1.0 - beta2 ** step_cnt)
                    p -= lr * m_hat / (np.sqrt(v_hat) + eps)

            final_acc = correct / N
            if verbose and ((ep + 1) % 5 == 0 or ep == epochs - 1):
                logger.info("LSTMR Epoch %2d/%2d: Loss = %.4f, Accuracy = %.1f%%", ep + 1, epochs, total_loss / N, final_acc * 100)

        return float(final_acc)

    def update(
        self,
        feature_vector: np.ndarray,
        raw_landmarks: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[str, float]]:
        """
        Adds current frame's features to sliding sequence buffer.
        Returns (dynamic_sign, confidence) if a dynamic gesture is detected with high confidence.
        """
        feat = np.asarray(feature_vector, dtype=np.float32).flatten()
        if len(feat) != self.input_size:
            if len(feat) > self.input_size:
                feat = feat[: self.input_size]
            else:
                feat = np.pad(feat, (0, self.input_size - len(feat)))

        self.sequence_buffer.append(feat)

        # Track fingertip positions
        if raw_landmarks is not None and len(raw_landmarks) == 21:
            self.tip_buffer.append({
                "wrist": raw_landmarks[0][:2].copy(),
                "index": raw_landmarks[8][:2].copy(),
                "pinky": raw_landmarks[20][:2].copy(),
            })

        if len(self.sequence_buffer) < max(8, self.window_size // 2):
            return None

        seq_arr = np.array(self.sequence_buffer, dtype=np.float32)

        # Kinetic displacement and path length validation
        # Use tip buffer if available, else canonical coordinates
        if len(self.tip_buffer) >= len(self.sequence_buffer):
            idx_pts = np.array([tb["index"] for tb in self.tip_buffer])
            pnk_pts = np.array([tb["pinky"] for tb in self.tip_buffer])
        else:
            # Fallback: landmarks 8 (24:27) and 20 (60:63)
            idx_pts = seq_arr[:, 24:27]
            pnk_pts = seq_arr[:, 60:63]

        idx_path = float(np.sum(np.linalg.norm(np.diff(idx_pts, axis=0), axis=1)))
        idx_disp = float(np.linalg.norm(idx_pts[-1] - idx_pts[0]))
        pnk_path = float(np.sum(np.linalg.norm(np.diff(pnk_pts, axis=0), axis=1)))
        pnk_disp = float(np.linalg.norm(pnk_pts[-1] - pnk_pts[0]))

        probs, _ = self.forward(seq_arr)
        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])
        pred_label = self.classes[best_idx]

        # Require genuine movement trajectories:
        if pred_label == "J":
            if pnk_path < 0.10 or pnk_disp < 0.04:
                pred_label = "STATIC"
        elif pred_label == "Z":
            if idx_path < 0.15 or idx_disp < 0.04:
                pred_label = "STATIC"

        if pred_label != "STATIC" and conf >= self.min_confidence:
            self.reset()
            return pred_label, conf

        return None

    def reset(self) -> None:
        """Clears sequence buffers."""
        self.sequence_buffer.clear()
        self.tip_buffer.clear()

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
