"""
Temporal Filtering, Stabilization, and Assistive Modules
Includes:
- GestureDebouncer: temporal smoothing of discrete classification decisions
- ExponentialMovingAverage: coordinate jitter reduction
- StrokeTrajectoryTracker: dynamic trajectory tracking for 'J' and 'Z' strokes
- WordPredictor: fast prefix auto-completion
- TextToSpeech: non-blocking audio voice synthesis
"""

from __future__ import annotations

from collections import Counter, deque
import logging
import os
import subprocess
import threading
import time
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class GestureDebouncer:
    """
    Suppresses single-frame misclassifications by requiring a gesture
    to appear consistently across consecutive frames before confirming state change.
    Provides fast, low-latency lock-on while preventing single-frame flicker.
    """

    def __init__(self, consecutive_frames: int = 3, history_length: int = 8) -> None:
        self.consecutive_frames = max(1, consecutive_frames)
        self.history: Deque[Tuple[str, float]] = deque(maxlen=max(self.consecutive_frames, history_length))
        self.current_gesture: str = "None"
        self.current_confidence: float = 0.0
        self.none_counter: int = 0

    def update(self, raw_gesture: str, confidence: float) -> Tuple[str, float]:
        """
        Updates filter with newest raw prediction and returns debounced gesture state.
        - Fast lock-on: confirms in consecutive_frames (~100ms at 30fps).
        - Single-frame glitches are rejected.
        - Graceful decay: requires 3 consecutive None frames before dropping held gesture.
        """
        self.history.append((raw_gesture, confidence))

        if raw_gesture == "None":
            self.none_counter += 1
            if self.none_counter >= 3:
                self.current_gesture = "None"
                self.current_confidence = 0.0
            return self.current_gesture, self.current_confidence
        else:
            self.none_counter = 0

        # Fast threshold: for high confidence (>= 0.90) or initial detection from None,
        # use min(3, consecutive_frames) to ensure lightning-fast reaction.
        req_frames = min(self.consecutive_frames, 3) if (self.current_gesture == "None" or confidence >= 0.92) else self.consecutive_frames

        if len(self.history) < req_frames:
            return self.current_gesture, self.current_confidence

        # Check the most recent req_frames
        recent = list(self.history)[-req_frames:]
        recent_labels = [label for label, _ in recent]

        # If all recent frames agree on the same gesture
        if len(set(recent_labels)) == 1 and recent_labels[0] != "None":
            candidate = recent_labels[0]
            avg_conf = sum(conf for _, conf in recent) / len(recent)
            self.current_gesture = candidate
            self.current_confidence = avg_conf
        else:
            # If current gesture is still dominant in the history buffer, maintain it
            counts = Counter([label for label, _ in self.history if label != "None"])
            if counts:
                most_common, freq = counts.most_common(1)[0]
                if most_common == self.current_gesture and freq >= req_frames:
                    pass  # Keep current gesture steady

        return self.current_gesture, self.current_confidence

    def reset(self) -> None:
        """Clears the buffer and resets state."""
        self.history.clear()
        self.current_gesture = "None"
        self.current_confidence = 0.0
        self.none_counter = 0


class OneEuroFilter:
    """
    Adaptive 1€ (One Euro) Filter for low-latency, jitter-free landmark smoothing.
    Automatically adapts cutoff frequency based on movement speed:
    - Low speed (holding still): applies high filtering to eliminate tremor/jitter.
    - High speed (moving hand): increases cutoff frequency for zero-lag tracking.
    Ref: Casiez, G., Roussel, N. and Vogel, D. (CHI 2012)
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(rate: float, cutoff: np.ndarray) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-4))
        te = 1.0 / max(rate, 1e-4)
        return 1.0 / (1.0 + tau / te)

    def update(self, x: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Filters input coordinate array.
        x: ndarray of shape (21, 3) or (N,)
        timestamp: monotonic seconds (optional, defaults to time.time())
        """
        now = time.time() if timestamp is None else float(timestamp)
        x = np.asarray(x, dtype=np.float32)

        if self.x_prev is None or self.t_prev is None or self.x_prev.shape != x.shape:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = now
            return x.copy()

        dt = now - self.t_prev
        if dt < 1e-4:
            dt = 1.0 / 30.0
        elif dt > 0.5:
            self.reset()
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = now
            return x.copy()

        rate = 1.0 / dt

        # Estimate derivative of input signal
        dx = (x - self.x_prev) * rate
        a_d = self._alpha(rate, np.full_like(x, self.d_cutoff))
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Adaptive cutoff based on velocity magnitude
        speed = np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(rate, cutoff)

        # Filter the signal
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = now

        return x_hat.copy()

    def reset(self) -> None:
        """Resets filter memory."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class ExponentialMovingAverage:
    """Smooths multi-dimensional coordinate arrays (e.g. landmarks) over time to remove jitter."""

    def __init__(self, alpha: float = 0.65) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.state: Optional[np.ndarray] = None

    def update(self, value: np.ndarray) -> np.ndarray:
        """Applies EMA smoothing to input array."""
        if self.state is None or self.state.shape != value.shape:
            self.state = value.copy().astype(np.float32)
            return self.state

        self.state = self.alpha * value + (1.0 - self.alpha) * self.state
        return self.state.copy()

    def reset(self) -> None:
        """Resets smoothed state."""
        self.state = None


class StrokeTrajectoryTracker:
    """
    Tracks fingertip motion trajectories relative to wrist to recognize dynamic ASL signs:
    - 'Z': Traced in the air with the index fingertip (stroke right -> down-left -> right).
    - 'J': Traced in the air with the pinky fingertip (stroke downward -> counter-clockwise hook up).
    """

    def __init__(self, window_size: int = 25, hold_frames: int = 12) -> None:
        self.window_size = window_size
        self.hold_frames = hold_frames
        # History buffers store relative (tip - wrist) coordinates
        self.index_history: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self.pinky_history: Deque[Tuple[float, float]] = deque(maxlen=window_size)

        self.active_stroke: Optional[str] = None
        self.active_stroke_hold: int = 0

    def update(
        self,
        landmarks: Optional[np.ndarray],
        metrics: Optional[Dict[str, any]] = None,
    ) -> Optional[Tuple[str, float]]:
        """
        Updates trajectory buffers with latest landmarks.
        Returns (stroke_letter, confidence) if a dynamic gesture is active/recognized.
        """
        # Decrement active hold timer if active
        if self.active_stroke_hold > 0:
            self.active_stroke_hold -= 1
            if self.active_stroke_hold > 0 and self.active_stroke:
                return self.active_stroke, 0.98
            else:
                self.active_stroke = None

        if landmarks is None or len(landmarks) < 21:
            self.reset()
            return None

        wrist = landmarks[0][:2]
        idx_tip_rel = (landmarks[8][:2] - wrist)  # Index tip relative to wrist
        pnk_tip_rel = (landmarks[20][:2] - wrist) # Pinky tip relative to wrist

        self.index_history.append((float(idx_tip_rel[0]), float(idx_tip_rel[1])))
        self.pinky_history.append((float(pnk_tip_rel[0]), float(pnk_tip_rel[1])))

        if len(self.index_history) < 12:
            return None

        # Check Z stroke if index is extended and other fingers are curled
        if metrics and metrics.get("ext_index") and not metrics.get("ext_middle"):
            if self._detect_z_stroke():
                self.active_stroke = "Z"
                self.active_stroke_hold = self.hold_frames
                self.index_history.clear()
                return "Z", 0.98

        # Check J stroke if pinky is extended and other fingers are curled
        if metrics and metrics.get("ext_pinky") and not metrics.get("ext_index"):
            if self._detect_j_stroke():
                self.active_stroke = "J"
                self.active_stroke_hold = self.hold_frames
                self.pinky_history.clear()
                return "J", 0.98

        return None

    def _detect_z_stroke(self) -> bool:
        """
        Detects Z pattern: 3 consecutive motion phases:
        Phase 1: Rightward movement (+dx)
        Phase 2: Diagonal down-left movement (-dx, +dy)
        Phase 3: Rightward movement (+dx)
        """
        pts = list(self.index_history)
        n = len(pts)
        if n < 12:
            return False

        # Segment into three temporal windows
        p1 = pts[: n // 3]
        p2 = pts[n // 3 : 2 * n // 3]
        p3 = pts[2 * n // 3 :]

        dx1 = p1[-1][0] - p1[0][0]
        dy1 = p1[-1][1] - p1[0][1]

        dx2 = p2[-1][0] - p2[0][0]
        dy2 = p2[-1][1] - p2[0][1]

        dx3 = p3[-1][0] - p3[0][0]
        dy3 = p3[-1][1] - p3[0][1]

        # Phase 1: horizontal motion to the right
        cond1 = dx1 > 0.04 and abs(dx1) > 0.8 * abs(dy1)
        # Phase 2: diagonal down and left
        cond2 = dx2 < -0.03 and dy2 > 0.03
        # Phase 3: horizontal motion to the right
        cond3 = dx3 > 0.04 and abs(dx3) > 0.8 * abs(dy3)

        return bool(cond1 and cond2 and cond3)

    def _detect_j_stroke(self) -> bool:
        """
        Detects J pattern:
        Pinky traces downwards, followed by curved sweep to the left and upward hook.
        """
        pts = list(self.pinky_history)
        n = len(pts)
        if n < 10:
            return False

        p_down = pts[: n // 2]
        p_hook = pts[n // 2 :]

        dy_down = p_down[-1][1] - p_down[0][1] # Moving down (+Y in image coordinates)
        dx_hook = p_hook[-1][0] - p_hook[0][0] # Sweeping left (-X) or curve
        dy_hook = p_hook[-1][1] - p_hook[0][1] # Hooking back up (-Y)

        cond_down = dy_down > 0.04
        cond_hook = (dx_hook < -0.02) or (dy_hook < -0.02)

        return bool(cond_down and cond_hook)

    def reset(self) -> None:
        """Clears trajectory history."""
        self.index_history.clear()
        self.pinky_history.clear()
        self.active_stroke = None
        self.active_stroke_hold = 0


class WordPredictor:
    """Lightweight in-memory lexicon for fast prefix word auto-completion."""

    COMMON_WORDS: List[str] = [
        "HELLO", "HELP", "HERE", "HOW", "HAVE", "HOME", "HAND", "HEAR", "HOPE",
        "WELCOME", "WHAT", "WHEN", "WHERE", "WHY", "WHO", "WHICH", "WITH", "WORK", "WORLD",
        "THANK", "THANKS", "THAT", "THIS", "THE", "THEY", "THERE", "THINK", "TIME",
        "PLEASE", "PEOPLE", "PLAY", "PEACE", "PRACTICE", "PROBLEM",
        "NAME", "NICE", "NEED", "NEW", "NOW", "NEVER", "NO", "NOT",
        "GOOD", "GREAT", "GIVE", "GO", "GET", "GOING", "GAME",
        "YES", "YOU", "YOUR", "YEAR",
        "MEET", "MAKE", "MORE", "MUCH", "MANY", "MY", "MAN",
        "CAN", "COME", "COULD", "CALL", "CARE",
        "FRIEND", "FAMILY", "FOOD", "FEEL", "FIND", "FIRST",
        "ASL", "SIGN", "STUDIO", "SPELL", "START", "STOP", "SEE", "SAY", "SPEAK",
        "LEARN", "LOVE", "LIKE", "LOOK", "LIFE", "LIVE",
        "ALL", "ABOUT", "AFTER", "AGAIN", "ALWAYS", "AND", "ARE",
        "TODAY", "TOMORROW", "TALK", "TELL", "TAKE",
        "UNDERSTAND", "USE", "US",
        "VERY", "VOICE", "VIDEO",
    ]

    def __init__(self, custom_words: Optional[List[str]] = None) -> None:
        vocab = set(self.COMMON_WORDS)
        if custom_words:
            vocab.update([w.upper().strip() for w in custom_words if w])
        self.words: List[str] = sorted(vocab)

    def predict(self, prefix: str, max_results: int = 3) -> List[str]:
        """Returns top matching full words starting with prefix."""
        pref = prefix.strip().upper()
        if not pref:
            return []
        matches = [w for w in self.words if w.startswith(pref) and w != pref]
        return matches[:max_results]


class TextToSpeech:
    """Non-blocking asynchronous speech synthesis helper."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._has_pyttsx3 = False
        self._engine = None

        if self.enabled:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 160)
                self._has_pyttsx3 = True
                logger.info("pyttsx3 Text-To-Speech engine initialized.")
            except Exception:
                self._has_pyttsx3 = False
                logger.info("Native OS speech synthesis fallback enabled.")

    def speak(self, text: str) -> None:
        """Speaks the text asynchronously in a background thread without blocking frames."""
        if not self.enabled or not text.strip():
            return

        clean_text = "".join(c for c in text.strip() if c.isalnum() or c.isspace())
        if not clean_text:
            return

        thread = threading.Thread(
            target=self._speak_worker,
            args=(clean_text,),
            daemon=True,
            name="TTSWorkerThread",
        )
        thread.start()

    def _speak_worker(self, text: str) -> None:
        if self._has_pyttsx3 and self._engine:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
                return
            except Exception as e:
                logger.debug("pyttsx3 exception: %s", e)

        # On Windows, use built-in System.Speech.Synthesis asynchronously
        if os.name == "nt":
            try:
                cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            except Exception as e:
                logger.debug("Windows System.Speech error: %s", e)
