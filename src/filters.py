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

import bisect
from collections import Counter, deque
import logging
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
            cur = self.current_gesture
            if cur != "None":
                freq = sum(1 for label, _ in self.history if label == cur)
                if freq >= req_frames:
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

        self.x_prev[:] = x_hat
        self.dx_prev[:] = dx_hat
        self.t_prev = now

        return x_hat

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
    - 'Z': Traced in the air with the index fingertip (3-phase zigzag right -> down-left -> right, or circular motion).
    - 'J': Traced in the air with the pinky fingertip (stroke downward -> counter-clockwise hook up).
    """

    def __init__(self, window_size: int = 25, hold_frames: int = 22) -> None:
        self.window_size = window_size
        self.hold_frames = hold_frames
        # History buffers store relative (tip - wrist) coordinates and absolute (tip) coordinates
        self.index_history: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self.index_abs_history: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self.pinky_history: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self.pinky_abs_history: Deque[Tuple[float, float]] = deque(maxlen=window_size)

        self.active_stroke: Optional[str] = None
        self.active_stroke_hold: int = 0

    def reset(self) -> None:
        """Clears all tracking history."""
        self.index_history.clear()
        self.index_abs_history.clear()
        self.pinky_history.clear()
        self.pinky_abs_history.clear()
        self.active_stroke = None
        self.active_stroke_hold = 0

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
        idx_tip = landmarks[8][:2]
        pnk_tip = landmarks[20][:2]
        idx_tip_rel = (idx_tip - wrist)  # Index tip relative to wrist
        pnk_tip_rel = (pnk_tip - wrist) # Pinky tip relative to wrist

        self.index_history.append((float(idx_tip_rel[0]), float(idx_tip_rel[1])))
        self.index_abs_history.append((float(idx_tip[0]), float(idx_tip[1])))
        self.pinky_history.append((float(pnk_tip_rel[0]), float(pnk_tip_rel[1])))
        self.pinky_abs_history.append((float(pnk_tip[0]), float(pnk_tip_rel[1])))

        if len(self.index_history) < 10:
            return None

        # Check Z stroke if index is extended and other fingers are curled (as in ASL 1 / D handshape)
        # Supports both standard 3-phase zigzag Z and circular motion Z.
        # Strictly rejects closed fists (S, T, A, E, M, N), hooked index (X), wide thumb (L), and open palm.
        is_index_gesture = bool(
            metrics
            and (metrics.get("ext_index") or metrics.get("up_index"))
            and metrics.get("straight_idx", 0.0) >= 0.82
            and not metrics.get("is_X_shape", False)
            and not metrics.get("index_hooked", False)
            and not metrics.get("is_L_shape", False)
            and not metrics.get("thumb_extended_out", False)
            and not metrics.get("is_flat_open_hand", False)
            and not (metrics.get("up_middle") or metrics.get("up_ring") or metrics.get("up_pinky"))
            and not (metrics.get("is_S_shape") or metrics.get("is_T_shape") or metrics.get("is_E_shape") or metrics.get("is_A_shape") or metrics.get("is_M_shape") or metrics.get("is_N_shape"))
            and (not metrics.get("ext_middle") or metrics.get("straight_mid", 0.0) < 0.75)
            and (not metrics.get("ext_ring") or metrics.get("straight_rng", 0.0) < 0.75)
            and (not metrics.get("ext_pinky") or metrics.get("straight_pnk", 0.0) < 0.75)
        )
        if is_index_gesture:
            if self._detect_z_stroke() or self._detect_circle_stroke():
                self.active_stroke = "Z"
                self.active_stroke_hold = self.hold_frames
                self.index_history.clear()
                self.index_abs_history.clear()
                return "Z", 0.98

        # Check J stroke if pinky is extended and other fingers are curled
        if (
            metrics
            and metrics.get("ext_pinky")
            and not metrics.get("ext_index")
            and not metrics.get("horiz_index")
            and not metrics.get("is_G_shape")
        ):
            if self._detect_j_stroke():
                self.active_stroke = "J"
                self.active_stroke_hold = self.hold_frames
                self.pinky_history.clear()
                self.pinky_abs_history.clear()
                return "J", 0.98

        return None

    def _detect_circle_stroke(self) -> bool:
        """
        Detects circular motion of index fingertip (clockwise or counter-clockwise).
        Checks both relative trajectory (wrist-anchored) and absolute trajectory in frame.
        """
        return self._is_circular(self.index_history) or self._is_circular(self.index_abs_history)

    def _is_circular(self, pts_queue: Deque[Tuple[float, float]]) -> bool:
        """
        Analyzes a 2D point sequence to determine if it describes a circular or closed-loop motion:
        1. Bounding box span in both X and Y must exceed minimum movement threshold (0.03).
        2. Aspect ratio (min_span / max_span) must be >= 0.30 (not a straight line).
        3. Cumulative angular displacement relative to trajectory centroid must exceed ~270 degrees (4.7 rad).
        4. Direction of angular progression must be consistent (>= 60% in dominant direction).
        """
        if len(pts_queue) < 10:
            return False

        pts = np.array(pts_queue, dtype=np.float32)
        span_x = float(np.ptp(pts[:, 0]))
        span_y = float(np.ptp(pts[:, 1]))

        # Minimum spatial movement in both axes
        if span_x < 0.03 or span_y < 0.03:
            return False

        # Must not be a flat line in X or Y
        aspect = float(min(span_x, span_y) / (max(span_x, span_y) + 1e-6))
        if aspect < 0.30:
            return False

        # Calculate angles relative to centroid
        center = np.mean(pts, axis=0)
        rel = pts - center
        angles = np.arctan2(rel[:, 1], rel[:, 0])

        # Angular difference with unwrapping across [-pi, pi]
        d_theta = np.diff(angles)
        d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
        total_angle = float(np.sum(d_theta))

        # At least ~270 degrees (4.71 radians) of continuous angular sweep in either direction
        if abs(total_angle) < 4.7:
            return False

        # Consistency of rotation direction (at least 60% of steps in dominant direction)
        dom_sign = 1.0 if total_angle > 0 else -1.0
        same_dir_steps = int(np.sum(np.sign(d_theta) == dom_sign))
        if same_dir_steps / len(d_theta) < 0.60:
            return False

        return True

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

        cond_down = dy_down > 0.035
        cond_hook = (abs(dx_hook) > 0.02) or (dy_hook < -0.02)

        return bool(cond_down and cond_hook)


class HandMotionDetector:
    """
    Detects whether the hand is moving or held steady.
    Tracks displacement of key landmarks (wrist, MCP knuckles, fingertips) normalized
    by hand scale (e.g. wrist to middle MCP distance).

    Suppresses transient letter predictions while the hand is in motion (moving between
    signs or adjusting position) and signals steadiness once the hand remains still for
    a configurable consecutive number of frames.
    """

    def __init__(
        self,
        motion_threshold: float = 0.040,
        min_steady_frames: int = 6,
        window_size: int = 7,
    ) -> None:
        self.motion_threshold = float(motion_threshold)
        self.min_steady_frames = max(1, int(min_steady_frames))
        self.window_size = max(2, int(window_size))

        self.prev_landmarks: Optional[np.ndarray] = None
        self.motion_history: Deque[float] = deque(maxlen=self.window_size)
        self.steady_counter: int = 0
        self.is_steady: bool = False
        self.last_motion_score: float = 0.0

    def update(
        self,
        landmarks: Optional[np.ndarray],
        ref_scale: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Updates motion history with current landmarks.
        Returns:
            (is_steady: bool, motion_score: float)
        """
        if landmarks is None or len(landmarks) < 21:
            self.reset()
            return False, 1.0

        current = np.asarray(landmarks, dtype=np.float32)[:, :2]

        if self.prev_landmarks is None:
            self.prev_landmarks = current.copy()
            self.steady_counter = 0
            self.is_steady = False
            self.last_motion_score = 0.0
            return False, 0.0

        # Compute hand scale if not provided
        if ref_scale is None or ref_scale <= 1e-4:
            scale = float(np.linalg.norm(current[9] - current[0]))
            if scale < 1e-4:
                scale = 0.15
        else:
            scale = float(ref_scale)

        # Track both palm translation (0, 5, 9, 13, 17) and fingertip articulation (4, 8, 12, 16, 20)
        palm_indices = [0, 5, 9, 13, 17]
        tip_indices = [4, 8, 12, 16, 20]
        all_indices = palm_indices + tip_indices

        displacements = np.linalg.norm(current[all_indices] - self.prev_landmarks[all_indices], axis=1)
        palm_disps = displacements[:len(palm_indices)]
        tip_disps = displacements[len(palm_indices):]

        disp_palm_mean = float(np.mean(palm_disps))
        disp_tip_max = float(np.max(tip_disps))
        disp_tip_mean = float(np.mean(tip_disps))
        disp_all_mean = float(np.mean(displacements))

        # Effective displacement captures hand translation AND finger curling/unfolding
        effective_disp = max(
            disp_palm_mean,
            0.55 * disp_tip_max + 0.45 * disp_tip_mean,
            0.50 * (disp_all_mean + float(np.max(displacements))),
        )

        norm_motion = effective_disp / scale
        self.last_motion_score = norm_motion
        self.motion_history.append(norm_motion)
        self.prev_landmarks = current.copy()

        # Check motion threshold with temporal window smoothing
        window_mean = float(np.mean(self.motion_history)) if self.motion_history else norm_motion
        is_frame_still = (
            norm_motion < self.motion_threshold
            and window_mean < self.motion_threshold
        )

        if is_frame_still:
            self.steady_counter += 1
            if self.steady_counter >= self.min_steady_frames:
                self.is_steady = True
        else:
            self.steady_counter = 0
            self.is_steady = False

        return self.is_steady, norm_motion

    def reset(self) -> None:
        """Resets motion state and history."""
        self.prev_landmarks = None
        self.motion_history.clear()
        self.steady_counter = 0
        self.is_steady = False
        self.last_motion_score = 0.0


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
        """Returns top matching full words starting with prefix using binary search."""
        pref = prefix.strip().upper()
        if not pref:
            return []
        idx = bisect.bisect_left(self.words, pref)
        matches: List[str] = []
        for i in range(idx, len(self.words)):
            w = self.words[i]
            if not w.startswith(pref):
                break
            if w != pref:
                matches.append(w)
                if len(matches) >= max_results:
                    break
        return matches
