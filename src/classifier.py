"""
ASL Gesture Classifier — Complete A-Z Heuristic Engine
Based on verified biomechanical analysis of official ASL fingerspelling charts.

=== DETECTION PRIORITY ORDER ===
Priority 0 : O  (tight finger-tip cluster meeting thumb)
Priority 1 : X  (index PIP elevated, tip hooked back down)
Priority 2 : Q  (index + thumb both pointing DOWN together)
Group A     : 4 extended fingers → B, C
Group B     : 3 extended fingers → W, F
Group C     : 2 extended fingers → P, H, K, R, V, U
Group D     : 1 extended finger  → G, L, D, Z (index)  /  Y, J, I (pinky)
Group E     : 0 extended fingers → M, N, T, S, A, E  (fist variants)
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Tuple
import numpy as np

from src.normalizer import LandmarkNormalizer

logger = logging.getLogger(__name__)


class GestureClassifier:
    """Classifies hand gestures into ASL Alphabet characters (A-Z)."""

    def __init__(
        self,
        mode: str = "heuristic",
        model_path: Optional[str] = None,
        min_confidence: float = 0.65,
    ) -> None:
        self.mode = mode.lower()
        self.min_confidence = min_confidence
        self.normalizer = LandmarkNormalizer()
        self.ml_model = None

        if self.mode == "ml" and model_path and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.ml_model = pickle.load(f)
                logger.info("Loaded ML model from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load ML model: %s", e)

    def classify(self, landmarks_3d: np.ndarray, handedness: str = "Right") -> Tuple[str, float]:
        """Classify 21 3D landmarks into an ASL letter. Returns ('None', 0.0) on no match."""
        if self.mode == "ml" and self.ml_model is not None:
            return self._predict_ml(landmarks_3d)
        return self.classify_asl(landmarks_3d, handedness)

    def _predict_ml(self, landmarks_3d: np.ndarray) -> Tuple[str, float]:
        features = self.normalizer.normalize(landmarks_3d).reshape(1, -1)
        try:
            if hasattr(self.ml_model, "predict_proba"):
                probs = self.ml_model.predict_proba(features)[0]
                best_idx = int(np.argmax(probs))
                conf = float(probs[best_idx])
                if conf >= self.min_confidence:
                    return str(self.ml_model.classes_[best_idx]), conf
            else:
                return str(self.ml_model.predict(features)[0]), 0.85
        except Exception as e:
            logger.error("ML inference error: %s", e)
        return "None", 0.0

    def classify_asl(self, lm: np.ndarray, handedness: str = "Right") -> Tuple[str, float]:
        """
        Comprehensive biomechanical classification of all 26 ASL letters.

        Landmark layout (MediaPipe):
          0=Wrist
          1-4  : Thumb  (CMC, MCP, IP, TIP)
          5-8  : Index  (MCP, PIP, DIP, TIP)
          9-12 : Middle (MCP, PIP, DIP, TIP)
          13-16: Ring   (MCP, PIP, DIP, TIP)
          17-20: Pinky  (MCP, PIP, DIP, TIP)

        Image Y increases downward. 'Up' = smaller Y value.
        """
        wrist = lm[0]
        hand_span = float(np.linalg.norm(lm[9] - wrist))
        if hand_span < 0.05:
            return "None", 0.0

        m = self.normalizer.analyze_hand(lm, handedness)
        ref = m["ref_scale"]

        # ------- Unpack frequently used metrics -------
        idx_ext = m["ext_index"]
        mid_ext = m["ext_middle"]
        rng_ext = m["ext_ring"]
        pnk_ext = m["ext_pinky"]

        idx_up    = m["up_index"]
        mid_up    = m["up_middle"]
        rng_up    = m["up_ring"]
        pnk_up    = m["up_pinky"]

        idx_down  = m["down_index"]
        mid_down  = m["down_middle"]

        idx_horiz = m["horiz_index"]
        mid_horiz = m["horiz_middle"]

        thumb_out = m["thumb_extended_out"]
        h_thumb   = m["h_thumb"]
        t_thumb   = m["t_thumb"]

        d_ti  = m["d_thumb_index_tip"]
        d_tm  = m["d_thumb_middle_tip"]
        d_tr  = m["d_thumb_ring_tip"]
        d_tp  = m["d_thumb_pinky_tip"]
        d_im  = m["d_index_middle_tip"]
        d_mr  = m["d_middle_ring_tip"]
        d_rp  = m["d_ring_pinky_tip"]
        d_tim = m["d_thumb_index_mcp"]
        ref   = m["ref_scale"]

        num_ext = sum([idx_ext, mid_ext, rng_ext, pnk_ext])

        # ==============================================================
        # PRIORITY 0 — Letter O
        # All fingertips converge with thumb in a tight circle/oval.
        # Must be checked FIRST before any other gesture.
        # ==============================================================
        if m["is_O_shape"]:
            return "O", 0.95

        # ==============================================================
        # PRIORITY 1 — Letter X
        # Index PIP joint extends high up, tip hooks back downward.
        # Detected BEFORE Q so a hooked-down index with fist isn't Q.
        # ==============================================================
        if m["index_hooked"]:
            return "X", 0.95

        # ==============================================================
        # PRIORITY 2 — Letter Q
        # Hand horizontal/downward; index points DOWN + thumb alongside.
        # Checked BEFORE the 2-finger group so Q isn't misread as P.
        # ==============================================================
        is_Q = bool(
            (idx_down or (lm[8][1] > lm[5][1] + 0.18 * ref))         # index points down
            and (m["is_thumb_pointing_down"] or (lm[4][1] > lm[2][1] + 0.08 * ref))  # thumb down
            and d_ti < 0.40                                            # thumb near index (pinch)
            and not (idx_up or mid_up or rng_up or pnk_up)           # no fingers pointing up
            and not m["is_O_shape"]
        )
        if is_Q:
            return "Q", 0.95

        # ==============================================================
        # GROUP A — Four extended fingers (B, C)
        # ==============================================================
        if num_ext == 4:
            # C: All 4 fingers curved in C-cup shape, thumb extended wide.
            # Fingers are extended but NOT straight upright (not a flat B).
            if m["is_C_shape"]:
                return "C", 0.92

            # B: All 4 fingers straight UP together, thumb folded across palm.
            # Fingertips should be close together (small inter-tip gaps).
            if not thumb_out and d_im < 0.50 and d_mr < 0.50 and d_rp < 0.50:
                return "B", 0.95

            # Fallback C: if thumb is out and fingers extended
            if thumb_out:
                return "C", 0.90

        # ==============================================================
        # GROUP B — Three extended fingers (W, F)
        # ==============================================================
        if num_ext == 3:
            # W: Index + Middle + Ring all UP and spread apart. Pinky curled.
            if idx_ext and mid_ext and rng_ext and not pnk_ext:
                return "W", 0.94

            # F: Middle + Ring + Pinky UP. Index tip touches thumb (small circle).
            if not idx_ext and mid_ext and rng_ext and pnk_ext:
                if d_ti < 0.42:
                    return "F", 0.95

        # ==============================================================
        # GROUP C — Two extended fingers (P, H, K, R, V, U)
        # Only the index+middle pair is handled here (main ASL 2-finger signs).
        # ==============================================================
        if num_ext == 2 and idx_ext and mid_ext and not rng_ext and not pnk_ext:

            # ---- P: Both index+middle point DOWNWARD ----
            # P is the 'downward peace sign'. Thumb is away from index (not pinching).
            if (idx_down and mid_down) and d_ti > 0.22:
                return "P", 0.95
            if m["is_hand_pointing_down"] and (idx_down or mid_down) and d_ti > 0.22:
                return "P", 0.95

            # ---- H: Both fingers point HORIZONTALLY SIDEWAYS ----
            # H is the horizontal 2-finger sign. Hand axis is lateral.
            if (idx_horiz and mid_horiz) or m["is_hand_horizontal"]:
                return "H", 0.95

            # ---- K: Index+Middle UP, thumb points UP BETWEEN them ----
            # Thumb tip is spatially sandwiched between index and middle MCPs.
            if m["is_thumb_between_index_middle"]:
                return "K", 0.95

            # ---- R: Index+Middle both UP and CROSSED over each other ----
            # The knuckles (MCPs) are apart but the tips have crossed each other's X positions.
            if m["are_index_middle_crossed"]:
                return "R", 0.95

            # ---- V vs U: Both fingers UP (not crossed, not horizontal) ----
            # Key insight: In U, fingers point PARALLEL from their natural knuckle positions.
            # In V, fingers are angled OUTWARD so tips spread FURTHER than the natural MCP gap.
            # Use tip-spread vs knuckle-spread RATIO to distinguish:
            #   V: tips spread > 1.25x the natural index-middle MCP spacing
            #   U: tips stay within ~1.0x the natural knuckle spacing (parallel)
            if idx_up and mid_up:
                d_mcp_im = float(np.linalg.norm(lm[5] - lm[9]))  # natural knuckle gap
                spread_ratio = (d_im * ref) / (d_mcp_im + 1e-5)
                if spread_ratio >= 1.25:
                    return "V", 0.95   # V: tips spread much wider than knuckles
                else:
                    return "U", 0.94   # U: tips stay parallel to knuckle positions

        # ==============================================================
        # GROUP D1 — Single index extended (G, L, D, Z)
        # ==============================================================
        if idx_ext and not mid_ext and not rng_ext and not pnk_ext:

            # ---- G: Index pointing SIDEWAYS, thumb parallel alongside ----
            # G is a horizontal pointing gesture. Hand is sideways.
            if m["thumb_index_parallel"] or (idx_horiz and m["dot_thumb_index"] > 0.22 and not idx_up):
                return "G", 0.95

            # ---- L: Index UP, thumb extended HORIZONTALLY at 90° ----
            # Classic L-shape: index points straight up, thumb points to the side.
            # dot_thumb_index is HIGH in L (thumb and index both point away from palm,
            # just in different directions - but in 2D projection they may align).
            # Use is_thumb_horizontal as the primary discriminator.
            if idx_up and thumb_out and m["is_thumb_horizontal"]:
                return "L", 0.96
            if idx_up and thumb_out:
                return "L", 0.94

            # ---- D: Index UP, middle+ring+pinky loop forward to touch thumb ----
            if idx_up and m["is_D_loop"]:
                return "D", 0.95

            # ---- Z: Index UP alone (tracing Z), no special thumb position ----
            if idx_up and not thumb_out:
                return "Z", 0.92

            # Fallback for index extended in unusual angle
            if idx_up:
                return "Z", 0.88

        # ==============================================================
        # GROUP D2 — Single pinky extended (Y, J, I)
        # ==============================================================
        if pnk_ext and not idx_ext and not mid_ext and not rng_ext:

            # Y: Pinky UP + Thumb extended OUT wide ("hang loose" / shaka sign)
            if thumb_out:
                return "Y", 0.96

            # J: Pinky extended, hand tilted or pinky sweeping sideways (motion sign)
            if m["is_pinky_tilted"] or m["is_hand_tilted"]:
                return "J", 0.94

            # I: Only pinky UP, hand vertical, thumb folded
            return "I", 0.95

        # ==============================================================
        # GROUP E — Zero extended fingers: Fist variants (M, N, T, S, A, E)
        # Discrimination is based on WHERE the thumb is positioned relative to
        # the transverse knuckle axis (t = 0.0 at index MCP → 1.0 at pinky MCP)
        # and the longitudinal height (h) of the thumb tip above the knuckle line.
        # ==============================================================
        if num_ext == 0:

            # ---- M: Thumb tucked UNDER 3 fingers (index+middle+ring drape over it) ----
            # Thumb reaches far across to the ring/pinky side. t >= 0.55.
            if t_thumb >= 0.55:
                return "M", 0.94

            # ---- N: Thumb tucked UNDER 2 fingers (index+middle drape over it) ----
            # Thumb reaches to the middle/ring area. 0.35 <= t < 0.55.
            if 0.35 <= t_thumb < 0.55:
                return "N", 0.93

            # ---- T: Thumb poked UP between index and middle, tip shows above knuckles ----
            # t is low-to-mid (0.08-0.35), thumb points UP.
            if 0.08 <= t_thumb < 0.35 and m["is_thumb_pointing_up"] and h_thumb > -0.05:
                return "T", 0.94

            # ---- A: Thumb rests on SIDE of index, pointing slightly UP ----
            # t < 0.20 (index side), h > 0.02 (thumb extends past index MCP level).
            if t_thumb < 0.20 and h_thumb > 0.02:
                return "A", 0.95

            # ---- S: Thumb wraps HORIZONTALLY ACROSS front of all curled fingers ----
            # Thumb crosses from index side to middle/ring side horizontally.
            # t is mid-range (0.15-0.42), thumb direction is horizontal.
            if 0.12 <= t_thumb <= 0.42 and m["is_thumb_horizontal"] and not m["is_thumb_pointing_up"]:
                return "S", 0.94

            # ---- E: Thumb pressed under all bent fingertips (lowest position) ----
            # Thumb is tucked very low (h <= 0.02, may have t anywhere under fingers).
            if h_thumb <= 0.02 and t_thumb < 0.55:
                return "E", 0.95

            # --- General fist fallback ---
            if h_thumb > 0.02:
                return "A", 0.88
            else:
                return "E", 0.88

        # ==============================================================
        # No matching gesture — return None (never guess randomly)
        # ==============================================================
        return "None", 0.0
