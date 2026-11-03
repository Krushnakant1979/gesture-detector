"""
ASL Gesture Classifier — Complete A-Z Heuristic & Hybrid Ensemble Engine
Based on verified biomechanical analysis of official ASL fingerspelling charts.
Supports Hybrid mode (deterministic structural heuristics + ML cascade fallback).

=== DETECTION PRIORITY ORDER ===
Priority 0 : Dynamic strokes (J and Z stroke trajectories)
Priority 1 : O  (tight finger-tip cluster meeting thumb)
Priority 2 : X  (index PIP elevated, tip hooked back down)
Priority 3 : Q  (index + thumb both pointing DOWN together)
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
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.normalizer import LandmarkNormalizer

logger = logging.getLogger(__name__)


class GestureClassifier:
    """Classifies hand gestures into ASL Alphabet characters (A-Z)."""

    TOPOLOGICAL_SIGNS = {
        "A", "O", "Q", "X", "B", "C", "W", "F", "L", "Y", "V", "U", "H", "P", "K", "R", "M", "N"
    }

    def __init__(
        self,
        mode: str = "hybrid",
        model_path: Optional[str] = None,
        min_confidence: float = 0.65,
        ml_fallback_min_confidence: float = 0.75,
        heuristics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.mode = mode.lower()
        self.min_confidence = min_confidence
        self.ml_fallback_min_confidence = ml_fallback_min_confidence
        self.heuristics: Dict[str, Any] = heuristics or {}
        self.normalizer = LandmarkNormalizer(heuristics=self.heuristics)
        self.ml_model = None

        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.ml_model = pickle.load(f)
                logger.info("Loaded ML model from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load ML model from %s: %s", model_path, e)

    def classify(
        self,
        landmarks_3d: np.ndarray,
        handedness: str = "Right",
        dynamic_gesture: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Classify 21 3D landmarks into an ASL letter.
        Supports dynamic stroke overrides, heuristic rules, and ML cascade fallback.
        """
        # Dynamic gesture priority (e.g. traced 'Z' or 'J' strokes)
        if dynamic_gesture in ("J", "Z"):
            return dynamic_gesture, 0.98

        if self.mode == "ml" and self.ml_model is not None:
            return self._predict_ml(landmarks_3d)

        if self.mode == "hybrid":
            h_letter, h_conf = self.classify_asl(landmarks_3d, handedness)
            # High-confidence distinct topological gestures: trust heuristic
            if h_letter in self.TOPOLOGICAL_SIGNS and h_conf >= 0.90:
                return h_letter, h_conf

            # Ambiguous fist signs or unclassified poses: check ML model
            if self.ml_model is not None:
                ml_letter, ml_conf = self._predict_ml(landmarks_3d)
                if ml_conf >= self.ml_fallback_min_confidence:
                    return ml_letter, ml_conf

            return h_letter, h_conf

        return self.classify_asl(landmarks_3d, handedness)

    def _predict_ml(self, landmarks_3d: np.ndarray) -> Tuple[str, float]:
        """Runs scikit-learn model inference on normalized coordinates."""
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
        """
        wrist = lm[0]
        hand_span = float(np.linalg.norm(lm[9] - wrist))
        min_span = float(self.heuristics.get("min_hand_span", 0.05))
        if hand_span < min_span:
            return "None", 0.0

        m = self.normalizer.analyze_hand(lm, handedness)
        ref = m["ref_scale"]

        # ------- Unpack metrics -------
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
        d_im  = m["d_index_middle_tip"]
        d_mr  = m["d_middle_ring_tip"]
        d_rp  = m["d_ring_pinky_tip"]

        num_ext = sum([idx_ext, mid_ext, rng_ext, pnk_ext])

        # ==============================================================
        # PRIORITY 1 — Letter O
        # ==============================================================
        if m["is_O_shape"]:
            return "O", 0.95

        # ==============================================================
        # PRIORITY 2 — Letter X
        # ==============================================================
        if m["index_hooked"]:
            return "X", 0.95

        # ==============================================================
        # PRIORITY 3 — Letter Q (Downward G-shape)
        # ==============================================================
        if m.get("is_Q_shape"):
            return "Q", 0.96

        # ==============================================================
        # PRIORITY 4 — Letter P (Downward K-shape)
        # ==============================================================
        if m.get("is_P_shape"):
            return "P", 0.96

        # ==============================================================
        # PRIORITY 5 — Letter N (2-finger forward drape)
        # ==============================================================
        if m.get("is_N_shape"):
            return "N", 0.96

        # ==============================================================
        # GROUP A — Four extended fingers (B, C)
        # ==============================================================
        if num_ext == 4:
            # 1. C Shape: Must strictly satisfy is_C_shape (fingers curved in C cup, never flat open palm)
            if m["is_C_shape"]:
                return "C", 0.95

            # 2. B Shape: All 4 fingers straight UP together, thumb tucked across/against palm
            up_all_4 = bool(idx_up and mid_up and rng_up and pnk_up)
            b_im_max = float(self.heuristics.get("b_finger_dist_im", 0.42))
            b_mr_max = float(self.heuristics.get("b_finger_dist_mr", 0.42))
            b_rp_max = float(self.heuristics.get("b_finger_dist_rp", 0.58))
            fingers_close = bool(d_im < b_im_max and d_mr < b_mr_max and d_rp < b_rp_max)

            # Thumb must be folded across or against the palm (strictly rejecting open hand / high-five thumb)
            thumb_tucked = bool(
                not thumb_out
                or (t_thumb >= 0.05 and not m["is_thumb_horizontal"] and d_ti > 0.20)
            )

            if up_all_4 and fingers_close and thumb_tucked:
                return "B", 0.95

            # If fingers are draped down/forward over thumb in a fist formation (M, N)
            if not (idx_up or mid_up or rng_up or pnk_up):
                fist_m = float(self.heuristics.get("fist_m_transverse", 0.52))
                fist_n = float(self.heuristics.get("fist_n_transverse", 0.35))
                if t_thumb >= fist_m and not thumb_out:
                    return "M", 0.95
                if fist_n <= t_thumb < fist_m and not thumb_out:
                    return "N", 0.94

            # 3. Any other 4-finger pose (like flat open palm / 5 with thumb out in Image 1):
            # Strictly return None (never guess C or B!)
            return "None", 0.0

        # ==============================================================
        # GROUP B — Three extended fingers (W, F)
        # ==============================================================
        if num_ext == 3:
            if idx_ext and mid_ext and rng_ext and not pnk_ext:
                if idx_up and mid_up and rng_up:
                    return "W", 0.94

            if not idx_ext and mid_ext and rng_ext and pnk_ext:
                if d_ti < 0.42:
                    return "F", 0.95

            # If fingers are draped down/forward over thumb in a fist formation (M, N)
            if not (idx_up or mid_up or rng_up or pnk_up):
                fist_m = float(self.heuristics.get("fist_m_transverse", 0.52))
                fist_n = float(self.heuristics.get("fist_n_transverse", 0.35))
                if t_thumb >= fist_m and not thumb_out:
                    return "M", 0.95
                if fist_n <= t_thumb < fist_m and not thumb_out:
                    return "N", 0.94

        # ==============================================================
        # GROUP C — Two extended fingers (P, H, K, R, V, U)
        # ==============================================================
        if num_ext == 2 and idx_ext and mid_ext and not rng_ext and not pnk_ext:
            # P vs N (fingers pointing downward):
            # N: Index + Middle parallel and together (2-finger forward drape)
            # P: Index + Middle spread apart like downward K (spread_ratio >= 1.20)
            is_downward_pair = bool(
                (idx_down and mid_down)
                or (m["is_hand_pointing_down"] and (idx_down or mid_down))
            )
            if is_downward_pair:
                d_mcp_im = float(np.linalg.norm(lm[5] - lm[9]))
                spread_ratio = (d_im * ref) / (d_mcp_im + 1e-5)
                # N: fingers parallel and together — widened zone (< 1.60) to catch gestures
                # that look like N but have a slightly wider spread than ideal.
                # Thumb must NOT be pinched to index (Q has d_ti < 0.40).
                if (spread_ratio < 1.60 or d_im < 0.35) and d_ti > 0.35:
                    return "N", 0.95
                # P: fingers spread clearly like scissors — require strict spread (>= 1.80)
                elif spread_ratio >= 1.80 and d_ti > 0.22:
                    return "P", 0.95
                # Gray zone (1.60-1.80): ambiguous, show nothing
                return "None", 0.0

            # H: Both fingers point HORIZONTALLY SIDEWAYS
            if (idx_horiz and mid_horiz) or m["is_hand_horizontal"]:
                return "H", 0.95

            # R: Index+Middle both UP and CROSSED — check BEFORE K.
            # When fingers cross, the thumb may appear inside the corridor (→ K)
            # but crossed geometry always wins: it's R, not K.
            if m["are_index_middle_crossed"]:
                return "R", 0.95

            # K: Index+Middle UP, thumb points UP BETWEEN them (only when NOT crossed)
            if m["is_thumb_between_index_middle"] or (m.get("is_thumb_elevated") and m.get("is_thumb_in_finger_corridor")):
                return "K", 0.95

            # V vs U: Both fingers UP
            if idx_up and mid_up:
                d_mcp_im = float(np.linalg.norm(lm[5] - lm[9]))
                spread_ratio = (d_im * ref) / (d_mcp_im + 1e-5)
                spread_thresh = float(self.heuristics.get("u_v_spread_ratio", 1.25))
                if spread_ratio >= spread_thresh:
                    return "V", 0.95
                else:
                    return "U", 0.94

        # ==============================================================
        # GROUP D1 — Single index extended (G, L, D, Z)
        # ==============================================================
        if idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            # 1. D: Index UP, middle loops to touch thumb (prioritize loop over L)
            if idx_up and m["is_D_loop"]:
                return "D", 0.95

            # 2. G: Index pointing SIDEWAYS, thumb parallel
            if m["thumb_index_parallel"] or (idx_horiz and m["dot_thumb_index"] > 0.22 and not idx_up):
                return "G", 0.95

            # 3. L: STRICT ASL L GESTURE
            # Index points straight UP, thumb extended horizontally OUT forming an 'L'
            if m["is_L_shape"]:
                return "L", 0.96

            # 4. Z / Index UP with tucked thumb
            if idx_up and not thumb_out and d_ti < 0.45:
                return "Z", 0.90

            # Reject any ambiguous or unclassified index gesture
            return "None", 0.0

        # ==============================================================
        # GROUP D2 — Single pinky extended (Y, J, I)
        # ==============================================================
        if pnk_ext and not idx_ext and not mid_ext and not rng_ext:
            if thumb_out:
                return "Y", 0.96

            # J is triggered dynamically by StrokeTrajectoryTracker; static is I
            if m["is_pinky_tilted"] or m["is_hand_tilted"]:
                return "J", 0.92

            return "I", 0.95

        # ==============================================================
        # GROUP E — Zero extended fingers: Fist variants (M, N, T, S, A, E)
        # ==============================================================
        if num_ext == 0 or (not (idx_up or mid_up or rng_up or pnk_up) and not m["is_C_shape"]):
            fist_m = float(self.heuristics.get("fist_m_transverse", 0.52))
            fist_n = float(self.heuristics.get("fist_n_transverse", 0.35))
            fist_e = float(self.heuristics.get("fist_e_thumb_height", 0.02))

            # M: Thumb tucked under 3 fingers
            if t_thumb >= fist_m and not thumb_out:
                return "M", 0.95

            # N: Thumb tucked under 2 fingers
            if fist_n <= t_thumb < fist_m and not thumb_out:
                return "N", 0.94

            # T: Thumb poked up between index and middle
            if 0.10 <= t_thumb < fist_n and m["is_thumb_pointing_up"] and h_thumb > 0.04 and abs(t_thumb - 0.22) < 0.12:
                return "T", 0.94

            # A: Thumb rests straight upright on the SIDE of the index finger
            # STRICT CRITERIA:
            # - Thumb MUST point UP: is_thumb_pointing_up is True
            # - Thumb MUST NOT point horizontally: not is_thumb_horizontal
            # - Thumb MUST NOT be extended out wide: not thumb_out (d_tim <= 0.40)
            # - Thumb MUST be snug alongside index finger: d_tim <= 0.40 and t_thumb <= 0.22
            # - Thumb tip is elevated along index knuckle: h_thumb > 0.04
            is_A = bool(
                m["is_thumb_pointing_up"]
                and not m["is_thumb_horizontal"]
                and not thumb_out
                and m["d_thumb_index_mcp"] <= 0.40
                and t_thumb <= 0.22
                and h_thumb > 0.04
            )
            if is_A:
                return "A", 0.96

            # S: Thumb wraps horizontally across front
            if 0.12 <= t_thumb <= 0.50 and m["is_thumb_horizontal"] and not m["is_thumb_pointing_up"]:
                return "S", 0.94

            # E: Thumb pressed under fingertips
            if h_thumb <= fist_e and t_thumb < fist_m and not thumb_out:
                return "E", 0.95

            # If no defined fist gesture matches strictly, return None (never guess randomly)
            return "None", 0.0

        return "None", 0.0
