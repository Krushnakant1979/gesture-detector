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

from src.lstm_recognizer import LSTMRecurrentClassifier
from src.normalizer import LandmarkNormalizer

logger = logging.getLogger(__name__)


class GestureClassifier:
    """Classifies hand gestures into ASL Alphabet characters (A-Z)."""

    TOPOLOGICAL_SIGNS = {
        "B", "C", "D", "E", "F", "G", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
    }

    def __init__(
        self,
        mode: str = "hybrid",
        model_path: Optional[str] = None,
        min_confidence: float = 0.92,
        ml_fallback_min_confidence: float = 0.92,
        heuristics: Optional[Dict[str, Any]] = None,
        enable_lstm: bool = True,
        lstm_model_path: Optional[str] = "models/lstm_gesture_model.npz",
    ) -> None:
        self.mode = mode.lower()
        self.min_confidence = float(min_confidence)
        self.ml_fallback_min_confidence = float(ml_fallback_min_confidence)
        self.heuristics: Dict[str, Any] = heuristics or {}
        self.normalizer = LandmarkNormalizer(heuristics=self.heuristics)
        self.ml_model = None

        # Preferred models: LightGBM -> Random Forest
        candidate_paths = []
        if model_path:
            candidate_paths.append(model_path)
        candidate_paths.extend(["models/lightgbm_classifier.pkl", "models/gesture_classifier.pkl"])

        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        self.ml_model = pickle.load(f)
                    model_type = "LightGBM" if "lightgbm" in path.lower() else "Random Forest"
                    logger.info("Loaded %s model from %s", model_type, path)
                    break
                except Exception as e:
                    logger.warning("Failed to load model from %s: %s", path, e)

        # LSTMR Recurrent Gesture Sequence Recognizer for dynamic motions ('J', 'Z')
        self.enable_lstm = enable_lstm
        self.lstm_recognizer = None
        if self.enable_lstm:
            self.lstm_recognizer = LSTMRecurrentClassifier(
                model_path=lstm_model_path if (lstm_model_path and os.path.exists(lstm_model_path)) else None
            )

    def classify(
        self,
        landmarks_3d: np.ndarray,
        handedness: str = "Right",
        dynamic_gesture: Optional[str] = None,
        precomputed_m: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float]:
        """
        Classify 21 3D landmarks into an ASL letter.
        Supports dynamic stroke overrides, LSTMR sequence recognition, heuristic rules, and ML cascade fallback.
        Accepts precomputed anatomical metrics dictionary to eliminate redundant calculations.
        """
        # Reuse or calculate anatomical features once
        m = precomputed_m if precomputed_m is not None else self.normalizer.analyze_hand(landmarks_3d, handedness)

        # Dynamic gesture priority (e.g. traced 'Z' or 'J' strokes)
        if dynamic_gesture == "Z":
            return "Z", 0.98
        elif dynamic_gesture == "J":
            # Guard: J strictly requires pinky movement (never G shape or extended index)
            if not (m.get("is_G_shape") or m.get("ext_index") or m.get("horiz_index")):
                return "J", 0.98

        # LSTMR Recurrent sequence evaluation for dynamic strokes
        if self.lstm_recognizer is not None:
            if "landmarks_canonical" in m:
                c_feat = m["landmarks_canonical"].flatten().astype(np.float32)
            else:
                c_feat = self.normalizer.normalize(landmarks_3d, canonical=True, handedness=handedness)
            lstm_res = self.lstm_recognizer.update(c_feat, raw_landmarks=landmarks_3d)
            if lstm_res:
                res_lbl, res_conf = lstm_res
                # Guard against false dynamic J when signing G
                if res_lbl == "J" and (m.get("is_G_shape") or m.get("ext_index") or m.get("horiz_index")):
                    pass
                elif res_lbl == "Z" and not (m.get("ext_index") and not m.get("ext_middle")):
                    pass
                else:
                    return lstm_res

        # Reject non-ASL gestures (such as flat open palm / high five / random spread hand)
        if m.get("is_flat_open_hand"):
            return "None", 0.0

        if self.mode == "ml" and self.ml_model is not None:
            return self._predict_ml(landmarks_3d, handedness=handedness, m=m)

        if self.mode == "hybrid":
            h_letter, h_conf = self.classify_asl(landmarks_3d, handedness, m=m)
            # High-confidence distinct topological gestures: trust heuristic
            if h_letter in self.TOPOLOGICAL_SIGNS and h_conf >= self.min_confidence:
                if self.validate_sign_anatomy(h_letter, m, lm=landmarks_3d, handedness=handedness):
                    return h_letter, h_conf

            # All other gestures (G, D, fist variants, V, etc.): route to LightGBM
            if self.ml_model is not None:
                ml_letter, ml_conf = self._predict_ml(landmarks_3d, handedness=handedness, m=m)
                if ml_conf >= self.min_confidence:
                    if self.validate_sign_anatomy(ml_letter, m, lm=landmarks_3d, handedness=handedness):
                        return ml_letter, ml_conf

            if h_conf >= self.min_confidence and h_letter != "None":
                if self.validate_sign_anatomy(h_letter, m, lm=landmarks_3d, handedness=handedness):
                    return h_letter, h_conf

            return "None", 0.0

        h_letter, h_conf = self.classify_asl(landmarks_3d, handedness, m=m)
        if h_conf >= self.min_confidence and h_letter != "None":
            if self.validate_sign_anatomy(h_letter, m, lm=landmarks_3d, handedness=handedness):
                return h_letter, h_conf
        return "None", 0.0

    def reset_recurrent_state(self) -> None:
        """Resets LSTMR sequence memory buffer."""
        if self.lstm_recognizer:
            self.lstm_recognizer.reset()

    def _predict_ml(
        self,
        landmarks_3d: np.ndarray,
        handedness: str = "Right",
        m: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float]:
        """Runs model inference on 3D canonical normalized coordinates."""
        try:
            if m is not None and "landmarks_canonical" in m:
                features = m["landmarks_canonical"].flatten().reshape(1, -1)
            else:
                features = self.normalizer.normalize(
                    landmarks_3d, canonical=True, handedness=handedness
                ).reshape(1, -1)

            m_eval = m if m is not None else self.normalizer.analyze_hand(landmarks_3d, handedness)

            if hasattr(self.ml_model, "predict_proba"):
                probs = self.ml_model.predict_proba(features)[0]
                best_idx = int(np.argmax(probs))
                conf = float(probs[best_idx])
                best_class = str(self.ml_model.classes_[best_idx])
                sorted_probs = np.sort(probs)
                margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else 1.0

                # Anatomical consistency guard: Verify predicted letter meets ASL biomechanical rules
                if not self.validate_sign_anatomy(best_class, m_eval, lm=landmarks_3d, handedness=handedness):
                    return "None", 0.0

                if conf >= self.min_confidence and margin >= 0.08:
                    return best_class, conf
            else:
                pred_letter = str(self.ml_model.predict(features)[0])
                if self.validate_sign_anatomy(pred_letter, m_eval, lm=landmarks_3d, handedness=handedness):
                    return pred_letter, 0.85
        except Exception as e:
            logger.error("ML inference error: %s", e)
        return "None", 0.0

    def validate_sign_anatomy(
        self,
        best_class: str,
        m: Dict[str, Any],
        lm: Optional[np.ndarray] = None,
        handedness: str = "Right",
    ) -> bool:
        """
        Strict anatomical verification for predicted ASL alphabets.
        Rejects random, incorrect, or ambiguous hand poses to eliminate false-positive predictions.
        Supports cached 3D canonical alignment fallback for rotated genuine hand poses.
        """
        # ASL I requires an upright hand in camera/physical space; never accept a tilted/horizontal/J hand
        if best_class == "I":
            if m.get("is_J_shape") or m.get("horiz_pinky") or m.get("is_hand_horizontal") or m.get("dx_pnk", 0.0) > 0.65 * m.get("dy_pnk", 1.0):
                return False
            return self._check_anatomy("I", m)

        # ASL J requires pinky pointing sideways/horizontal or tilted in physical space; never canonical upright
        if best_class == "J":
            return self._check_anatomy("J", m)

        # ASL S and T require strict thumb placement in physical space
        if best_class in {"S", "T"}:
            return self._check_anatomy(best_class, m)

        if self._check_anatomy(best_class, m):
            return True

        # Check canonical alignment fallback if initial check failed
        if "_m_can_cache" not in m:
            c_lm = m.get("landmarks_canonical")
            if c_lm is None and lm is not None and lm.shape == (21, 3):
                c_lm, _ = self.normalizer.canonical_align(lm, handedness=handedness)
            if c_lm is not None:
                m["_m_can_cache"] = self.normalizer.analyze_hand(c_lm, handedness=handedness)

        if "_m_can_cache" in m and self._check_anatomy(best_class, m["_m_can_cache"]):
            return True

        return False

    def _check_anatomy(self, best_class: str, m: Dict[str, Any]) -> bool:
        if best_class in ("None", "-", "Unknown", "Unknown Hand Gesture", ""):
            return False

        if m.get("is_flat_open_hand", False):
            return False

        idx_ext = m.get("ext_index", False)
        mid_ext = m.get("ext_middle", False)
        rng_ext = m.get("ext_ring", False)
        pnk_ext = m.get("ext_pinky", False)

        idx_up = m.get("up_index", False)
        mid_up = m.get("up_middle", False)
        rng_up = m.get("up_ring", False)
        pnk_up = m.get("up_pinky", False)

        idx_down = m.get("down_index", False)
        mid_down = m.get("down_middle", False)
        rng_down = m.get("down_ring", False)
        pnk_down = m.get("down_pinky", False)

        idx_horiz = m.get("horiz_index", False)
        mid_horiz = m.get("horiz_middle", False)
        rng_horiz = m.get("horiz_ring", False)
        pnk_horiz = m.get("horiz_pinky", False)

        thumb_out = m.get("thumb_extended_out", False)
        t_thumb = m.get("t_thumb", 0.0)
        h_thumb = m.get("h_thumb", 0.0)
        d_ti = m.get("d_thumb_index_tip", 1.0)
        d_im = m.get("d_index_middle_tip", 1.0)
        d_mr = m.get("d_middle_ring_tip", 1.0)
        d_rp = m.get("d_ring_pinky_tip", 1.0)
        d_mcp = m.get("d_thumb_index_mcp", 1.0)

        up_all_4 = bool(idx_up and mid_up and rng_up and pnk_up)
        ext_all_4 = bool(idx_ext and mid_ext and rng_ext and pnk_ext)

        # 1. Closed Fist / Forward Drape Signs (A, E, M, N, S, T): 0 extended fingers pointing UP
        if best_class in {"A", "E", "M", "N", "S", "T"}:
            if idx_up or mid_up or rng_up or pnk_up:
                return False
            if best_class in {"A", "E", "S", "T"}:
                if (idx_ext and not idx_down) or (mid_ext and not mid_down) or (rng_ext and not rng_down) or (pnk_ext and not pnk_down):
                    return False
            if m.get("is_C_shape") or m.get("is_O_shape") or (thumb_out and best_class not in {"N", "M", "A"}):
                return False

            if best_class == "S":
                # Thumb MUST wrap horizontally across front, NEVER poke UP as in T, NEVER on lateral side as in A
                if m.get("is_T_shape") or m.get("is_thumb_distal_pointing_up", False):
                    return False
                if not (m.get("is_S_shape") or (0.14 <= t_thumb <= 0.68 and m.get("is_thumb_distal_horizontal", False))):
                    return False
                if t_thumb < 0.14:
                    return False
                return True
            elif best_class == "T":
                # Thumb MUST poke UP between index and middle, NEVER wrap horizontally across as in S
                if m.get("is_S_shape") or (m.get("is_thumb_distal_horizontal", False) and not m.get("is_thumb_distal_pointing_up", False)):
                    return False
                if not (m.get("is_T_shape") or (0.10 <= t_thumb <= 0.35 and m.get("is_thumb_distal_pointing_up", False))):
                    return False
                return True
            elif best_class == "A":
                # Thumb rests upright on the lateral/radial side of the index finger
                if not m.get("is_knuckles_horizontal", True):
                    return False
                if t_thumb > 0.22 or m.get("is_S_shape") or m.get("is_T_shape") or m.get("is_M_shape") or m.get("is_E_shape"):
                    return False
                if not m.get("is_thumb_pointing_up", False) or m.get("is_thumb_pointing_down", False) or m.get("is_thumb_horizontal", False):
                    return False
                if h_thumb < -0.05:
                    return False
                return True
            elif best_class == "E":
                if m.get("is_S_shape") or m.get("is_T_shape") or m.get("is_M_shape") or m.get("is_N_shape"):
                    return False
                if thumb_out:
                    return False
                if m.get("h_thumb", 0.0) > -0.05:
                    return False
                return True
            elif best_class == "M":
                if m.get("is_S_shape") or m.get("is_T_shape"):
                    return False
                if m.get("is_M_shape"):
                    return True
                if pnk_ext and m.get("d_ring_pinky_tip", 1.0) > 0.45:
                    return False
                if ext_all_4:
                    return False
                if idx_down and mid_down and rng_down and not pnk_down and d_im < 0.58 and d_mr < 0.55:
                    return True
                if not (idx_ext or mid_ext or rng_ext or pnk_ext) and abs(t_thumb) >= 0.65 and h_thumb > -0.32:
                    return True
                return False
            elif best_class == "N":
                if m.get("is_S_shape") or m.get("is_T_shape") or m.get("is_P_shape"):
                    return False
                if m.get("is_N_shape"):
                    return True
                if pnk_ext and m.get("d_ring_pinky_tip", 1.0) > 0.45:
                    return False
                if rng_ext and m.get("d_middle_ring_tip", 1.0) > 0.45:
                    return False
                if ext_all_4:
                    return False
                if idx_down and mid_down and not rng_down and not pnk_down and d_im < 0.50:
                    return True
                if not (idx_ext or mid_ext or rng_ext or pnk_ext) and 0.25 <= abs(t_thumb) < 0.65 and h_thumb > -0.32:
                    return True
                return False
            return True

        # 2. Four-finger paddle (B)
        if best_class == "B":
            if not (up_all_4 and ext_all_4):
                return False
            if d_im > 0.48 or d_mr > 0.48 or d_rp > 0.62:
                return False
            if m.get("is_C_shape") or m.get("is_O_shape") or m.get("are_fingers_arched"):
                return False
            # Thumb MUST be folded across or against the palm (NEVER sticking out)
            if thumb_out or d_mcp > 0.55 or d_ti > 1.25 or t_thumb <= -0.20:
                return False
            return True

        # 3. C Shape: Curved fingers and thumb opposition
        if best_class == "C":
            if not m.get("is_C_shape") or m.get("is_O_shape"):
                return False
            if up_all_4 and m.get("straight_fingers", False):
                return False
            if idx_down and mid_down and rng_down and pnk_down and not idx_horiz:
                return False
            return True

        # 4. One-finger signs (D, G, I, L, X)
        if best_class == "D":
            if not (idx_up and idx_ext):
                return False
            if mid_up or rng_up or pnk_up:
                return False
            if not m.get("is_D_loop") or d_ti <= 0.55 or m.get("is_L_shape"):
                return False
            if m.get("d_thumb_middle_tip", 1.0) > 0.20:
                return False
            if m.get("idx_tip_elevation", 1.0) < 0.35:
                return False
            if m.get("d_index_tip_mcp", 1.0) < 0.58:
                return False
            return True

        if best_class == "G":
            # Remaining fingers must not point UP (ignore curled pinky inside horizontal fist)
            real_pnk_up = bool(pnk_up and m.get("straight_pnk", 0.5) > 0.80 and not m.get("is_hand_horizontal"))
            if mid_up or rng_up or real_pnk_up:
                return False
            # Middle, ring, pinky must not be extended out horizontally (like H)
            if mid_ext and (mid_horiz or d_im < 0.35):
                return False
            # Not L, Q, or downward pointing
            if m.get("is_L_shape") or m.get("is_Q_shape") or m.get("is_hand_pointing_down") or (m.get("down_index") and not m.get("horiz_index") and not m.get("is_hand_horizontal")):
                return False
            if not idx_ext:
                return False
            # Must point sideways/horizontally or hand horizontal or thumb parallel
            dot = m.get("dot_thumb_index", 0.0)
            is_horiz_g = bool(
                m.get("is_G_shape")
                or idx_horiz
                or m.get("is_hand_horizontal")
                or m.get("thumb_index_parallel")
                or (dot > 0.20 and not idx_up)
                or (idx_horiz and dot > 0.15)
                or (m.get("is_hand_horizontal") and dot > 0.15)
            )
            if not is_horiz_g:
                return False
            # Parallel pinching gap
            if d_ti > 0.90:
                return False
            return True

        if best_class == "I":
            if not (pnk_up and pnk_ext):
                return False
            # ASL I requires straight upright pinky on an upright hand (never horizontal, G gesture, or J)
            if m.get("straight_pnk", 0.5) < 0.82 or m.get("is_hand_horizontal") or m.get("is_G_shape") or m.get("is_J_shape") or m.get("horiz_pinky"):
                return False
            if idx_ext or mid_ext or rng_ext or idx_horiz:
                return False
            if thumb_out:
                return False
            dx_pnk = m.get("dx_pnk", 0.0)
            dy_pnk = m.get("dy_pnk", 1.0)
            if dx_pnk > 0.70 * dy_pnk:
                return False
            return True

        if best_class == "J":
            if not pnk_ext:
                return False
            if m.get("straight_pnk", 0.5) < 0.80:
                return False
            if idx_ext or mid_ext or rng_ext or idx_up or mid_up or rng_up or m.get("is_G_shape"):
                return False
            if thumb_out:
                return False
            # ASL J handshape: pinky pointing sideways/horizontal or tilted, or dynamic J
            dx_pnk = m.get("dx_pnk", 1.0)
            dy_pnk = m.get("dy_pnk", 1.0)
            is_sideways = bool(
                m.get("horiz_pinky")
                or m.get("is_hand_horizontal")
                or m.get("is_hand_tilted")
                or m.get("is_pinky_tilted")
                or (dx_pnk > 0.65 * dy_pnk)
            )
            if not is_sideways:
                return False
            return True

        if best_class == "L":
            if not (idx_up and idx_ext):
                return False
            if mid_ext or rng_ext or pnk_ext:
                return False
            if not (thumb_out or m.get("is_thumb_horizontal") or d_mcp > 0.45):
                return False
            if d_ti <= 0.50:
                return False
            if m.get("dot_thumb_index", 0.0) > 0.45:
                return False
            return True

        if best_class == "X":
            if mid_ext or rng_ext or pnk_ext:
                return False
            return True

        # 5. Letter P: Downward K-shape
        if best_class == "P":
            return bool(
                m.get("is_P_shape")
                or (
                    idx_ext
                    and (mid_down or m.get("downward_mid_slope", 0.0) >= 0.35)
                    and not (rng_up or pnk_up)
                    and m.get("p_spread_ratio", 0.0) >= 1.80
                    and d_im >= 0.40
                )
            )

        # 6. Two-finger signs (H, K, R, U, V)
        if best_class in {"H", "K", "R", "U", "V"}:
            if not (idx_ext and mid_ext):
                return False
            if rng_up or pnk_up:
                return False
            if best_class == "H":
                if idx_up and mid_up and not m.get("is_hand_horizontal"):
                    return False
            elif best_class == "K":
                if not m.get("is_thumb_between_index_middle"):
                    return False
            elif best_class == "R":
                if not (idx_up and mid_up):
                    return False
            elif best_class == "U":
                if not (idx_up and mid_up):
                    return False
                if d_im >= 0.42:
                    return False
            elif best_class == "V":
                if not (idx_up and mid_up):
                    return False
                if d_im < 0.28:
                    return False
            return True

        # 6. Three-finger signs (W, F)
        if best_class == "W":
            if not (idx_up and mid_up and rng_up):
                return False
            if pnk_ext or pnk_up:
                return False
            return True

        if best_class == "F":
            if not (mid_up and rng_up and pnk_up and mid_ext and rng_ext and pnk_ext):
                return False
            if d_ti > 0.45 or m.get("is_O_shape"):
                return False
            return True

        # 7. Thumb + Pinky sign (Y)
        if best_class == "Y":
            if not pnk_ext:
                return False
            if idx_ext or mid_ext or rng_ext or idx_up or mid_up or rng_up:
                return False
            if not (thumb_out or d_mcp > 0.45):
                return False
            return True

        # 8. O sign
        if best_class == "O":
            return bool(m.get("is_O_shape"))

        # 9. Q sign: Downward G-shape (Index + Thumb pincher pointing downward)
        if best_class == "Q":
            return bool(m.get("is_Q_shape") or ((m.get("downward_idx_slope", 0.0) >= 0.35 or (idx_down and not m.get("horiz_index"))) and d_ti <= 0.70 and not (mid_up or rng_up or pnk_up)))

        return True

    def classify_asl(
        self,
        lm: np.ndarray,
        handedness: str = "Right",
        m: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float]:
        """
        Comprehensive biomechanical classification of all 26 ASL letters.
        Reuses precomputed metrics dictionary m if provided to eliminate redundant calculations.
        """
        wrist = lm[0]
        hand_span = float(np.linalg.norm(lm[9] - wrist))
        min_span = float(self.heuristics.get("min_hand_span", 0.05))
        if hand_span < min_span:
            return "None", 0.0

        if m is None:
            m = self.normalizer.analyze_hand(lm, handedness)
        if m.get("is_flat_open_hand"):
            return "None", 0.0

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

        idx_down  = m.get("down_index", False)
        mid_down  = m.get("down_middle", False)
        rng_down  = m.get("down_ring", False)
        pnk_down  = m.get("down_pinky", False)

        idx_horiz = m.get("horiz_index", False)
        mid_horiz = m.get("horiz_middle", False)
        rng_horiz = m.get("horiz_ring", False)
        pnk_horiz = m.get("horiz_pinky", False)

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
        # PRIORITY 3a — Letter M (3-finger forward drape)
        # ==============================================================
        if m.get("is_M_shape"):
            return "M", 0.96

        # ==============================================================
        # PRIORITY 3b — Letter N (2-finger forward drape)
        # ==============================================================
        if m.get("is_N_shape"):
            return "N", 0.96

        # ==============================================================
        # PRIORITY 4 — Letter Q (Downward G-shape)
        # ==============================================================
        if m.get("is_Q_shape"):
            return "Q", 0.96

        # ==============================================================
        # PRIORITY 5 — Letter P (Downward K-shape)
        # ==============================================================
        if m.get("is_P_shape"):
            return "P", 0.96

        # ==============================================================
        # PRIORITY 6 — Letter C (Arched C-cup formation)
        # Checked with strict geometric guards rejecting open palms
        # ==============================================================
        if m.get("is_C_shape"):
            return "C", 0.96

        # ==============================================================
        # PRIORITY 7 — Letter G (Index sideways, thumb parallel)
        # ==============================================================
        if m.get("is_G_shape"):
            return "G", 0.96

        # ==============================================================
        # GROUP A — Four extended fingers (B, C)
        # ==============================================================
        if num_ext == 4:
            # 1. C Shape: Must strictly satisfy is_C_shape (fingers curved in C cup, never flat open palm)
            if m["is_C_shape"]:
                return "C", 0.95

            # 2. B Shape: All 4 fingers straight UP together, thumb tucked across/against palm
            up_all_4 = bool(idx_up and mid_up and rng_up and pnk_up)
            ext_all_4 = bool(idx_ext and mid_ext and rng_ext and pnk_ext)
            b_im_max = float(self.heuristics.get("b_finger_dist_im", 0.45))
            b_mr_max = float(self.heuristics.get("b_finger_dist_mr", 0.45))
            b_rp_max = float(self.heuristics.get("b_finger_dist_rp", 0.60))
            fingers_close = bool(d_im < b_im_max and d_mr < b_mr_max and d_rp < b_rp_max)

            # Thumb must be folded across or against the palm (strictly rejecting open hand / random spread gestures)
            thumb_tucked = bool(
                not thumb_out
                and m.get("d_thumb_index_mcp", 1.0) <= 0.55
                and d_ti <= 1.25
                and not (m.get("is_thumb_horizontal") and m.get("d_thumb_index_mcp", 1.0) > 0.48)
                and t_thumb > -0.20
            )

            if up_all_4 and ext_all_4 and fingers_close and thumb_tucked and not m["is_C_shape"] and not m["is_O_shape"] and not m.get("are_fingers_arched"):
                return "B", 0.95

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

            return "None", 0.0

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
                or (m.get("downward_mid_slope", 0.0) >= 0.35 and (idx_down or idx_horiz))
            )
            if is_downward_pair:
                d_mcp_im = float(np.linalg.norm(lm[5] - lm[9]))
                spread_ratio = (d_im * ref) / (d_mcp_im + 1e-5)
                # N: fingers together, thumb NOT pinched to index (Q has d_ti < 0.40)
                if (spread_ratio < 1.20 or d_im < 0.30) and d_ti > 0.35:
                    return "N", 0.95
                elif m.get("is_P_shape") or (spread_ratio >= 1.80 and d_im >= 0.40 and not m.get("is_Q_shape")):
                    return "P", 0.96

            # H: Both fingers point HORIZONTALLY SIDEWAYS
            if (idx_horiz and mid_horiz) or m["is_hand_horizontal"]:
                return "H", 0.95

            # K: Index+Middle UP, thumb points UP BETWEEN them
            if m["is_thumb_between_index_middle"] or (m.get("is_thumb_elevated") and m.get("is_thumb_in_finger_corridor")):
                return "K", 0.95

            # R: Index+Middle both UP and CROSSED
            if m["are_index_middle_crossed"]:
                return "R", 0.95

            # V vs U: Both fingers UP
            if idx_up and mid_up:
                d_mcp_im = float(np.linalg.norm(lm[5] - lm[9]))
                spread_ratio = (d_im * ref) / (d_mcp_im + 1e-5)
                spread_thresh = float(self.heuristics.get("u_v_spread_ratio", 1.25))
                if spread_ratio >= spread_thresh and d_im >= 0.25:
                    return "V", 0.95
                else:
                    return "U", 0.94

        # ==============================================================
        # GROUP D1 — Single index extended (G, L, D, Q)
        # ==============================================================
        if idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            # 1. D: Index straight UP, middle loops to touch thumb (D-loop), thumb far from index tip (never G)
            if idx_up and m["is_D_loop"] and d_ti > 0.60 and not m["is_L_shape"]:
                return "D", 0.95

            # 2. Q: Index pointing DOWNWARD, thumb pincher parallel
            if m.get("is_Q_shape") or ((m.get("downward_idx_slope", 0.0) >= 0.35 or (idx_down and not m.get("horiz_index"))) and d_ti <= 0.70 and not idx_up and not m.get("is_P_shape")):
                return "Q", 0.96

            # 3. G: Index pointing SIDEWAYS, thumb parallel
            if m.get("is_G_shape") or m.get("thumb_index_parallel") or (idx_horiz and m.get("dot_thumb_index", 0.0) > 0.15 and not idx_up) or (m.get("is_hand_horizontal") and d_ti <= 0.90 and not idx_up):
                return "G", 0.95

            # 3. L: STRICT ASL L GESTURE
            # Index points straight UP, thumb extended horizontally OUT forming an 'L'
            if m["is_L_shape"]:
                return "L", 0.96

            # Reject any ambiguous or unclassified index gesture (never guess dynamic Z statically)
            return "None", 0.0

        # ==============================================================
        # GROUP D2 — Single pinky extended (Y, J, I)
        # ==============================================================
        if pnk_ext and not idx_ext and not mid_ext and not rng_ext:
            if thumb_out:
                return "Y", 0.96

            # J handshape: Pinky extended sideways/horizontally or tilted in J pose
            if m.get("is_J_shape"):
                return "J", 0.96

            # ASL I requires an upright hand with pinky pointing straight UP (never horizontal or J gesture)
            dx_pnk = abs(lm[20][0] - lm[17][0])
            dy_pnk = abs(lm[20][1] - lm[17][1])
            is_vertical_pinky = bool(pnk_up and not m.get("horiz_pinky") and dy_pnk > 1.15 * dx_pnk)
            if not m.get("is_hand_horizontal") and not m.get("is_G_shape") and not m.get("horiz_index") and is_vertical_pinky:
                return "I", 0.95

            return "None", 0.0

        # ==============================================================
        # GROUP E — Zero extended fingers: Fist variants (M, N, T, S, A, E)
        # ==============================================================
        if num_ext == 0 or (not (idx_up or mid_up or rng_up or pnk_up) and not m["is_C_shape"]):
            fist_m = float(self.heuristics.get("fist_m_transverse", 0.52))
            fist_n = float(self.heuristics.get("fist_n_transverse", 0.35))
            fist_e = float(self.heuristics.get("fist_e_thumb_height", 0.02))

            # S: Thumb wraps horizontally across front (User S shape or curled fist with horizontal distal thumb across index/middle)
            if m.get("is_S_shape") or (0.14 <= t_thumb <= 0.42 and m.get("is_thumb_distal_horizontal", False) and not m.get("is_thumb_distal_pointing_up", False) and h_thumb > -0.32):
                return "S", 0.95

            # T: Thumb poked up between index and middle
            if m.get("is_T_shape") or (0.10 <= t_thumb < fist_n and m.get("is_thumb_distal_pointing_up", False) and not m.get("is_thumb_distal_horizontal", False)):
                return "T", 0.94

            # M: 3 draped fingers or explicit M shape
            if m.get("is_M_shape") or (
                m.get("down_index") and m.get("down_middle") and m.get("down_ring")
                and not m.get("down_pinky")
                and not (m.get("ext_pinky") and m.get("d_ring_pinky_tip", 1.0) > 0.45)
                and m.get("d_index_middle_tip", 1.0) < 0.58
                and m.get("d_middle_ring_tip", 1.0) < 0.55
                and abs(t_thumb) >= 0.45
            ):
                return "M", 0.95

            # E: Thumb pressed low under curled fingertips (real-world ASL E or synthetic E)
            if m.get("is_E_shape") or (
                ((h_thumb <= -0.08 and t_thumb <= 0.20) or (h_thumb <= -0.32 and t_thumb >= 0.50))
                and not thumb_out
                and not m.get("is_S_shape")
                and not m.get("is_T_shape")
                and not m.get("is_M_shape")
                and not m.get("is_N_shape")
                and m.get("straight_idx", 1.0) <= 0.35
            ):
                return "E", 0.96

            # N: 2 draped fingers or explicit N shape
            if m.get("is_N_shape") or (
                m.get("down_index") and m.get("down_middle")
                and not m.get("down_ring") and not m.get("down_pinky")
                and not (m.get("ext_ring") and m.get("d_middle_ring_tip", 1.0) > 0.45)
                and not (m.get("ext_pinky") and m.get("d_ring_pinky_tip", 1.0) > 0.45)
                and m.get("d_index_middle_tip", 1.0) < 0.50
                and not m.get("is_P_shape")
            ):
                return "N", 0.94

            # A: Thumb rests straight upright on the lateral SIDE of the fist
            # STRICT CRITERIA:
            # - Thumb MUST point UP: is_thumb_pointing_up is True
            # - Thumb MUST NOT point down: not is_thumb_pointing_down
            # - Thumb MUST NOT point horizontally: not is_thumb_horizontal
            # - Thumb MUST be on lateral/radial side: t_thumb <= 0.22
            # - Thumb tip elevated along/above knuckles: h_thumb >= -0.05
            # - Knuckles MUST be horizontal across the hand: is_knuckles_horizontal is True
            # - Rejects S, T, M, N, E shapes
            is_A = bool(
                m.get("is_knuckles_horizontal", True)
                and m["is_thumb_pointing_up"]
                and not m["is_thumb_horizontal"]
                and not m.get("is_thumb_pointing_down", False)
                and t_thumb <= 0.22
                and h_thumb >= -0.05
                and not m.get("is_S_shape")
                and not m.get("is_T_shape")
                and not m.get("is_M_shape")
                and not m.get("is_N_shape")
                and not m.get("is_E_shape")
            )
            if is_A:
                return "A", 0.96

            # If no defined fist gesture matches strictly, return None (never guess randomly)
            return "None", 0.0

        return "None", 0.0
