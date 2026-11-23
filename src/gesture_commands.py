"""
Control Gesture Recognition Module for ASL Fingerspelling Studio.
Recognizes non-alphabet control gestures such as THUMBS_UP (word confirmation)
and THUMBS_DOWN (word/letter deletion) without conflicting with ASL alphabet signs (e.g. 'A').
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple
import numpy as np


class ControlGestureType(str, Enum):
    NONE = "NONE"
    THUMBS_UP = "THUMBS_UP"
    THUMBS_DOWN = "THUMBS_DOWN"


class ControlGestureRecognizer:
    """
    Detects and validates control gestures separate from standard ASL alphabet signs.
    Ensures that gestures like THUMBS_UP are never confused with ASL 'A' or 'E'.
    """

    def __init__(self, min_confidence: float = 0.92) -> None:
        self.min_confidence = float(min_confidence)

    def recognize(
        self,
        landmarks: np.ndarray,
        m: Dict[str, Any],
        handedness: str = "Right",
    ) -> Tuple[ControlGestureType, float]:
        """
        Evaluates 21 landmarks and precomputed anatomical features for control gestures.
        Returns (ControlGestureType, confidence).
        """
        if landmarks is None or len(landmarks) < 21 or not m:
            return ControlGestureType.NONE, 0.0

        # 1. Evaluate THUMBS_UP (Confirmation)
        is_tu, tu_conf = self.is_thumbs_up(landmarks, m, handedness)
        if is_tu and tu_conf >= self.min_confidence:
            return ControlGestureType.THUMBS_UP, tu_conf

        # 2. Evaluate THUMBS_DOWN (Delete / Cancel)
        is_td, td_conf = self.is_thumbs_down(landmarks, m, handedness)
        if is_td and td_conf >= self.min_confidence:
            return ControlGestureType.THUMBS_DOWN, td_conf

        return ControlGestureType.NONE, 0.0

    def is_thumbs_up(
        self,
        lm: np.ndarray,
        m: Dict[str, Any],
        handedness: str = "Right",
    ) -> Tuple[bool, float]:
        """
        Determines whether the hand is in a distinct THUMBS_UP gesture.

        Strict Anatomical Criteria:
        1. All 4 fingers (Index, Middle, Ring, Pinky) are clenched tight in a fist.
           - None of the 4 fingers are extended or pointing up.
           - Tips are folded close to the palm.
        2. The thumb is fully extended and pointing UPWARD (isolated from the fist).
           - lm[4][1] < lm[3][1] < lm[2][1] (tip higher than IP, IP higher than MCP).
           - is_thumb_pointing_up is True.
           - Thumb tip is the highest landmark in the entire hand by a wide margin.
           - Thumb tip is substantially elevated above the knuckles (h_thumb >= 0.20).
        3. Strict Distinction from ASL 'A':
           - In ASL 'A', the thumb tip rests snugly against the side of the index knuckle (h_thumb <= 0.15, d_tim <= 0.22).
           - In THUMBS_UP, the thumb is fully erect and elevated high above the fist (h_thumb >= 0.22).
        4. Rejects open hands, C-shape, O-shape, and hooked fingers.
        """
        # Guard: 4 fingers must NOT be extended or pointing up
        idx_ext = m.get("ext_index", False)
        mid_ext = m.get("ext_middle", False)
        rng_ext = m.get("ext_ring", False)
        pnk_ext = m.get("ext_pinky", False)

        idx_up = m.get("up_index", False)
        mid_up = m.get("up_middle", False)
        rng_up = m.get("up_ring", False)
        pnk_up = m.get("up_pinky", False)

        if idx_ext or mid_ext or rng_ext or pnk_ext or idx_up or mid_up or rng_up or pnk_up:
            return False, 0.0

        if m.get("is_flat_open_hand", False) or m.get("is_C_shape", False) or m.get("is_O_shape", False):
            return False, 0.0

        ref_scale = float(m.get("ref_scale", 1.0))
        if ref_scale <= 1e-6:
            return False, 0.0

        # Thumb vertical direction check
        tip_y = lm[4][1]
        ip_y = lm[3][1]
        mcp_y = lm[2][1]

        dx_thumb = abs(lm[4][0] - lm[2][0])
        dy_thumb = lm[2][1] - lm[4][1]  # positive when tip is higher in space than MCP

        is_down = m.get("is_thumb_pointing_down", False)

        # Must be pointing upward (tip higher than MCP and IP, not pointing down, positive upward dy)
        if is_down or dy_thumb <= 0:
            return False, 0.0

        if not (tip_y < ip_y and ip_y < mcp_y + 0.04 * ref_scale):
            return False, 0.0

        # Thumb upward slant: dy_thumb >= 0.45 * dx_thumb (angle > ~24 degrees above horizontal)
        # Supports authentic human thumbs-up poses tilted comfortably toward camera
        if dy_thumb < 0.45 * dx_thumb:
            return False, 0.0

        # Knuckle orientation (MCP 5 to MCP 17)
        dx_knuckles = abs(lm[17][0] - lm[5][0])
        dy_knuckles = abs(lm[17][1] - lm[5][1])

        # Strict Distinction from ASL 'A' and closed fists (A, E, S, T, M, N):
        # In ASL fists (like 'A'), the fist faces the camera: knuckles (MCP 5 to 17) run horizontally (dx > 1.45 * dy).
        # In THUMBS_UP, the hand is held sideways: knuckles run vertically or tilted down.
        if dx_knuckles > 1.45 * dy_knuckles:
            return False, 0.0

        # Knuckle heights (MCP 5, 9, 13, 17)
        knuckle_min_y = min(lm[5][1], lm[9][1], lm[13][1], lm[17][1])
        elevation_above_knuckles = (knuckle_min_y - tip_y) / ref_scale

        # Must tower high above all knuckles (at least 0.20 normalized ref_scale above)
        if elevation_above_knuckles < 0.20:
            return False, 0.0

        h_thumb = float(m.get("h_thumb", 0.0))
        # Clear distinction from ASL 'A': ASL 'A' has h_thumb near 0.0 (max ~0.15)
        if h_thumb < 0.18:
            return False, 0.0

        # Fingertips (8, 12, 16, 20) must be lower than thumb tip
        for tip_idx in (8, 12, 16, 20):
            if lm[tip_idx][1] < tip_y + 0.06 * ref_scale:
                return False, 0.0

        # Compute confidence score
        score = 0.94
        if elevation_above_knuckles >= 0.35:
            score += 0.03
        if dy_thumb >= 1.0 * dx_thumb:
            score += 0.02
        confidence = float(np.clip(score, 0.92, 0.99))

        return True, confidence

    def is_thumbs_down(
        self,
        lm: np.ndarray,
        m: Dict[str, Any],
        handedness: str = "Right",
    ) -> Tuple[bool, float]:
        """
        Determines whether the hand is in a distinct THUMBS_DOWN gesture.
        Used as an optional gesture command for DELETE / CLEAR.
        """
        # Guard: 4 fingers must NOT be extended or pointing up
        idx_ext = m.get("ext_index", False)
        mid_ext = m.get("ext_middle", False)
        rng_ext = m.get("ext_ring", False)
        pnk_ext = m.get("ext_pinky", False)

        if idx_ext or mid_ext or rng_ext or pnk_ext:
            return False, 0.0

        if m.get("is_flat_open_hand", False) or m.get("is_C_shape", False) or m.get("is_O_shape", False):
            return False, 0.0

        ref_scale = float(m.get("ref_scale", 1.0))
        if ref_scale <= 1e-6:
            return False, 0.0

        is_down = m.get("is_thumb_pointing_down", False)
        is_up = m.get("is_thumb_pointing_up", False)
        is_horiz = m.get("is_thumb_horizontal", False)

        if not is_down or is_up or is_horiz:
            return False, 0.0

        # Anatomical ordering: Tip (4) lower than IP (3) lower than MCP (2)
        tip_y = lm[4][1]
        ip_y = lm[3][1]
        mcp_y = lm[2][1]

        if not (tip_y > ip_y and ip_y > mcp_y - 0.02 * ref_scale):
            return False, 0.0

        knuckle_max_y = max(lm[5][1], lm[9][1], lm[13][1], lm[17][1])
        drop_below_knuckles = (tip_y - knuckle_max_y) / ref_scale

        if drop_below_knuckles < 0.12:
            return False, 0.0

        dx_thumb = abs(lm[4][0] - lm[2][0])
        dy_thumb = lm[4][1] - lm[2][1]  # positive when pointing down
        if dy_thumb < 1.15 * dx_thumb:
            return False, 0.0

        confidence = float(np.clip(0.92 + 0.04 * (drop_below_knuckles / 0.25), 0.92, 0.98))
        return True, confidence

