"""
Geometric Landmark Normalizer and Anatomical Analyzer
Extracts rotation-invariant and handedness-invariant metrics for ASL alphabet recognition.
Supports externalized heuristic thresholds via configuration.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np


class LandmarkNormalizer:
    """Performs geometric normalization and anatomical feature extraction on 21 hand landmarks."""

    def __init__(self, epsilon: float = 1e-6, heuristics: Optional[Dict[str, Any]] = None) -> None:
        self.epsilon = epsilon
        self.heuristics: Dict[str, Any] = heuristics or {}

    def normalize(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """
        Normalizes 21 3D landmark points to be invariant to translation and camera distance.
        Centers wrist at (0, 0, 0) and scales by maximum Euclidean distance.
        """
        if landmarks_3d.shape != (21, 3):
            raise ValueError(f"Expected input array of shape (21, 3), got {landmarks_3d.shape}")

        wrist = landmarks_3d[0]
        centered = landmarks_3d - wrist

        distances = np.linalg.norm(centered, axis=1)
        max_dist = np.max(distances)
        if max_dist < self.epsilon:
            max_dist = self.epsilon

        scaled = centered / max_dist
        return scaled.flatten().astype(np.float32)

    def analyze_hand(
        self,
        lm: np.ndarray,
        handedness: str = "Right",
        heuristics_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive 3D anatomical analysis for ASL fingerspelling.
        Handles ALL 26 letter orientations: upright, horizontal/sideways, and downward-pointing.
        """
        h_cfg = self.heuristics.copy()
        if heuristics_override:
            h_cfg.update(heuristics_override)

        wrist = lm[0]

        # Scale reference: distance from wrist(0) to middle MCP(9)
        ref_scale = float(np.linalg.norm(lm[9] - wrist))
        if ref_scale < 1e-4:
            ref_scale = 1.0

        # Longitudinal hand vector: Wrist(0) -> Middle MCP(9)
        v_long = lm[9] - wrist
        norm_long = float(np.linalg.norm(v_long))
        u_long = v_long / (norm_long + 1e-5)

        # Transverse palm vector: Index MCP(5) -> Pinky MCP(17)
        v_knuckles = lm[17] - lm[5]
        knuckle_span = float(np.linalg.norm(v_knuckles))
        knuckle_span_sq = float(np.dot(v_knuckles, v_knuckles))
        if knuckle_span_sq < 1e-5:
            knuckle_span_sq = 1e-5

        # ----------------------------------------------------------------
        # GLOBAL HAND ORIENTATION
        # ----------------------------------------------------------------
        is_hand_horizontal = bool(abs(v_long[0]) > 0.75 * abs(v_long[1]))
        is_hand_pointing_down = bool(
            v_long[1] > 0.10 * ref_scale
            and abs(v_long[1]) > 0.55 * abs(v_long[0])
        )
        is_hand_upright = bool(
            v_long[1] < -0.10 * ref_scale
            and abs(v_long[1]) > 0.55 * abs(v_long[0])
        )

        # ----------------------------------------------------------------
        # PER-FINGER EXTENSION ANALYSIS
        # ----------------------------------------------------------------
        def check_finger(tip_idx: int, pip_idx: int, mcp_idx: int) -> Tuple[bool, bool, bool, bool, float]:
            tip = lm[tip_idx]
            pip = lm[pip_idx]
            mcp = lm[mcp_idx]

            d_tip_mcp = float(np.linalg.norm(tip - mcp))
            d_pip_mcp = float(np.linalg.norm(pip - mcp))
            ratio = float(d_tip_mcp / (d_pip_mcp + 1e-5))

            is_ext_len = bool(d_tip_mcp > 0.42 * ref_scale)
            is_up = bool(is_ext_len and tip[1] < mcp[1] - 0.15 * ref_scale)
            is_down = bool(d_tip_mcp > 0.55 * ref_scale and tip[1] > pip[1] + 0.12 * ref_scale)

            dx = abs(tip[0] - mcp[0])
            dy = abs(tip[1] - mcp[1])
            is_horiz = bool(is_ext_len and dx > 0.28 * ref_scale and dx > 0.75 * dy)

            extended = bool(is_up or is_down or is_horiz)
            return extended, is_up, is_horiz, is_down, ratio

        ext_index, up_index, horiz_index, down_index, r_index = check_finger(8, 6, 5)
        ext_middle, up_middle, horiz_middle, down_middle, r_middle = check_finger(12, 10, 9)
        ext_ring, up_ring, horiz_ring, down_ring, r_ring = check_finger(16, 14, 13)
        ext_pinky, up_pinky, horiz_pinky, down_pinky, r_pinky = check_finger(20, 18, 17)

        # ----------------------------------------------------------------
        # THUMB GEOMETRY
        # ----------------------------------------------------------------
        thumb_tip = lm[4]
        thumb_mcp = lm[2]

        v_thumb = thumb_tip - thumb_mcp
        norm_thumb = float(np.linalg.norm(v_thumb))

        # Transverse thumb position: 0.0 = at index MCP, 1.0 = at pinky MCP
        t_thumb = float(np.dot(thumb_tip - lm[5], v_knuckles) / knuckle_span_sq)

        # Longitudinal height: how far thumb extends past index MCP along hand axis
        h_thumb = float(np.dot(thumb_tip - lm[5], u_long) / ref_scale)

        d_thumb_index_mcp = float(np.linalg.norm(thumb_tip - lm[5]) / ref_scale)
        d_thumb_wrist = float(np.linalg.norm(thumb_tip - wrist) / ref_scale)

        is_thumb_pointing_up = bool(
            v_thumb[1] < -0.04 * ref_scale
            and abs(v_thumb[1]) > 0.55 * abs(v_thumb[0])
        )
        l_thumb_ratio = float(h_cfg.get("l_thumb_horizontal_ratio", 0.85))
        is_thumb_horizontal = bool(abs(v_thumb[0]) > l_thumb_ratio * abs(v_thumb[1]))
        is_thumb_pointing_down = bool(
            v_thumb[1] > 0.04 * ref_scale
            and abs(v_thumb[1]) > 0.55 * abs(v_thumb[0])
        )

        thumb_extended_out = bool(d_thumb_index_mcp > 0.42 and d_thumb_wrist > 0.65 and t_thumb < 0.25)

        # ----------------------------------------------------------------
        # LETTER K: Thumb elevated UP between index and middle fingers
        # In K, the thumb tip points upward and rests between index and middle fingers
        # (close to index PIP or middle PIP/MCP, elevated above knuckles).
        # In contrast, in V the thumb is folded down across palm/ring finger.
        # ----------------------------------------------------------------
        d_thumb_index_pip = float(np.linalg.norm(thumb_tip - lm[6]) / ref_scale)
        d_thumb_middle_pip = float(np.linalg.norm(thumb_tip - lm[10]) / ref_scale)
        d_thumb_middle_mcp = float(np.linalg.norm(thumb_tip - lm[9]) / ref_scale)

        # X corridor spanning the index and middle fingers (from MCP through TIP)
        x_min_idx = min(lm[5][0], lm[6][0], lm[8][0])
        x_max_idx = max(lm[5][0], lm[6][0], lm[8][0])
        x_min_mid = min(lm[9][0], lm[10][0], lm[12][0])
        x_max_mid = max(lm[9][0], lm[10][0], lm[12][0])
        x_left = min(x_min_idx, x_min_mid) - 0.06 * ref_scale
        x_right = max(x_max_idx, x_max_mid) + 0.06 * ref_scale

        is_thumb_in_finger_corridor = bool(x_left <= thumb_tip[0] <= x_right)

        # Thumb is elevated (not folded flat onto ring/pinky finger)
        is_thumb_elevated = bool(
            (h_thumb > 0.04 or thumb_tip[1] < min(lm[5][1], lm[9][1]) + 0.08 * ref_scale)
            and (is_thumb_pointing_up or v_thumb[1] < 0.0)
        )

        k_prox_thresh = float(h_cfg.get("k_thumb_proximity", 0.45))
        is_thumb_near_im = bool(
            (d_thumb_index_pip < k_prox_thresh or d_thumb_middle_pip < k_prox_thresh or d_thumb_middle_mcp < k_prox_thresh)
            and is_thumb_in_finger_corridor
        )

        is_thumb_between_index_middle = bool(
            is_thumb_elevated
            and (is_thumb_near_im or (is_thumb_in_finger_corridor and is_thumb_pointing_up))
        )

        # ----------------------------------------------------------------
        # VECTOR ALIGNMENT: Index finger vs Thumb (for G and L)
        # ----------------------------------------------------------------
        v_index_seg = lm[8] - lm[5]
        norm_idx = float(np.linalg.norm(v_index_seg))
        u_index_seg = v_index_seg / (norm_idx + 1e-5)
        u_thumb_seg = v_thumb / (norm_thumb + 1e-5)
        dot_thumb_index = float(np.dot(u_index_seg[:2], u_thumb_seg[:2]))

        thumb_index_parallel = bool(
            dot_thumb_index > 0.25
            and not up_index
            and (is_hand_horizontal or horiz_index or (not up_index and not down_index))
        )

        # ----------------------------------------------------------------
        # INTER-TIP DISTANCE METRICS
        # ----------------------------------------------------------------
        d_thumb_index_tip = float(np.linalg.norm(thumb_tip - lm[8]) / ref_scale)
        d_thumb_middle_tip = float(np.linalg.norm(thumb_tip - lm[12]) / ref_scale)
        d_thumb_ring_tip = float(np.linalg.norm(thumb_tip - lm[16]) / ref_scale)
        d_thumb_pinky_tip = float(np.linalg.norm(thumb_tip - lm[20]) / ref_scale)
        d_index_middle_tip = float(np.linalg.norm(lm[8] - lm[12]) / ref_scale)
        d_middle_ring_tip = float(np.linalg.norm(lm[12] - lm[16]) / ref_scale)
        d_ring_pinky_tip = float(np.linalg.norm(lm[16] - lm[20]) / ref_scale)

        # ----------------------------------------------------------------
        # LETTER R: Index and Middle CROSSED
        # ----------------------------------------------------------------
        crossed_by_x = bool((lm[8][0] - lm[12][0]) * (lm[5][0] - lm[9][0]) < 0)
        are_index_middle_crossed = bool(
            up_index and up_middle
            and not is_hand_horizontal
            and not is_thumb_between_index_middle
            and (crossed_by_x or d_index_middle_tip < 0.09)
        )

        # ----------------------------------------------------------------
        # LETTER X: Index hooked back
        # ----------------------------------------------------------------
        d_index_tip_mcp = float(np.linalg.norm(lm[8] - lm[5]) / ref_scale)
        pip_elevated = bool(
            lm[6][1] < lm[5][1] - 0.15 * ref_scale
            and lm[6][1] < lm[10][1] - 0.05 * ref_scale
        )
        tip_hooked_down = bool(lm[8][1] > lm[6][1])
        index_hooked = bool(
            not ext_index
            and pip_elevated
            and tip_hooked_down
            and d_index_tip_mcp > 0.22
            and d_thumb_index_tip > 0.32
            and not ext_middle and not ext_ring and not ext_pinky
        )

        # ----------------------------------------------------------------
        # LETTER O: Fingertips converge to thumb tip in circle
        # In O, all 4 fingertips converge to the thumb tip to form an open circle/loop.
        # In contrast, in F only index touches thumb, and other fingers extend straight UP.
        # ----------------------------------------------------------------
        o_ti_max = float(h_cfg.get("o_thumb_index_dist", 0.36))
        o_tm_max = float(h_cfg.get("o_thumb_middle_dist", 0.36))
        o_tr_max = float(h_cfg.get("o_thumb_ring_dist", 0.38))
        o_tp_max = float(h_cfg.get("o_thumb_pinky_dist", 0.45))
        o_mcp5_min = float(h_cfg.get("o_mcp5_wrist_dist", 0.65))

        d_mcp5_wrist = float(np.linalg.norm(lm[5] - wrist) / ref_scale)
        is_O_shape = bool(
            d_thumb_index_tip < o_ti_max
            and d_thumb_middle_tip < o_tm_max
            and d_thumb_ring_tip < o_tr_max
            and d_thumb_pinky_tip < o_tp_max
            and d_mcp5_wrist > o_mcp5_min
            and not (up_middle and up_ring and up_pinky)
            and not (down_index and lm[8][1] > lm[5][1] + 0.15 * ref_scale)
        )

        # ----------------------------------------------------------------
        # LETTER D: Index UP, middle loops to touch thumb
        # ----------------------------------------------------------------
        d_middle_wrist = float(np.linalg.norm(lm[12] - wrist) / ref_scale)
        is_D_loop = bool(
            d_thumb_middle_tip < 0.35
            and d_middle_wrist > 0.40
        )

        # ----------------------------------------------------------------
        # C SHAPE DETECTION: All 4 fingers curved in C formation
        # In C, fingers are arched in a curved C-cup (never flat upright open palm).
        # Thumb curves opposite to form the lower jaw of C with open aperture (0.22 - 0.85).
        # ----------------------------------------------------------------
        d_index_wrist_norm = float(np.linalg.norm(lm[8] - wrist) / ref_scale)
        d_ring_wrist_norm = float(np.linalg.norm(lm[16] - wrist) / ref_scale)

        # Flat open hand (Image 1): all 4 fingers straight upright and thumb out
        is_flat_open_hand = bool(
            up_index and up_middle and up_ring and up_pinky
            and (thumb_extended_out or d_index_middle_tip > 0.35 or d_middle_ring_tip > 0.35)
        )

        is_C_shape = bool(
            not is_O_shape
            and not is_flat_open_hand
            and d_index_middle_tip < 0.38
            and d_thumb_index_tip > 0.22
            and d_thumb_index_tip < 0.85
            and d_thumb_middle_tip < 0.88
            and d_index_wrist_norm > 0.45
            and d_ring_wrist_norm > 0.35
            and (thumb_extended_out or d_thumb_index_mcp > 0.35)
            and (horiz_index or horiz_middle or is_hand_horizontal or not up_index or lm[8][2] < lm[5][2] - 0.10)
        )

        # ----------------------------------------------------------------
        # LETTER P: Downward K-shape (Index forward/down, Middle straight down)
        # Index points forward/horizontally or down; Middle points down.
        # Index and middle are spread apart (d_im > 0.38).
        # Thumb rests near the middle finger joint.
        # Ring and pinky are folded (not pointing up).
        # Hand is turned horizontally or pointing down.
        # ----------------------------------------------------------------
        d_thumb_mid_pip = float(np.linalg.norm(thumb_tip - lm[10]) / ref_scale)
        d_thumb_mid_dip = float(np.linalg.norm(thumb_tip - lm[11]) / ref_scale)
        is_P_shape = bool(
            ext_index
            and down_middle
            and not up_index
            and not up_ring
            and not up_pinky
            and d_index_middle_tip > 0.38
            and (is_hand_horizontal or is_hand_pointing_down or horiz_index or down_index)
            and (d_thumb_mid_pip < 0.50 or d_thumb_mid_dip < 0.45 or d_thumb_middle_tip < 0.60)
        )

        # ----------------------------------------------------------------
        # LETTER Q: Downward G-shape (Index & thumb pointing downward parallel)
        # Index points down (lm[8][1] > lm[5][1] + 0.15 * ref_scale or down_index).
        # Thumb points downward/horizontally alongside index.
        # Thumb and index are parallel (thumb_index_parallel or dot_thumb_index > 0.60).
        # Distance between thumb tip and index tip is small (d_thumb_index_tip < 0.40).
        # Fingers are not pointing up: not (up_index or up_middle or up_ring or up_pinky).
        # Index and middle are NOT spread wide apart (not scissors/P: d_index_middle_tip < 0.38).
        # ----------------------------------------------------------------
        is_Q_shape = bool(
            (down_index or lm[8][1] > lm[5][1] + 0.15 * ref_scale)
            and not up_index
            and not up_middle
            and not up_ring
            and not up_pinky
            and (thumb_index_parallel or dot_thumb_index > 0.60)
            and d_thumb_index_tip < 0.40
            and d_index_middle_tip < 0.38
            and (lm[4][1] > lm[2][1] - 0.05 * ref_scale or is_thumb_pointing_down or is_thumb_horizontal)
        )

        # ----------------------------------------------------------------
        # LETTER L: Index UP, thumb extended OUT to the side in L-shape
        # Index points straight UP (not hooked or pointing sideways).
        # Thumb extends outward laterally (horizontal / perpendicular to index).
        # Remaining fingers (middle, ring, pinky) are folded down.
        # Thumb is not looping to touch middle finger (unlike D).
        # ----------------------------------------------------------------
        l_horiz_ratio = float(h_cfg.get("l_thumb_horizontal_ratio", 0.70))
        is_L_shape = bool(
            up_index
            and not index_hooked
            and not is_D_loop
            and (thumb_extended_out or d_thumb_index_mcp > 0.48)
            and (is_thumb_horizontal or abs(v_thumb[0]) > l_horiz_ratio * abs(v_thumb[1]))
            and d_thumb_index_tip > 0.55
            and d_thumb_middle_tip > 0.35
            and not is_thumb_pointing_up
            and not up_middle and not up_ring and not up_pinky
        )

        # ----------------------------------------------------------------
        # PINKY TILT
        # ----------------------------------------------------------------
        v_pinky_vec = lm[20] - wrist
        is_pinky_tilted = bool(abs(v_pinky_vec[0]) > 0.35 * abs(v_pinky_vec[1]))
        is_hand_tilted = bool(abs(v_long[0]) > 0.35 * abs(v_long[1]))

        return {
            # Finger extension flags
            "ext_index": ext_index,
            "ext_middle": ext_middle,
            "ext_ring": ext_ring,
            "ext_pinky": ext_pinky,
            "up_index": up_index,
            "up_middle": up_middle,
            "up_ring": up_ring,
            "up_pinky": up_pinky,
            "horiz_index": horiz_index,
            "horiz_middle": horiz_middle,
            "down_index": down_index,
            "down_middle": down_middle,
            # Hand orientation
            "is_hand_horizontal": is_hand_horizontal,
            "is_hand_pointing_down": is_hand_pointing_down,
            "is_hand_upright": is_hand_upright,
            "is_hand_tilted": is_hand_tilted,
            # Thumb geometry
            "t_thumb": t_thumb,
            "h_thumb": h_thumb,
            "is_thumb_pointing_up": is_thumb_pointing_up,
            "is_thumb_horizontal": is_thumb_horizontal,
            "is_thumb_pointing_down": is_thumb_pointing_down,
            "is_thumb_between_index_middle": is_thumb_between_index_middle,
            "is_thumb_elevated": is_thumb_elevated,
            "is_thumb_in_finger_corridor": is_thumb_in_finger_corridor,
            "d_thumb_index_pip": d_thumb_index_pip,
            "d_thumb_middle_pip": d_thumb_middle_pip,
            "thumb_extended_out": thumb_extended_out,
            "thumb_index_parallel": thumb_index_parallel,
            "dot_thumb_index": dot_thumb_index,
            # Complex shape detectors
            "are_index_middle_crossed": are_index_middle_crossed,
            "index_hooked": index_hooked,
            "is_O_shape": is_O_shape,
            "is_C_shape": is_C_shape,
            "is_D_loop": is_D_loop,
            "is_L_shape": is_L_shape,
            "is_P_shape": is_P_shape,
            "is_Q_shape": is_Q_shape,
            # Inter-tip distances
            "d_thumb_index_tip": d_thumb_index_tip,
            "d_thumb_middle_tip": d_thumb_middle_tip,
            "d_thumb_ring_tip": d_thumb_ring_tip,
            "d_thumb_pinky_tip": d_thumb_pinky_tip,
            "d_index_middle_tip": d_index_middle_tip,
            "d_middle_ring_tip": d_middle_ring_tip,
            "d_ring_pinky_tip": d_ring_pinky_tip,
            "d_thumb_index_mcp": d_thumb_index_mcp,
            "d_mcp5_wrist": d_mcp5_wrist,
            "d_middle_wrist": d_middle_wrist,
            # Pinky flags
            "is_pinky_tilted": is_pinky_tilted,
            "ref_scale": ref_scale,
        }
