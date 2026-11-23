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

    def canonical_align(
        self,
        landmarks_3d: np.ndarray,
        handedness: str = "Right",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms 21 3D landmarks into a canonical orientation-invariant coordinate frame.
        - Wrist at (0, 0, 0)
        - Middle MCP at (0, 1, 0) (longitudinal Y-axis)
        - Palm normal along Z-axis
        - Knuckles along X-axis (chirality-invariant between Left and Right hands)
        Returns:
            canonical_landmarks: Shape (21, 3) invariant to 3D rotation, pitch, roll, and yaw.
            rotation_matrix: Shape (3, 3) orthogonal transformation matrix.
        """
        if landmarks_3d.shape != (21, 3):
            raise ValueError(f"Expected input array of shape (21, 3), got {landmarks_3d.shape}")

        wrist = landmarks_3d[0]
        centered = landmarks_3d - wrist

        # Longitudinal axis (Wrist -> Middle MCP)
        v_long = centered[9]
        norm_long = float(np.linalg.norm(v_long))
        if norm_long < self.epsilon:
            norm_long = 1.0
        u_y = v_long / norm_long

        # Knuckle vector: adjusted for handedness so both hands map to identical canonical geometry
        if handedness.lower() == "left":
            v_knuck = centered[17] - centered[5]
        else:
            v_knuck = centered[5] - centered[17]

        # Palm normal vector
        v_z = np.cross(v_knuck, u_y)
        norm_z = float(np.linalg.norm(v_z))
        if norm_z < self.epsilon:
            u_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            u_z = v_z / norm_z

        # Transverse vector
        u_x = np.cross(u_y, u_z)
        norm_x = float(np.linalg.norm(u_x))
        if norm_x > self.epsilon:
            u_x = u_x / norm_x

        # 3x3 Rotation matrix
        R = np.vstack([u_x, u_y, u_z]).astype(np.float32)

        # Scale reference (wrist to middle MCP)
        scale = norm_long
        canonical = (centered / scale) @ R.T

        # Chirality reflection: mirror Left hand so canonical coordinates match Right hand identically
        if handedness.lower() == "left":
            canonical[:, 0] = -canonical[:, 0]

        return canonical.astype(np.float32), R

    def normalize(
        self,
        landmarks_3d: np.ndarray,
        canonical: bool = False,
        handedness: str = "Right",
    ) -> np.ndarray:
        """
        Normalizes 21 3D landmark points to be invariant to translation, scale, and optionally 3D rotation.
        """
        if canonical:
            c_lm, _ = self.canonical_align(landmarks_3d, handedness=handedness)
            return c_lm.flatten().astype(np.float32)

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
        knuckle_span = float(np.linalg.norm(lm[17] - lm[5]))
        if ref_scale < 0.65 * knuckle_span and knuckle_span > 1e-4:
            # Wrist collapsed into palm during heavy finger occlusion (e.g. draped fist)
            ref_scale = max(ref_scale, 0.85 * knuckle_span)
        elif ref_scale < 1e-4:
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
        # In MediaPipe screen space, y=0 is at top (ceiling), y=1 is at bottom (floor)
        # In 3D canonical space, (0, 1, 0) is longitudinal middle MCP axis
        # In meter space, wrist is near origin and middle MCP is along +Y
        # ----------------------------------------------------------------
        is_canonical = bool(
            abs(wrist[0]) < 1e-2
            and abs(wrist[1]) < 1e-2
            and abs(v_long[0]) < 0.05
            and (abs(v_long[1] - 1.0) < 0.05 or (v_long[1] > 0.08 and abs(v_long[1] - ref_scale) < 1e-2))
        )
        sign_y = 1.0 if is_canonical else -1.0

        is_hand_horizontal = bool(abs(v_long[0]) > 0.75 * abs(v_long[1]))
        is_hand_pointing_down = bool(
            (v_long[1] * sign_y < -0.10 * ref_scale) if is_canonical else (v_long[1] > 0.10 * ref_scale)
            and abs(v_long[1]) > 0.55 * abs(v_long[0])
        )
        is_hand_upright = bool(
            (v_long[1] * sign_y > 0.10 * ref_scale) if is_canonical else (v_long[1] < -0.10 * ref_scale)
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

            # Pinky has a higher threshold to avoid counting a curled pinky inside fist or U/V as extended
            min_ext_thresh = 0.48 * ref_scale if tip_idx == 20 else 0.36 * ref_scale
            is_ext_len = bool(d_tip_mcp > min_ext_thresh)
            dx = abs(tip[0] - mcp[0])
            dy = abs(tip[1] - mcp[1])
            dz = abs(tip[2] - mcp[2])
            is_horiz = bool(is_ext_len and dx > 0.25 * ref_scale and dx > 0.65 * dy)
            is_forward = bool(is_ext_len and dz > 0.20 * ref_scale)
            pip_up_ok = bool(tip_idx != 20 or ((tip[1] - pip[1]) * sign_y > 0.02 * ref_scale and (tip[1] - lm[19][1]) * sign_y >= -0.01 * ref_scale))
            is_up = bool(is_ext_len and ((tip[1] - mcp[1]) * sign_y > 0.15 * ref_scale) and (dy > 0.70 * dx) and not (tip_idx == 20 and is_hand_horizontal) and pip_up_ok)
            is_down = bool(d_tip_mcp > 0.55 * ref_scale and ((tip[1] - pip[1]) * sign_y < -0.12 * ref_scale))

            extended = bool(is_up or is_down or is_horiz or is_forward or d_tip_mcp > 0.45 * ref_scale)
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
            v_thumb[1] * sign_y > 0.04 * ref_scale
            and abs(v_thumb[1]) > 0.55 * abs(v_thumb[0])
        )
        l_thumb_ratio = float(h_cfg.get("l_thumb_horizontal_ratio", 0.85))
        is_thumb_horizontal = bool(abs(v_thumb[0]) > l_thumb_ratio * abs(v_thumb[1]))
        is_thumb_pointing_down = bool(
            v_thumb[1] * sign_y < -0.04 * ref_scale
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

        # Check for crossed fingers geometry (used to prevent false K on R handshape)
        v_mcp_im = lm[9][:2] - lm[5][:2]
        v_tip_im = lm[12][:2] - lm[8][:2]
        dot_tips_mcp = float(np.dot(v_tip_im, v_mcp_im))
        crossed_by_x = bool((lm[8][0] - lm[12][0]) * (lm[5][0] - lm[9][0]) < 0)
        crossed_by_dot = bool(dot_tips_mcp < -1e-6)

        def _ccw_r(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
            return bool((C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0]))

        def _segments_intersect_r(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> bool:
            return bool(_ccw_r(A, C, D) != _ccw_r(B, C, D) and _ccw_r(A, B, C) != _ccw_r(A, B, D))

        seg_crossed = bool(
            _segments_intersect_r(lm[5][:2], lm[8][:2], lm[9][:2], lm[12][:2])
            or _segments_intersect_r(lm[6][:2], lm[8][:2], lm[10][:2], lm[12][:2])
        )
        are_crossed_geometry = bool(crossed_by_x or crossed_by_dot or seg_crossed)

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

        # Transverse / lateral distance between index and middle fingers perpendicular to hand axis
        u_long_2d = u_long[:2] / (np.linalg.norm(u_long[:2]) + 1e-5)
        u_lat = np.array([-u_long_2d[1], u_long_2d[0]], dtype=np.float32)
        v_tips_im = lm[12][:2] - lm[8][:2]
        t_tips_im = abs(float(np.dot(v_tips_im, u_lat))) / ref_scale
        v_mcp_im = lm[9][:2] - lm[5][:2]
        t_mcp_im = abs(float(np.dot(v_mcp_im, u_lat))) / ref_scale
        u_v_transverse_ratio = float(t_tips_im / (t_mcp_im + 1e-5))

        d_mcp_im_2f = float(np.linalg.norm(lm[5][:2] - lm[9][:2]))
        u_v_euclidean_ratio = float((d_index_middle_tip * ref_scale) / (d_mcp_im_2f + 1e-5))
        u_v_spread_ratio = float(min(u_v_euclidean_ratio, u_v_transverse_ratio * 1.15))

        fingers_together = bool(t_tips_im < 0.18 or u_v_transverse_ratio < 1.15 or d_index_middle_tip < 0.20)
        fingers_spread_v = bool((t_tips_im >= 0.18 or u_v_transverse_ratio >= 1.15) and d_index_middle_tip >= 0.22)

        # In ASL K, the thumb is wedged between index and middle, strictly requiring the fingers to be spread apart in a V (fingers_spread_v and not fingers_together).
        # When fingers are touching together (fingers_together), the thumb cannot be between them (it rests in front, as in U).
        is_thumb_between_index_middle = bool(
            is_thumb_elevated
            and (is_thumb_near_im or (is_thumb_in_finger_corridor and is_thumb_pointing_up))
            and not are_crossed_geometry
            and fingers_spread_v
            and not fingers_together
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
        # LETTER R: Index and Middle CROSSED
        # ----------------------------------------------------------------
        are_index_middle_crossed = bool(
            up_index and up_middle
            and not is_hand_horizontal
            and not is_hand_pointing_down
            and not (up_ring or up_pinky)
            and are_crossed_geometry
        )
        is_R_shape = are_index_middle_crossed


        # ----------------------------------------------------------------
        # FINGER STRAIGHTNESS & CURVATURE INVARIANTS
        # Ratio of MCP-to-TIP Euclidean distance to total joint perimeter.
        # Straight fingers (B, open palm, high five): straightness >= 0.88
        # Arched/curled fingers (C, O, fist variants): straightness < 0.84
        # ----------------------------------------------------------------
        def _finger_straightness(mcp_idx: int) -> float:
            chord = float(np.linalg.norm(lm[mcp_idx + 3] - lm[mcp_idx]))
            perimeter = (
                float(np.linalg.norm(lm[mcp_idx + 1] - lm[mcp_idx]))
                + float(np.linalg.norm(lm[mcp_idx + 2] - lm[mcp_idx + 1]))
                + float(np.linalg.norm(lm[mcp_idx + 3] - lm[mcp_idx + 2]))
            )
            return chord / (perimeter + 1e-5)

        straight_idx = _finger_straightness(5)
        straight_mid = _finger_straightness(9)
        straight_rng = _finger_straightness(13)
        straight_pnk = _finger_straightness(17)

        are_fingers_arched = bool(straight_idx < 0.84 or straight_mid < 0.84)
        up_all_4 = bool(up_index and up_middle and up_ring and up_pinky)
        straight_fingers = bool(straight_idx > 0.88 and straight_mid > 0.88 and straight_rng > 0.88)

        # ----------------------------------------------------------------
        # LETTER X: Index hooked back
        # ----------------------------------------------------------------
        d_index_tip_mcp = float(np.linalg.norm(lm[8] - lm[5]) / ref_scale)
        pip_elevated = bool((lm[6][1] - lm[5][1]) * sign_y > 0.03 * ref_scale)
        tip_hooked_down = bool((lm[8][1] - lm[6][1]) * sign_y < -0.02 * ref_scale)

        v_mcp_pip = lm[6][:2] - lm[5][:2]
        v_pip_tip = lm[8][:2] - lm[6][:2]
        norm_mp = float(np.linalg.norm(v_mcp_pip))
        norm_pt = float(np.linalg.norm(v_pip_tip))
        dot_pip_idx = float(np.dot(v_mcp_pip, v_pip_tip) / (norm_mp * norm_pt + 1e-5))

        is_index_bent = bool(
            dot_pip_idx <= 0.88
            or straight_idx <= 0.84
            or tip_hooked_down
        )
        is_elevated_index = bool(
            d_index_tip_mcp >= 0.28
            and (ext_index or up_index or d_index_tip_mcp >= 0.30)
        )
        downward_mid_slope_x = float((lm[12][1] - lm[9][1]) * (-sign_y) / ref_scale)
        downward_idx_slope_x = float((lm[8][1] - lm[5][1]) * (-sign_y) / ref_scale)

        index_hooked = bool(
            is_elevated_index
            and pip_elevated
            and is_index_bent
            and not (up_middle or up_ring or up_pinky)
            and not (horiz_middle and straight_mid >= 0.80)
            and not (straight_mid > 0.50 and straight_rng > 0.50 and d_index_middle_tip < 0.38)
            and not (downward_mid_slope_x >= 0.35)
            and not (downward_idx_slope_x >= 0.35)
            and not thumb_index_parallel
            and not (horiz_index and not up_index)
            and not is_hand_horizontal
            and d_thumb_middle_tip > 0.20
            and not is_hand_pointing_down
            and not down_index
        )
        is_X_shape = index_hooked

        # ----------------------------------------------------------------
        # LETTER O: Fingertips converge to thumb tip in circle
        # In O, all 4 fingertips converge to the thumb tip to form an open circle/loop.
        # In contrast, in F only index touches thumb, and other fingers extend straight UP.
        # PIP joints are arched forward into a curved loop (d_pip_wrist > 0.60),
        # strictly distinguishing O from closed fists (S, E, A).
        # ----------------------------------------------------------------
        o_ti_max = float(h_cfg.get("o_thumb_index_dist", 0.36))
        o_tm_max = float(h_cfg.get("o_thumb_middle_dist", 0.36))
        o_tr_max = float(h_cfg.get("o_thumb_ring_dist", 0.35))
        o_tp_max = float(h_cfg.get("o_thumb_pinky_dist", 0.45))
        o_mcp5_min = float(h_cfg.get("o_mcp5_wrist_dist", 0.65))

        d_mcp5_wrist = float(np.linalg.norm(lm[5] - wrist) / ref_scale)
        d_pip6_wrist = float(np.linalg.norm(lm[6] - wrist) / ref_scale)
        d_pip10_wrist = float(np.linalg.norm(lm[10] - wrist) / ref_scale)
        pip_arched = bool(d_pip6_wrist > 0.60 and d_pip10_wrist > 0.60)

        is_O_shape = bool(
            d_thumb_index_tip < o_ti_max
            and d_thumb_middle_tip < o_tm_max
            and d_thumb_ring_tip < o_tr_max
            and d_thumb_pinky_tip < o_tp_max
            and d_mcp5_wrist > o_mcp5_min
            and pip_arched
            and (ext_index and ext_middle and (ext_ring or ext_pinky))
            and not (up_middle and up_ring and up_pinky)
            and not (down_index and lm[8][1] > lm[5][1] + 0.15 * ref_scale)
        )

        # ----------------------------------------------------------------
        # LETTER D: Index UP, middle loops to touch thumb
        # Index tip must be prominently elevated above curled middle PIP (10)
        # Thumb tip must contact middle finger tip (<= 0.20 * ref_scale)
        # ----------------------------------------------------------------
        d_middle_wrist = float(np.linalg.norm(lm[12] - wrist) / ref_scale)
        idx_tip_elevation = float((lm[8][1] - lm[10][1]) * sign_y / ref_scale)
        d_index_tip_mcp = float(np.linalg.norm(lm[8] - lm[5]) / ref_scale)

        is_D_loop = bool(
            d_thumb_middle_tip <= 0.20
            and d_middle_wrist > 0.40
            and idx_tip_elevation >= 0.35
            and d_index_tip_mcp >= 0.58
        )


        # ----------------------------------------------------------------
        # LETTER U and V: Two fingers pointing UP
        # In U: Index and Middle extended UP together (parallel/touching, spread_ratio < 1.25)
        # In V: Index and Middle extended UP spread apart into a V (spread_ratio >= 1.25)
        # Ring and pinky curled into palm (not up).
        # Thumb folded across ring/pinky (not extended out, not between fingers).
        # Fingers not crossed (not R).
        # Hand upright (not horizontal H, not downward P/N).
        # ----------------------------------------------------------------
        is_U_shape = bool(
            up_index
            and up_middle
            and not (up_ring or up_pinky)
            and not are_index_middle_crossed
            and not is_thumb_between_index_middle
            and not (thumb_extended_out and (t_thumb < -0.20 or is_thumb_horizontal))
            and not is_hand_horizontal
            and not is_hand_pointing_down
            and not (horiz_index and horiz_middle)
            and not (down_index or down_middle)
            and not index_hooked
            and straight_idx >= 0.85
            and straight_mid >= 0.85
            and fingers_together
        )

        is_V_shape = bool(
            up_index
            and up_middle
            and not (up_ring or up_pinky)
            and not are_index_middle_crossed
            and not is_thumb_between_index_middle
            and not thumb_extended_out
            and not is_hand_horizontal
            and not is_hand_pointing_down
            and not (horiz_index and horiz_middle)
            and not (down_index or down_middle)
            and not index_hooked
            and straight_idx >= 0.85
            and straight_mid >= 0.85
            and fingers_spread_v
            and not fingers_together
        )

        # ----------------------------------------------------------------
        # LETTER P: Downward K-shape (Index forward/down, Middle straight down)
        # Index points forward/horizontally or down; Middle points down.
        # Index and middle are spread apart like a K/scissors (spread_ratio > 1.80).
        # Thumb rests near the middle finger joint.
        # Ring and pinky are folded (not pointing up).
        # Hand is turned horizontally or pointing down.
        # ----------------------------------------------------------------
        d_thumb_mid_pip = float(np.linalg.norm(thumb_tip - lm[10]) / ref_scale)
        d_thumb_mid_dip = float(np.linalg.norm(thumb_tip - lm[11]) / ref_scale)
        d_tip_im = float(np.linalg.norm(lm[8][:2] - lm[12][:2]))
        d_mcp_im = float(np.linalg.norm(lm[5][:2] - lm[9][:2]))
        p_spread_ratio = float(d_tip_im / (d_mcp_im + 1e-5))
        downward_mid_slope = float((lm[12][1] - lm[9][1]) * (-sign_y) / ref_scale)
        is_P_shape = bool(
            ext_index
            and downward_mid_slope >= 0.35
            and not up_middle
            and not up_ring
            and not up_pinky
            and d_index_middle_tip > 0.40
            and p_spread_ratio >= 1.80
            and (d_thumb_mid_pip < 0.55 or d_thumb_mid_dip < 0.50 or d_thumb_middle_tip < 0.65)
        )

        # ----------------------------------------------------------------
        # LETTER Q: Downward G-shape (Index & thumb pointing downward parallel)
        # Index points down or diagonally down (downward_idx_slope >= 0.35 or down_index).
        # Index is extended straight: straight_idx >= 0.85.
        # Thumb points downward/horizontally alongside index.
        # Thumb and index are parallel (dot_thumb_index > 0.35 or thumb_index_parallel).
        # Distance between thumb tip and index tip is small (d_thumb_index_tip <= 0.70).
        # Middle, ring, pinky are curled in fist: not (up_middle or up_ring or up_pinky).
        # Index is extended (ext_index) and not pointing up (not up_index).
        # Not P shape (middle finger is curled in fist, not extended scissors).
        # ----------------------------------------------------------------
        downward_idx_slope = float((lm[8][1] - lm[5][1]) * (-sign_y) / ref_scale)
        downward_thumb_slope = float((lm[4][1] - lm[2][1]) * (-sign_y) / ref_scale)

        is_dangling_claw = bool(
            down_middle
            and down_ring
            and d_index_middle_tip < 0.15
            and d_thumb_index_tip > 0.35
        )

        is_Q_shape = bool(
            ext_index
            and straight_idx >= 0.85
            and not up_index
            and (downward_idx_slope >= 0.35 or (down_index and not horiz_index))
            and not (up_middle or up_ring or up_pinky)
            and not is_dangling_claw
            and (dot_thumb_index > 0.35 or thumb_index_parallel)
            and (downward_thumb_slope > -0.15 or is_thumb_pointing_down or is_thumb_horizontal)
            and d_thumb_index_tip <= 0.70
            and not is_P_shape
        )

        # ----------------------------------------------------------------
        # LETTER M: 3-finger forward drape (Index, Middle, Ring over thumb)
        # ----------------------------------------------------------------
        m_all_3_draped = bool(
            (lm[8][1] > lm[5][1] + 0.05 * ref_scale)
            and (lm[12][1] > lm[9][1] + 0.05 * ref_scale)
            and (lm[16][1] > lm[13][1] + 0.05 * ref_scale)
        )
        m_three_together = bool(
            d_index_middle_tip < 0.58
            and d_middle_ring_tip < 0.55
        )
        # In ASL M, the pinky is curled into the fist (never an extended 4th finger spread away)
        m_pinky_tucked = bool(
            not ext_pinky
            or (lm[20][1] < lm[16][1] - 0.08 * ref_scale)
            or (d_ring_pinky_tip <= 0.45 and m_all_3_draped)
        )

        is_M_shape = bool(
            not is_Q_shape
            and not is_P_shape
            and not (up_index or up_middle or up_ring or up_pinky)
            and not thumb_extended_out
            and not is_O_shape
            and m_pinky_tucked
            and t_thumb >= 0.45
            and (
                (m_all_3_draped and m_three_together)
                or (down_index and down_middle and down_ring and m_three_together and not (ext_pinky and d_ring_pinky_tip > 0.40))
                or (t_thumb >= 0.72 and not (ext_index or ext_middle or ext_ring or ext_pinky) and h_thumb > -0.32)
            )
        )

        # ----------------------------------------------------------------
        # C SHAPE DETECTION: All 4 fingers curved in C formation
        # In C, fingers are arched in a curved C-cup (never flat upright open palm).
        # Thumb curves opposite to form the lower jaw of C with open aperture (0.18 - 0.98).
        # Supports horizontal, upright, and arbitrarily tilted camera views.
        # ----------------------------------------------------------------
        d_index_wrist_norm = float(np.linalg.norm(lm[8] - wrist) / ref_scale)
        d_ring_wrist_norm = float(np.linalg.norm(lm[16] - wrist) / ref_scale)

        dx_idx = abs(lm[8][0] - lm[5][0])
        dy_idx = abs(lm[8][1] - lm[5][1])

        # In C, if fingers point upright to ceiling, they MUST be arched forward in a C cup.
        # When fingers are horizontal, fingers form the upper arc of the horizontal C.
        valid_c_curvature = bool(
            not (up_all_4 and straight_fingers)
            and (are_fingers_arched or horiz_index or horiz_middle or is_hand_horizontal or dx_idx > 0.35 * dy_idx)
        )

        is_C_shape = bool(
            not is_O_shape
            and not is_M_shape
            and not is_Q_shape
            and not (down_index and down_middle and down_ring and down_pinky and not horiz_index)
            and valid_c_curvature
            and (ext_index and ext_middle and (ext_ring or ext_pinky))
            and d_index_middle_tip < 0.42
            and 0.18 <= d_thumb_index_tip <= 0.98
            and d_thumb_middle_tip < 1.05
            and d_index_wrist_norm > 0.35
            and d_ring_wrist_norm > 0.26
            and d_thumb_wrist > 0.32
            and (horiz_index or horiz_middle or is_hand_horizontal or dx_idx > 0.35 * dy_idx or abs(lm[8][2] - lm[5][2]) > 0.035 or are_fingers_arched)
        )

        # ----------------------------------------------------------------
        # FLAT OPEN PALM / RANDOM SPREAD HAND:
        # Rejects open/spread hands in any orientation (upright, downward, horizontal)
        # In ASL B, thumb must be folded across palm (d_mcp <= 0.55, d_ti <= 1.25, not thumb_out).
        # In ASL C, fingers form a C-cup with thumb (0.18 <= d_ti <= 0.98).
        # Any other hand with 4 extended fingers (such as claws, open paws, high-fives)
        # is an invalid/random hand pose and must be rejected.
        # ----------------------------------------------------------------
        is_flat_open_hand = bool(
            (
                up_all_4
                and (
                    (straight_fingers and (t_thumb <= -0.25 or thumb_extended_out) and d_thumb_index_tip > 0.98)
                    or (d_thumb_index_tip > 1.25 and not is_O_shape)
                )
            )
            or (
                (ext_index and ext_middle and ext_ring and ext_pinky)
                and not is_C_shape
                and not is_O_shape
                and not is_M_shape
                and not is_P_shape
                and not is_Q_shape
                and not is_X_shape
                and not index_hooked
                and (
                    # 4 fingers pointing down together (random dangling hand / open claw)
                    (down_index and down_middle and down_ring and down_pinky)
                    # Or 4 fingers pointing horizontally together with spread fingers
                    or (horiz_index and horiz_middle and horiz_ring and horiz_pinky and (d_index_middle_tip > 0.40 or d_middle_ring_tip > 0.40 or d_ring_pinky_tip > 0.40))
                )
            )
        )

        # ----------------------------------------------------------------
        # LETTER N: 2-finger forward drape (Index + Middle extended down together)
        # Index and middle point downward together (down_index and down_middle).
        # Ring and pinky are folded/curled (not extended down, not up).
        # Index and middle are parallel and side-by-side (spread_ratio < 1.50).
        # Middle extends significantly further down than ring or ring is curled.
        # Thumb is tucked under/near index & middle (0.20 <= t_thumb < 0.58).
        # KEY GUARD: Thumb must NOT be pinched to index tip (d_thumb_index_tip > 0.35)
        # to prevent N from firing when the hand drifts slightly out of Q position.
        # ----------------------------------------------------------------
        d_thumb_im_min = min(d_thumb_index_pip, d_thumb_middle_pip, d_thumb_index_tip)
        is_N_shape = bool(
            down_index
            and down_middle
            and not up_index
            and not up_middle
            and not up_ring
            and not up_pinky
            and not is_M_shape
            and not is_flat_open_hand
            and d_index_middle_tip < 0.50
            and not (ext_pinky and d_ring_pinky_tip > 0.40)
            and not down_pinky
            and p_spread_ratio < 1.50
            and d_thumb_index_tip > 0.35           # Thumb NOT pinched to index (Q has d_ti < 0.40)
            and (
                (lm[12][1] > lm[16][1] + 0.06 * ref_scale)
                or (d_thumb_im_min < 0.85)
                or not ext_ring
                or d_middle_ring_tip > 0.25
            )
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
            and dot_thumb_index < 0.45
        )

        # ----------------------------------------------------------------
        # LETTER G: Index extended sideways/horizontally, thumb parallel along index
        # Supports both common ASL G postures:
        # 1) Natural horizontal hand (wrist at side, index extending horizontally along hand)
        # 2) Upright hand with index pointing horizontally across knuckles
        # Remaining fingers (middle, ring, pinky) are folded in a fist.
        # Not H (middle not extended horizontally alongside index).
        # Not L (thumb not perpendicular in right angle).
        # Not Q (not pointing downward).
        # Not D (index not pointing UP in D loop).
        # ----------------------------------------------------------------
        # Real upright pinky (cannot be a curled pinky inside horizontal fist)
        real_pnk_up = bool(up_pinky and straight_pnk > 0.80 and not is_hand_horizontal)

        # In ASL G: Index extended horizontally, thumb parallel, pinky must be curled into fist
        is_G_shape = bool(
            ext_index
            and straight_idx >= 0.80
            and not (ext_pinky and straight_pnk > 0.80)
            and not (up_middle or up_ring or real_pnk_up)
            and not (ext_middle and (horiz_middle or d_index_middle_tip < 0.35))
            and not (is_Q_shape or is_hand_pointing_down or (down_index and not horiz_index and not is_hand_horizontal))
            and not is_L_shape
            and not (up_index and is_D_loop and d_thumb_index_tip > 0.55)
            and (
                horiz_index
                or is_hand_horizontal
                or thumb_index_parallel
                or (dot_thumb_index > 0.20 and not up_index)
                or (horiz_index and dot_thumb_index > 0.15)
                or (is_hand_horizontal and dot_thumb_index > 0.15)
            )
            and d_thumb_index_tip <= 0.90
        )

        # ----------------------------------------------------------------
        # PINKY TILT & ORIENTATION (For I vs J)
        # ----------------------------------------------------------------
        v_pinky_vec = lm[20] - wrist
        is_pinky_tilted = bool(abs(v_pinky_vec[0]) > 0.35 * abs(v_pinky_vec[1]))
        is_hand_tilted = bool(abs(v_long[0]) > 0.35 * abs(v_long[1]))

        dx_pnk_tip_mcp = abs(lm[20][0] - lm[17][0])
        dy_pnk_tip_mcp = abs(lm[20][1] - lm[17][1])

        # Fingers 1-3 curled in fist (not extended straight)
        fingers_123_curled = bool(
            (not ext_index or straight_idx < 0.80)
            and (not ext_middle or straight_mid < 0.80)
            and (not ext_ring or straight_rng < 0.80)
            and not (up_index or up_middle or up_ring)
        )

        # In ASL I: Pinky points strictly UP (vertical) on an upright hand
        is_pinky_vertical_up = bool(
            up_pinky
            and dy_pnk_tip_mcp > 1.20 * dx_pnk_tip_mcp
            and not horiz_pinky
            and not is_hand_horizontal
        )

        # In ASL J: Pinky points SIDEWAYS / HORIZONTALLY or tilted in J pose
        is_pinky_sideways_j = bool(
            (horiz_pinky or dx_pnk_tip_mcp > 0.65 * dy_pnk_tip_mcp or is_pinky_tilted or is_hand_horizontal)
            and not is_pinky_vertical_up
        )

        # ----------------------------------------------------------------
        # LETTER J: Pinky extended sideways/horizontally, other fingers curled in fist
        # ----------------------------------------------------------------
        is_J_shape = bool(
            ext_pinky
            and straight_pnk >= 0.80
            and fingers_123_curled
            and not thumb_extended_out
            and not is_G_shape
            and is_pinky_sideways_j
        )

        # ----------------------------------------------------------------
        # LETTER I: Pinky extended vertically UP, other fingers curled in fist
        # ----------------------------------------------------------------
        is_I_shape = bool(
            ext_pinky
            and straight_pnk >= 0.80
            and fingers_123_curled
            and not thumb_extended_out
            and not is_G_shape
            and not is_J_shape
            and is_pinky_vertical_up
        )

        # ----------------------------------------------------------------
        # THUMB DISTAL PHALANX GEOMETRY (Thumb IP -> Thumb TIP)
        # Distinguishes S (thumb wrapping horizontally across front) from
        # T (thumb tip poked UP between index and middle fingers)
        # ----------------------------------------------------------------
        v_thumb_distal = thumb_tip - lm[3]
        dx_thumb_distal = float(abs(v_thumb_distal[0]))
        dy_thumb_distal = float(abs(v_thumb_distal[1]))
        is_thumb_distal_horizontal = bool(dx_thumb_distal > 0.70 * dy_thumb_distal or dx_thumb_distal > 0.08 * ref_scale)
        is_thumb_distal_pointing_up = bool(v_thumb_distal[1] * sign_y > 0.015 * ref_scale and dy_thumb_distal > 1.20 * dx_thumb_distal)

        # Knuckle span and orientation (MCP 5 Index to MCP 17 Pinky)
        dx_knuckles = float(abs(lm[17][0] - lm[5][0]))
        dy_knuckles = float(abs(lm[17][1] - lm[5][1]))
        is_knuckles_horizontal = bool(dx_knuckles > 1.15 * dy_knuckles)

        all_fingers_curled = bool(
            not (up_index or up_middle or up_ring or up_pinky)
            and not (ext_index or ext_middle or ext_ring or ext_pinky)
            and not is_C_shape
            and not is_O_shape
            and not thumb_extended_out
        )

        # LETTER S: Closed fist with thumb wrapped HORIZONTALLY across front of fingers
        # Canonical 3D alignment for rotation-invariant analysis
        canonical_lm, R_canonical = self.canonical_align(lm, handedness=handedness)

        # In canonical space, X is the transverse axis (Index MCP > 0, Middle MCP = 0, Pinky MCP < 0).
        # t_pip_canonical measures where thumb tip sits: 0.0 at Index PIP, 1.0 at Middle PIP.
        idx_pip_x = float(canonical_lm[6][0])
        mid_pip_x = float(canonical_lm[10][0])
        pip_span_c = idx_pip_x - mid_pip_x
        t_pip_canonical = float((idx_pip_x - canonical_lm[4][0]) / (pip_span_c + 1e-5))
        d_ti_pip_c = float(np.linalg.norm(canonical_lm[4] - canonical_lm[6]))
        d_tm_pip_c = float(np.linalg.norm(canonical_lm[4] - canonical_lm[10]))

        # LETTER S: Closed fist with thumb wrapped HORIZONTALLY across front of fingers
        is_S_shape = bool(
            all_fingers_curled
            and 0.14 <= t_thumb <= 0.45
            and is_thumb_distal_horizontal
            and not is_thumb_distal_pointing_up
            and h_thumb > -0.32
        )

        # LETTER T: Closed fist with thumb tip poked UP between index and middle fingers
        is_T_canonical = bool(
            all_fingers_curled
            and 0.12 <= t_pip_canonical <= 0.88
            and min(d_ti_pip_c, d_tm_pip_c) < 0.35
            and canonical_lm[4][1] >= canonical_lm[5][1] - 0.15
            and not is_S_shape
        )

        is_T_shape = bool(
            is_T_canonical
            or (
                all_fingers_curled
                and 0.10 <= t_thumb <= 0.35
                and is_thumb_distal_pointing_up
                and not is_thumb_distal_horizontal
                and not is_S_shape
            )
        )

        # LETTER E: Curled fist with thumb tucked underneath fingertips across palm
        is_E_shape = bool(
            all_fingers_curled
            and not is_S_shape
            and not is_T_shape
            and not is_M_shape
            and straight_idx <= 0.35
            and h_thumb <= -0.08
            and (
                t_thumb <= 0.20
                or (h_thumb <= -0.32 and t_thumb >= 0.50)
            )
        )

        return {
            # Canonical 3D representations
            "landmarks_canonical": canonical_lm,
            "canonical_rotation": R_canonical,
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
            "horiz_ring": horiz_ring,
            "horiz_pinky": horiz_pinky,
            "down_index": down_index,
            "down_middle": down_middle,
            "down_ring": down_ring,
            "down_pinky": down_pinky,
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
            "is_R_shape": is_R_shape,
            "is_U_shape": is_U_shape,
            "is_V_shape": is_V_shape,
            "u_v_spread_ratio": u_v_spread_ratio,
            "index_hooked": index_hooked,
            "is_O_shape": is_O_shape,
            "is_C_shape": is_C_shape,
            "is_flat_open_hand": is_flat_open_hand,
            "straight_idx": straight_idx,
            "straight_mid": straight_mid,
            "straight_rng": straight_rng,
            "straight_pnk": straight_pnk,
            "are_fingers_arched": are_fingers_arched,
            "is_D_loop": is_D_loop,
            "idx_tip_elevation": idx_tip_elevation,
            "d_index_tip_mcp": d_index_tip_mcp,
            "is_G_shape": is_G_shape,
            "is_I_shape": is_I_shape,
            "is_J_shape": is_J_shape,
            "is_L_shape": is_L_shape,
            "is_M_shape": is_M_shape,
            "is_N_shape": is_N_shape,
            "is_X_shape": is_X_shape,
            "index_hooked": index_hooked,
            "is_P_shape": is_P_shape,
            "p_spread_ratio": p_spread_ratio,
            "downward_mid_slope": downward_mid_slope,
            "is_Q_shape": is_Q_shape,
            "downward_idx_slope": downward_idx_slope,
            "downward_thumb_slope": downward_thumb_slope,
            "is_S_shape": is_S_shape,
            "is_T_shape": is_T_shape,
            "t_pip_canonical": t_pip_canonical,
            "is_E_shape": is_E_shape,
            "is_thumb_distal_horizontal": is_thumb_distal_horizontal,
            "is_thumb_distal_pointing_up": is_thumb_distal_pointing_up,
            "is_canonical": is_canonical,
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
            # Knuckle orientation
            "dx_knuckles": dx_knuckles,
            "dy_knuckles": dy_knuckles,
            "is_knuckles_horizontal": is_knuckles_horizontal,
            # Pinky flags
            "is_pinky_tilted": is_pinky_tilted,
            "dx_pnk": float(dx_pnk_tip_mcp),
            "dy_pnk": float(dy_pnk_tip_mcp),
            "ref_scale": ref_scale,
        }
