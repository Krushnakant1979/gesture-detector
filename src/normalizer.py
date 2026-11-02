"""
Geometric Landmark Normalizer and Anatomical Analyzer
Extracts rotation-invariant and handedness-invariant metrics for ASL alphabet recognition.

=== HAND GESTURE ORIENTATION REFERENCE (from official ASL charts) ===

UPRIGHT / VERTICAL (wrist at bottom, fingers pointing UP, image Y decreases going up):
  A  - Closed fist, thumb rests on SIDE of index pointing slightly UP (t<0.20, h>0.02)
  B  - All 4 fingers straight UP together, thumb folded across palm
  C  - All 4 fingers curved in C-cup shape, thumb curves opposite; hand open
  D  - Only index finger UP, middle+ring+pinky loop forward to touch thumb tip
  E  - All 4 fingers bent/hooked at knuckle, thumb pressed under bent fingertips
  F  - Middle+ring+pinky UP, index tip touches thumb tip forming small circle
  I  - Only pinky UP, all others curled, thumb folded
  J  - Only pinky UP (+ J sweeping motion downward) - same static as I
  K  - Index+middle both UP in V shape, thumb pointing UP between them
  L  - Index finger UP, thumb extended HORIZONTALLY at 90 degrees (L shape)
  M  - Closed fist, thumb tucked UNDER 3 fingers (index+middle+ring drape over it)
  N  - Closed fist, thumb tucked UNDER 2 fingers (index+middle drape over it)
  O  - All fingers arch forward, tips converge with thumb forming O/circle
  R  - Index+middle both UP, CROSSED over each other (index on top of middle)
  S  - Closed fist, thumb wraps HORIZONTALLY ACROSS front of all 4 fingers
  T  - Closed fist, thumb poked UP between index and middle (tip shows between them)
  U  - Index+middle both UP and CLOSE TOGETHER (parallel, side by side)
  V  - Index+middle both UP and SPREAD APART (wider V shape)
  W  - Index+middle+ring all UP and spread apart
  X  - Only index, with HOOK shape: PIP joint extends up, tip bends back DOWN
  Y  - Thumb and pinky both extended OUT (hang-loose / shaka), others curled
  Z  - Index UP alone (tracing Z motion), others curled, thumb tucked

HORIZONTAL / SIDEWAYS (hand axis points sideways, palm faces camera):
  G  - Index pointing SIDEWAYS, thumb parallel alongside index (downward G = Q)
  H  - Index+middle both pointing SIDEWAYS together, palm faces camera

POINTING DOWN (wrist at top, fingers pointing DOWN):
  P  - Index+middle pointing DOWN, thumb perpendicular out to the side
  Q  - Index pointing DOWN + thumb pointing DOWN alongside (downward version of G)
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


class LandmarkNormalizer:
    """Performs geometric normalization and anatomical feature extraction on 21 hand landmarks."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

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

    @staticmethod
    def analyze_hand(lm: np.ndarray, handedness: str = "Right") -> Dict[str, any]:
        """
        Comprehensive 3D anatomical analysis for ASL fingerspelling.
        Handles ALL 26 letter orientations: upright, horizontal/sideways, and downward-pointing.

        MediaPipe landmark indices:
          0=Wrist, 1-4=Thumb(CMC,MCP,IP,TIP), 5-8=Index(MCP,PIP,DIP,TIP),
          9-12=Middle(MCP,PIP,DIP,TIP), 13-16=Ring(MCP,PIP,DIP,TIP), 17-20=Pinky(MCP,PIP,DIP,TIP)

        Image coordinate system: Y increases DOWNWARDS. So 'above' = smaller Y value.
        """
        wrist = lm[0]

        # Scale reference: distance from wrist(0) to middle MCP(9)
        ref_scale = float(np.linalg.norm(lm[9] - wrist))
        if ref_scale < 1e-4:
            ref_scale = 1.0

        # Longitudinal hand vector: Wrist(0) -> Middle MCP(9)
        v_long = lm[9] - wrist
        norm_long = float(np.linalg.norm(v_long))
        u_long = v_long / (norm_long + 1e-5)  # unit vector along palm axis

        # Transverse palm vector: Index MCP(5) -> Pinky MCP(17)
        v_knuckles = lm[17] - lm[5]
        knuckle_span = float(np.linalg.norm(v_knuckles))
        knuckle_span_sq = float(np.dot(v_knuckles, v_knuckles))
        if knuckle_span_sq < 1e-5:
            knuckle_span_sq = 1e-5

        # ----------------------------------------------------------------
        # GLOBAL HAND ORIENTATION
        # ----------------------------------------------------------------
        # In image coords, v_long[1] < 0 means hand points UP (vertical).
        # v_long[1] > 0 means hand points DOWN (inverted, e.g. P, Q).
        # abs(v_long[0]) large relative to abs(v_long[1]) means HORIZONTAL (G, H).

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
        # Returns: (is_extended, is_up, is_horiz, is_down, ratio)
        # ----------------------------------------------------------------
        def check_finger(tip_idx: int, pip_idx: int, mcp_idx: int) -> Tuple[bool, bool, bool, bool, float]:
            tip = lm[tip_idx]
            pip = lm[pip_idx]
            mcp = lm[mcp_idx]

            d_tip_mcp = float(np.linalg.norm(tip - mcp))
            d_pip_mcp = float(np.linalg.norm(pip - mcp))
            ratio = float(d_tip_mcp / (d_pip_mcp + 1e-5))

            # Extended if tip is significantly far from knuckle (more than half a ref_scale unit)
            is_ext_len = bool(d_tip_mcp > 0.42 * ref_scale)

            # UP: tip is above MCP by at least 15% of hand scale
            is_up = bool(is_ext_len and tip[1] < mcp[1] - 0.15 * ref_scale)

            # DOWN: tip is below MCP by at least 15% of hand scale (P, Q gesture)
            is_down = bool(is_ext_len and tip[1] > mcp[1] + 0.15 * ref_scale)

            # HORIZONTAL: tip extends sideways more than it extends vertically (G, H)
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
        thumb_cmc = lm[1]

        v_thumb = thumb_tip - thumb_mcp
        norm_thumb = float(np.linalg.norm(v_thumb))

        # Transverse thumb position: 0.0 = at index MCP, 1.0 = at pinky MCP
        t_thumb = float(np.dot(thumb_tip - lm[5], v_knuckles) / knuckle_span_sq)

        # Longitudinal height: how far thumb extends past index MCP along hand axis
        h_thumb = float(np.dot(thumb_tip - lm[5], u_long) / ref_scale)

        d_thumb_index_mcp = float(np.linalg.norm(thumb_tip - lm[5]) / ref_scale)
        d_thumb_middle_mcp = float(np.linalg.norm(thumb_tip - lm[9]) / ref_scale)
        d_thumb_wrist = float(np.linalg.norm(thumb_tip - wrist) / ref_scale)

        # Thumb direction flags:
        # UP: thumb vector points strongly upward (negative Y in image coords)
        is_thumb_pointing_up = bool(
            v_thumb[1] < -0.04 * ref_scale
            and abs(v_thumb[1]) > 0.55 * abs(v_thumb[0])
        )
        # HORIZONTAL: thumb vector is mostly horizontal
        is_thumb_horizontal = bool(abs(v_thumb[0]) > 0.85 * abs(v_thumb[1]))
        # DOWN: thumb vector points downward
        is_thumb_pointing_down = bool(
            v_thumb[1] > 0.04 * ref_scale
            and abs(v_thumb[1]) > 0.55 * abs(v_thumb[0])
        )

        # Extended wide (for L and Y): thumb tip is far from index MCP and from wrist
        thumb_extended_out = bool(d_thumb_index_mcp > 0.42 and d_thumb_wrist > 0.65)

        # ----------------------------------------------------------------
        # LETTER K: Thumb pointing UP between index and middle fingers
        # The thumb tip must be spatially between the index MCP (5) and middle MCP (9)
        # and at about the level of the PIP joints or higher.
        # ----------------------------------------------------------------
        # X-coordinate boundaries of index MCP and middle MCP (with tolerance)
        x_left = min(lm[5][0], lm[9][0]) - 0.08 * ref_scale
        x_right = max(lm[5][0], lm[9][0]) + 0.08 * ref_scale
        is_thumb_between_index_middle = bool(
            is_thumb_pointing_up
            and x_left <= thumb_tip[0] <= x_right
            and thumb_tip[1] < lm[5][1] + 0.10 * ref_scale  # at knuckle level or higher
        )

        # ----------------------------------------------------------------
        # VECTOR ALIGNMENT: Index finger vs Thumb (for G and L)
        # ----------------------------------------------------------------
        v_index_seg = lm[8] - lm[5]
        norm_idx = float(np.linalg.norm(v_index_seg))
        u_index_seg = v_index_seg / (norm_idx + 1e-5)

        u_thumb_seg = v_thumb / (norm_thumb + 1e-5)

        # dot > 0 means thumb and index point in same general direction
        dot_thumb_index = float(np.dot(u_index_seg[:2], u_thumb_seg[:2]))

        # G: hand sideways, index and thumb both point in same lateral direction
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
        # LETTER R: Index and Middle CROSSED (both upright, finger X-coords crossed)
        # When crossed: tip X of index is on the OPPOSITE side of tip X of middle
        # relative to their MCPs. Must be upright, not horizontal, not K.
        # ----------------------------------------------------------------
        # In standard R (non-mirrored): index tip is to the left of middle tip,
        # but index MCP is to the right of middle MCP (or vice versa).
        crossed_by_x = bool((lm[8][0] - lm[12][0]) * (lm[5][0] - lm[9][0]) < 0)
        are_index_middle_crossed = bool(
            up_index and up_middle
            and not is_hand_horizontal
            and not is_thumb_between_index_middle  # K uses same finger config
            and (crossed_by_x or d_index_middle_tip < 0.09)
        )

        # ----------------------------------------------------------------
        # LETTER X: Index PIP extends HIGH up, tip HOOKS back DOWN
        # Distinctive: In a fist, PIP stays near knuckle.
        # In X, PIP extends above knuckle but tip bends back.
        # ----------------------------------------------------------------
        d_index_tip_mcp = float(np.linalg.norm(lm[8] - lm[5]) / ref_scale)
        pip_elevated = bool(lm[6][1] < lm[5][1] - 0.15 * ref_scale)  # PIP high above knuckle
        tip_hooked_down = bool(lm[8][1] > lm[6][1])                   # tip below PIP
        index_hooked = bool(
            not ext_index           # Tip is NOT classed as fully extended
            and pip_elevated        # But PIP IS elevated (finger bending at first joint)
            and tip_hooked_down     # Tip hooks back downward
            and d_index_tip_mcp > 0.22
            and d_thumb_index_tip > 0.32  # Thumb doesn't touch tip (would be O/F/D)
            and not ext_middle and not ext_ring and not ext_pinky
        )

        # ----------------------------------------------------------------
        # LETTER O: All fingertips converge to meet thumb tip in an O circle
        # Key properties:
        #  - Thumb tip is very close to index tip AND middle tip (tight cluster)
        #  - Ring tip is also nearby
        #  - Finger knuckles (MCPs) are far from wrist (fingers extended out, not fisted)
        #  - Pinky is not shooting straight up
        # ----------------------------------------------------------------
        d_mcp5_wrist = float(np.linalg.norm(lm[5] - wrist) / ref_scale)
        d_index_wrist = float(np.linalg.norm(lm[8] - wrist) / ref_scale)
        d_middle_wrist = float(np.linalg.norm(lm[12] - wrist) / ref_scale)
        is_O_shape = bool(
            d_thumb_index_tip < 0.16         # Thumb meets index closely
            and d_thumb_middle_tip < 0.20    # Thumb meets middle closely
            and d_thumb_ring_tip < 0.30      # Ring is in the O group
            and d_mcp5_wrist > 0.65          # Fingers are extended (not collapsed fist)
            and not up_pinky                 # Pinky not shooting up (would be Y)
        )

        # ----------------------------------------------------------------
        # LETTER D: Index UP alone, middle+ring+pinky arc forward to touch thumb
        # The "D loop": middle finger tip close to thumb tip AND middle far from wrist
        # ----------------------------------------------------------------
        is_D_loop = bool(
            d_thumb_middle_tip < 0.35       # middle curves to touch thumb
            and d_middle_wrist > 0.40       # middle finger is extended, not collapsed
        )

        # ----------------------------------------------------------------
        # C SHAPE DETECTION: All 4 fingers curved in C formation
        # In C, fingers are partially curled (not straight up, not fisted).
        # The thumb and index tips form a wide C gap.
        # Key: thumb is out wide (extended), d_thumb_index_tip is moderate (0.35-0.80)
        # ----------------------------------------------------------------
        # Fingers in C are extended (tip far from MCP) but not fully upright.
        # We check by: each finger tip is far from wrist AND from each other
        d_index_wrist_norm = float(np.linalg.norm(lm[8] - wrist) / ref_scale)
        d_ring_wrist_norm = float(np.linalg.norm(lm[16] - wrist) / ref_scale)
        is_C_shape = bool(
            not is_O_shape               # Not O
            and not up_index             # Fingers are not straight upright
            and d_thumb_index_tip > 0.30 # Thumb is not touching index (not closed)
            and d_index_wrist_norm > 0.60 # Index tip is far from wrist (extended)
            and d_ring_wrist_norm > 0.45  # Ring tip is also extended
            and thumb_extended_out        # Thumb curves out wide
        )

        # ----------------------------------------------------------------
        # PINKY TILT: For J vs I discrimination
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
            "thumb_extended_out": thumb_extended_out,
            "thumb_index_parallel": thumb_index_parallel,
            "dot_thumb_index": dot_thumb_index,
            # Complex shape detectors
            "are_index_middle_crossed": are_index_middle_crossed,
            "index_hooked": index_hooked,
            "is_O_shape": is_O_shape,
            "is_C_shape": is_C_shape,
            "is_D_loop": is_D_loop,
            # Inter-tip distances (normalized by ref_scale)
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
