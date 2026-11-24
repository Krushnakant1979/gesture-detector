"""
ASL Gesture Recognition Studio - Unit Tests
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict, List, Optional
import numpy as np

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.classifier import GestureClassifier
from src.filters import (
    ExponentialMovingAverage,
    GestureDebouncer,
    HandMotionDetector,
    OneEuroFilter,
    StrokeTrajectoryTracker,
    WordPredictor,
)
from src.gesture_event_manager import EventType, GestureEventManager, SystemState
from src.lstm_recognizer import LSTMRecurrentClassifier
from src.normalizer import LandmarkNormalizer
from src.tracker import HandData, HandTracker
from src.visualizer import ModernVisualizer


def test_k_vs_v_distinction():
    """Verifies that K (elevated thumb between fingers) is reliably separated from V (folded thumb)."""
    classifier = GestureClassifier(mode="heuristic")

    def make_2finger_hand(cls: str) -> np.ndarray:
        vec = np.zeros((21, 3), dtype=np.float32)
        vec[0] = [0, 0, 0]
        vec[1] = [-0.15, -0.2, 0.0]
        vec[2] = [-0.25, -0.35, 0.0]
        vec[3] = [-0.3, -0.45, 0.0]
        vec[5] = [-0.15, -0.5, 0.0]
        vec[9] = [0.0, -0.52, 0.0]
        vec[13] = [0.15, -0.5, 0.0]
        vec[17] = [0.28, -0.45, 0.0]
        vec[16] = [0.15, -0.35, 0.1]
        vec[20] = [0.25, -0.35, 0.1]
        # Extended index & middle
        vec[6] = [-0.16, -0.65, 0.0]
        vec[7] = [-0.18, -0.80, 0.0]
        vec[8] = [-0.20, -0.95, 0.0]
        vec[10] = [0.10, -0.65, 0.0]
        vec[11] = [0.15, -0.80, 0.0]
        vec[12] = [0.20, -0.95, 0.0]
        if cls == "K":
            # Thumb elevated between fingers
            vec[4] = [0.0, -0.70, 0.05]
        else:
            # Thumb folded across palm (V)
            vec[4] = [0.05, -0.4, 0.1]
        return vec

    k_letter, _ = classifier.classify(make_2finger_hand("K"))
    assert k_letter == "K"

    v_letter, _ = classifier.classify(make_2finger_hand("V"))
    assert v_letter == "V"




def test_strict_a_gesture():
    """Verifies that only an upright snug thumb produces 'A', and hitchhiker thumbs are rejected."""
    classifier = GestureClassifier(mode="heuristic")

    def make_fist_variant(thumb_mode: str) -> np.ndarray:
        vec = np.zeros((21, 3), dtype=np.float32)
        vec[0] = [0, 0, 0]
        vec[5] = [-0.15, -0.5, 0.0]
        vec[9] = [0.0, -0.52, 0.0]
        vec[13] = [0.15, -0.5, 0.0]
        vec[17] = [0.28, -0.45, 0.0]
        for mcp_idx, tip_idx in [(5, 8), (9, 12), (13, 16), (17, 20)]:
            vec[mcp_idx + 1] = [vec[mcp_idx][0], -0.60, 0.0]
            vec[mcp_idx + 2] = [vec[mcp_idx][0], -0.52, 0.05]
            vec[tip_idx] = [vec[mcp_idx][0], -0.42, 0.08]
        vec[1] = [-0.15, -0.2, 0.0]
        vec[2] = [-0.25, -0.35, 0.0]
        vec[3] = [-0.25, -0.48, 0.0]
        if thumb_mode == "upright_snug":  # True A sign
            vec[4] = [-0.24, -0.62, 0.0]
        else:  # Thumb sticking out wide horizontally (hitchhiker, Image 1)
            vec[3] = [-0.45, -0.37, 0.0]
            vec[4] = [-0.60, -0.38, 0.0]
        return vec

    true_a, conf_a = classifier.classify(make_fist_variant("upright_snug"))
    assert true_a == "A"
    assert conf_a >= 0.95

    wrong_pose, _ = classifier.classify(make_fist_variant("thumb_out_horizontal"))
    assert wrong_pose == "None"




def test_c_vs_open_palm_distinction():
    """Verifies that only an arched C-cup gesture produces 'C', and flat open palms are rejected."""
    classifier = GestureClassifier(mode="heuristic")

    # Image 1 style: flat open palm (5 / high five)
    open_palm = np.zeros((21, 3), dtype=np.float32)
    open_palm[0] = [0, 0, 0]
    open_palm[1] = [0.15, -0.2, 0.0]
    open_palm[2] = [0.25, -0.35, 0.0]
    open_palm[3] = [0.45, -0.40, 0.0]
    open_palm[4] = [0.65, -0.45, 0.0]
    open_palm[5] = [0.15, -0.5, 0.0]
    open_palm[9] = [0.0, -0.52, 0.0]
    open_palm[13] = [-0.15, -0.5, 0.0]
    open_palm[17] = [-0.28, -0.45, 0.0]
    open_palm[8] = [0.25, -0.95, 0.0]
    open_palm[12] = [0.0, -0.98, 0.0]
    open_palm[16] = [-0.25, -0.95, 0.0]
    open_palm[20] = [-0.40, -0.90, 0.0]
    open_palm[6] = [0.20, -0.7, 0.0]
    open_palm[10] = [0.0, -0.75, 0.0]
    open_palm[14] = [-0.20, -0.7, 0.0]
    open_palm[18] = [-0.35, -0.65, 0.0]

    res_open, _ = classifier.classify(open_palm)
    assert res_open == "None", f"Open palm should be rejected, got {res_open}"

    # Image 2 style: true curved C cup
    true_c = np.zeros((21, 3), dtype=np.float32)
    true_c[0] = [0, 0, 0]
    true_c[5] = [0.40, -0.30, 0.0]
    true_c[9] = [0.42, -0.25, 0.0]
    true_c[13] = [0.40, -0.20, 0.0]
    true_c[17] = [0.38, -0.15, 0.0]
    true_c[6] = [0.55, -0.35, 0.0]
    true_c[7] = [0.65, -0.32, 0.0]
    true_c[8] = [0.70, -0.28, 0.0]
    true_c[10] = [0.58, -0.30, 0.0]
    true_c[11] = [0.68, -0.27, 0.0]
    true_c[12] = [0.72, -0.24, 0.0]
    true_c[14] = [0.55, -0.25, 0.0]
    true_c[15] = [0.65, -0.22, 0.0]
    true_c[16] = [0.70, -0.19, 0.0]
    true_c[18] = [0.50, -0.20, 0.0]
    true_c[19] = [0.60, -0.17, 0.0]
    true_c[20] = [0.65, -0.14, 0.0]
    true_c[1] = [0.15, -0.05, 0.0]
    true_c[2] = [0.30, 0.05, 0.0]
    true_c[3] = [0.50, 0.10, 0.0]
    true_c[4] = [0.68, 0.08, 0.0]

    res_c, conf_c = classifier.classify(true_c)
    assert res_c == "C", f"True C should be classified as C, got {res_c}"
    assert conf_c >= 0.95

    # 3. User C screenshot gesture
    w_c, h_c = 775.0, 547.0
    user_c = np.zeros((21, 3), dtype=np.float32)
    user_c[0] = [531.2 / w_c, 385.2 / h_c, 0]
    user_c[1] = [437.8 / w_c, 391.5 / h_c, 0]
    user_c[2] = [338.8 / w_c, 372.7 / h_c, 0]
    user_c[3] = [267.0 / w_c, 372.8 / h_c, 0]
    user_c[4] = [232.5 / w_c, 339.6 / h_c, 0]
    user_c[5] = [341.9 / w_c, 222.9 / h_c, 0]
    user_c[6] = [300.0 / w_c, 119.1 / h_c, 0]
    user_c[7] = [236.5 / w_c, 155.0 / h_c, 0]
    user_c[8] = [200.8 / w_c, 182.9 / h_c, 0]
    user_c[9] = [376.5 / w_c, 206.2 / h_c, 0]
    user_c[10] = [314.8 / w_c, 125.0 / h_c, 0]
    user_c[11] = [262.4 / w_c, 128.9 / h_c, 0]
    user_c[12] = [223.2 / w_c, 166.3 / h_c, 0]
    user_c[13] = [417.1 / w_c, 197.6 / h_c, 0]
    user_c[14] = [358.5 / w_c, 115.1 / h_c, 0]
    user_c[15] = [300.0 / w_c, 119.1 / h_c, 0]
    user_c[16] = [256.6 / w_c, 152.8 / h_c, 0]
    user_c[17] = [457.3 / w_c, 204.9 / h_c, 0]
    user_c[18] = [391.5 / w_c, 152.5 / h_c, 0]
    user_c[19] = [336.2 / w_c, 143.8 / h_c, 0]
    user_c[20] = [291.0 / w_c, 152.8 / h_c, 0]

    res_user_c, conf_user_c = classifier.classify(user_c, handedness="Left")
    assert res_user_c == "C", f"User C gesture should classify as C, got {res_user_c}"
    assert conf_user_c >= 0.95

    # Right-hand parity
    user_c_right = np.copy(user_c)
    user_c_right[:, 0] = 1.0 - user_c_right[:, 0]
    res_user_c_r, _ = classifier.classify(user_c_right, handedness="Right")
    assert res_user_c_r == "C", f"Right hand C should classify as C, got {res_user_c_r}"

    # 4. User Image 1: Real-world flat open hand / random gesture with upright fingers and abducted thumb
    image1_flat_palm = np.array([
        [0.863215, 0.932129, 0.000000],
        [0.803975, 0.880369, -0.029401],
        [0.768474, 0.759819, -0.041055],
        [0.757461, 0.643194, -0.049767],
        [0.736553, 0.556257, -0.056540],
        [0.793513, 0.594630, -0.012122],
        [0.787591, 0.457019, -0.027294],
        [0.787344, 0.376112, -0.041876],
        [0.788217, 0.307354, -0.052779],
        [0.826407, 0.583662, -0.009680],
        [0.818124, 0.430069, -0.022419],
        [0.814646, 0.333949, -0.035727],
        [0.811243, 0.253432, -0.045178],
        [0.857988, 0.601528, -0.011870],
        [0.851488, 0.453440, -0.025925],
        [0.847798, 0.367221, -0.036015],
        [0.844407, 0.295509, -0.042508],
        [0.892556, 0.640817, -0.017190],
        [0.892527, 0.529070, -0.029440],
        [0.891556, 0.459799, -0.034927],
        [0.891534, 0.397410, -0.038205],
    ], dtype=np.float32)
    res_img1, conf_img1 = classifier.classify(image1_flat_palm, handedness="Left")
    assert res_img1 == "None", f"Image 1 flat hand should be rejected as None, got {res_img1} ({conf_img1})"




def test_strict_b_gesture():
    """Verifies that all 4 fingers straight UP together with thumb tucked detects B, rejecting open palms."""
    classifier = GestureClassifier(mode="heuristic")

    w, h = 527.0, 677.0
    user_b = np.zeros((21, 3), dtype=np.float32)
    user_b[0] = [299.5/w, 619.1/h, 0]
    user_b[1] = [200.4/w, 586.7/h, 0]
    user_b[2] = [126.9/w, 510.2/h, 0]
    user_b[3] = [111.8/w, 435.4/h, 0]
    user_b[4] = [150.8/w, 415.0/h, 0]
    user_b[5] = [177.8/w, 349.1/h, 0]
    user_b[6] = [171.9/w, 237.7/h, 0]
    user_b[7] = [173.6/w, 168.5/h, 0]
    user_b[8] = [178.1/w, 105.5/h, 0]
    user_b[9] = [242.7/w, 339.2/h, 0]
    user_b[10] = [228.9/w, 210.1/h, 0]
    user_b[11] = [226.1/w, 131.9/h, 0]
    user_b[12] = [221.7/w, 57.1/h, 0]
    user_b[13] = [305.1/w, 355.7/h, 0]
    user_b[14] = [294.9/w, 237.6/h, 0]
    user_b[15] = [290.2/w, 164.1/h, 0]
    user_b[16] = [285.8/w, 95.5/h, 0]
    user_b[17] = [374.5/w, 392.9/h, 0]
    user_b[18] = [374.5/w, 306.4/h, 0]
    user_b[19] = [374.4/w, 249.6/h, 0]
    user_b[20] = [369.8/w, 191.8/h, 0]

    res_b, conf_b = classifier.classify(user_b, handedness="Left")
    assert res_b == "B", f"User B gesture should classify as B, got {res_b}"
    assert conf_b >= 0.95

    # Thumb sticking out laterally should NOT classify as B
    b_thumb_out = np.copy(user_b)
    b_thumb_out[4] = [-0.1, 415.0/h, 0]
    b_thumb_out[3] = [-0.05, 435.0/h, 0]
    res_thumb_out, _ = classifier.classify(b_thumb_out, handedness="Left")
    assert res_thumb_out == "None", f"B with thumb out should be None, got {res_thumb_out}"

    # 3. User random gesture from media_1793875796929_b4fa751a.jpg (4 fingers up, thumb sticking out)
    # Must NEVER be classified as B, and must return None
    user_b_random = np.array([
        [715.5/1024.0, 343.9/576.0, 0.0],
        [665.0/1024.0, 344.6/576.0, 0.0],
        [617.9/1024.0, 336.7/576.0, 0.0],
        [584.5/1024.0, 344.5/576.0, 0.0],
        [553.9/1024.0, 356.2/576.0, 0.0],
        [597.8/1024.0, 250.3/576.0, 0.0],
        [565.7/1024.0, 201.7/576.0, 0.0],
        [542.5/1024.0, 177.0/576.0, 0.0],
        [520.1/1024.0, 157.8/576.0, 0.0],
        [614.8/1024.0, 242.1/576.0, 0.0],
        [582.7/1024.0, 184.3/576.0, 0.0],
        [558.7/1024.0, 150.8/576.0, 0.0],
        [536.2/1024.0, 124.3/576.0, 0.0],
        [637.9/1024.0, 246.6/576.0, 0.0],
        [608.0/1024.0, 180.0/576.0, 0.0],
        [588.5/1024.0, 156.3/576.0, 0.0],
        [565.3/1024.0, 129.8/576.0, 0.0],
        [666.7/1024.0, 263.5/576.0, 0.0],
        [654.6/1024.0, 225.5/576.0, 0.0],
        [635.7/1024.0, 203.5/576.0, 0.0],
        [613.9/1024.0, 186.6/576.0, 0.0],
    ], dtype=np.float32)
    clf_hybrid = GestureClassifier(mode="hybrid")
    res_rand, conf_rand = clf_hybrid.classify(user_b_random, handedness="Left")
    assert res_rand == "None", f"User random gesture with thumb out should be None, got {res_rand} ({conf_rand:.2f})"




def test_strict_d_gesture():
    """Verifies that index up with middle-thumb loop detects D in hybrid mode, rejecting T and fists."""
    classifier = GestureClassifier(mode="hybrid")

    # Real-world User D screenshot landmarks (media_1793873847865_1880492d.jpg)
    user_d = np.array([
        [0.830499, 0.769031, -0.000000],
        [0.765715, 0.702835, -0.036154],
        [0.731700, 0.615020, -0.051207],
        [0.730261, 0.532758, -0.063544],
        [0.755124, 0.487184, -0.072615],
        [0.750060, 0.472632, -0.017132],
        [0.742304, 0.328784, -0.047385],
        [0.735376, 0.239449, -0.074696],
        [0.732757, 0.166613, -0.095475],
        [0.793401, 0.465041, -0.011992],
        [0.782962, 0.352027, -0.044290],
        [0.786150, 0.433169, -0.061976],
        [0.789730, 0.497713, -0.069073],
        [0.837426, 0.474187, -0.012897],
        [0.835212, 0.363061, -0.043092],
        [0.831126, 0.440004, -0.047620],
        [0.825855, 0.504023, -0.044265],
        [0.877138, 0.496242, -0.017615],
        [0.881440, 0.420347, -0.041554],
        [0.871129, 0.455991, -0.045087],
        [0.862161, 0.498316, -0.043304],
    ], dtype=np.float32)

    # 1. User D image must classify as D (never T or A)
    res_d, conf_d = classifier.classify(user_d, handedness="Left")
    assert res_d == "D", f"User D gesture should classify as D, got {res_d} ({conf_d:.2f})"
    assert conf_d >= 0.90

    # 2. Right-hand / mirrored parity
    user_d_right = np.copy(user_d)
    user_d_right[:, 0] = 1.0 - user_d_right[:, 0]
    res_d_r, conf_d_r = classifier.classify(user_d_right, handedness="Right")
    assert res_d_r == "D", f"Right hand D should classify as D, got {res_d_r} ({conf_d_r:.2f})"

    # 3. Closed fist (T) must NOT classify as D
    fist_t = np.zeros((21, 3), dtype=np.float32)
    fist_t[0] = [0, 0, 0]
    fist_t[5] = [0.15, -0.5, 0.0]
    fist_t[6] = [0.15, -0.35, 0.1]
    fist_t[7] = [0.12, -0.25, 0.1]
    fist_t[8] = [0.10, -0.20, 0.05]
    fist_t[9] = [0.0, -0.52, 0.0]
    fist_t[10] = [0.0, -0.35, 0.1]
    fist_t[11] = [0.0, -0.25, 0.1]
    fist_t[12] = [0.0, -0.20, 0.05]
    fist_t[13] = [-0.15, -0.5, 0.0]
    fist_t[14] = [-0.15, -0.35, 0.1]
    fist_t[15] = [-0.15, -0.25, 0.1]
    fist_t[16] = [-0.12, -0.20, 0.05]
    fist_t[17] = [-0.28, -0.45, 0.0]
    fist_t[18] = [-0.28, -0.35, 0.1]
    fist_t[19] = [-0.28, -0.25, 0.1]
    fist_t[20] = [-0.25, -0.20, 0.05]
    fist_t[1] = [0.10, -0.15, 0.0]
    fist_t[2] = [0.12, -0.30, 0.05]
    fist_t[3] = [0.10, -0.42, 0.1]
    fist_t[4] = [0.06, -0.48, 0.15]
    res_t, _ = classifier.classify(fist_t)
    assert res_t != "D", f"Closed fist should never classify as D, got {res_t}"

    # 4. User fist with hallucinated index finger (media_1794220743013_ce6bed1c.jpg)
    # Must NEVER classify as D, even when index tip was placed in empty space
    w_u, h_u = 1024.0, 576.0
    user_hallucinated_fist = np.zeros((21, 3), dtype=np.float32)
    user_hallucinated_fist[0] = [847/w_u, 428/h_u, 0]
    user_hallucinated_fist[1] = [796/w_u, 400/h_u, 0]
    user_hallucinated_fist[2] = [724/w_u, 369/h_u, 0]
    user_hallucinated_fist[3] = [714/w_u, 314/h_u, 0]
    user_hallucinated_fist[4] = [751/w_u, 308/h_u, 0]
    user_hallucinated_fist[5] = [761/w_u, 279/h_u, 0]
    user_hallucinated_fist[6] = [758/w_u, 228/h_u, 0]
    user_hallucinated_fist[7] = [755/w_u, 203/h_u, 0]
    user_hallucinated_fist[8] = [756/w_u, 176/h_u, 0]  # Hallucinated tip in air
    user_hallucinated_fist[9] = [804/w_u, 275/h_u, 0]
    user_hallucinated_fist[10] = [805/w_u, 223/h_u, 0]
    user_hallucinated_fist[11] = [804/w_u, 255/h_u, 0]
    user_hallucinated_fist[12] = [802/w_u, 287/h_u, 0]
    user_hallucinated_fist[13] = [844/w_u, 276/h_u, 0]
    user_hallucinated_fist[14] = [851/w_u, 234/h_u, 0]
    user_hallucinated_fist[15] = [845/w_u, 268/h_u, 0]
    user_hallucinated_fist[16] = [839/w_u, 302/h_u, 0]
    user_hallucinated_fist[17] = [877/w_u, 279/h_u, 0]
    user_hallucinated_fist[18] = [889/w_u, 245/h_u, 0]
    user_hallucinated_fist[19] = [881/w_u, 273/h_u, 0]
    user_hallucinated_fist[20] = [873/w_u, 300/h_u, 0]

    res_hallu, conf_hallu = classifier.classify(user_hallucinated_fist, handedness="Left")
    assert res_hallu != "D", f"User hallucinated fist must NEVER classify as D, got {res_hallu} ({conf_hallu})"

    # 5. Tracker hallucination sanitizer clamp verification
    from src.tracker import HandTracker
    px_coords = np.zeros((21, 2), dtype=np.int32)
    for i in range(21):
        px_coords[i, 0] = int(user_hallucinated_fist[i, 0] * w_u)
        px_coords[i, 1] = int(user_hallucinated_fist[i, 1] * h_u)
    c_norm, c_px, _, _, _, _ = HandTracker.sanitize_hallucinations(
        user_hallucinated_fist.copy(), px_coords.copy(), int(w_u), int(h_u)
    )
    # The hallucinated tip at y=176 must be clamped down into the fist (y >= 250)
    assert c_px[8, 1] >= 250, f"Sanitized index tip should be folded into fist, got y={c_px[8, 1]}"




def test_strict_l_gesture():
    """Verifies that index up with horizontal thumb detects L, rejecting wrong gestures."""
    classifier = GestureClassifier(mode="heuristic")

    w, h = 836.0, 695.0
    user_l = np.zeros((21, 3), dtype=np.float32)
    user_l[0] = [545.0/w, 565.3/h, 0]
    user_l[1] = [450.3/w, 553.9/h, 0]
    user_l[2] = [342.6/w, 524.0/h, 0]
    user_l[3] = [238.8/w, 527.8/h, 0]
    user_l[4] = [160.9/w, 542.9/h, 0]
    user_l[5] = [382.9/w, 335.4/h, 0]
    user_l[6] = [362.0/w, 222.4/h, 0]
    user_l[7] = [348.6/w, 147.0/h, 0]
    user_l[8] = [342.5/w, 78.0/h, 0]
    user_l[9] = [435.4/w, 323.9/h, 0]
    user_l[10] = [454.8/w, 350.0/h, 0]
    user_l[11] = [445.0/w, 380.0/h, 0]
    user_l[12] = [436.7/w, 415.0/h, 0]
    user_l[13] = [513.5/w, 344.8/h, 0]
    user_l[14] = [520.0/w, 370.0/h, 0]
    user_l[15] = [505.0/w, 400.0/h, 0]
    user_l[16] = [493.8/w, 426.1/h, 0]
    user_l[17] = [585.3/w, 349.0/h, 0]
    user_l[18] = [590.0/w, 375.0/h, 0]
    user_l[19] = [575.0/w, 405.0/h, 0]
    user_l[20] = [556.7/w, 428.9/h, 0]

    # 1. User L image must classify as L
    res_l, conf_l = classifier.classify(user_l, handedness="Left")
    assert res_l == "L", f"User L gesture should classify as L, got {res_l}"
    assert conf_l >= 0.95

    # 2. Right-hand parity
    user_l_right = np.copy(user_l)
    user_l_right[:, 0] = 1.0 - user_l_right[:, 0]
    res_l_r, conf_l_r = classifier.classify(user_l_right, handedness="Right")
    assert res_l_r == "L", f"Right hand L should classify as L, got {res_l_r}"

    # 3. Wrong pose: index up, but thumb is pointing straight UP alongside index (NOT L)
    wrong_thumb_up = np.copy(user_l)
    wrong_thumb_up[4] = [342.6/w, 100.0/h, 0]
    wrong_thumb_up[3] = [342.6/w, 200.0/h, 0]
    res_thumb_up, _ = classifier.classify(wrong_thumb_up, handedness="Left")
    assert res_thumb_up != "L", f"Thumb pointing straight up should not be L, got {res_thumb_up}"

    # 4. Wrong pose: index up, but thumb curled into palm (like 1 / Z)
    wrong_thumb_tucked = np.copy(user_l)
    wrong_thumb_tucked[4] = [382.9/w, 400.0/h, 0]
    wrong_thumb_tucked[3] = [382.9/w, 450.0/h, 0]
    wrong_thumb_tucked[2] = [382.9/w, 480.0/h, 0]
    res_thumb_tucked, _ = classifier.classify(wrong_thumb_tucked, handedness="Left")
    assert res_thumb_tucked != "L", f"Tucked thumb should not be L, got {res_thumb_tucked}"




def test_strict_m_gesture():
    """Verifies that thumb tucked under 3 fingers detects M, rejecting wrong gestures."""
    classifier = GestureClassifier(mode="heuristic")

    w, h = 616.0, 696.0
    user_m = np.zeros((21, 3), dtype=np.float32)
    user_m[0] = [369.5/w, 76.6/h, 0]
    user_m[1] = [247.8/w, 140.3/h, 0]
    user_m[2] = [192.5/w, 250.3/h, 0]
    user_m[3] = [208.8/w, 353.2/h, 0]
    user_m[4] = [299.1/w, 338.1/h, 0]
    user_m[5] = [148.7/w, 269.1/h, 0]
    user_m[6] = [109.9/w, 439.1/h, 0]
    user_m[7] = [94.9/w, 525.3/h, 0]
    user_m[8] = [84.5/w, 598.4/h, 0]
    user_m[9] = [239.0/w, 275.2/h, 0]
    user_m[10] = [219.4/w, 461.8/h, 0]
    user_m[11] = [210.4/w, 563.0/h, 0]
    user_m[12] = [201.6/w, 641.1/h, 0]
    user_m[13] = [330.2/w, 281.7/h, 0]
    user_m[14] = [313.8/w, 357.8/h, 0]
    user_m[15] = [313.6/w, 446.7/h, 0]
    user_m[16] = [293.0/w, 615.0/h, 0]
    user_m[17] = [408.3/w, 285.7/h, 0]
    user_m[18] = [348.2/w, 371.5/h, 0]
    user_m[19] = [297.4/w, 543.1/h, 0]
    user_m[20] = [252.5/w, 429.0/h, 0]

    # 1. User M image must classify as M
    res_m, conf_m = classifier.classify(user_m, handedness="Left")
    assert res_m == "M", f"User M gesture should classify as M, got {res_m}"
    assert conf_m >= 0.95

    # 2. Right-hand parity
    user_m_right = np.copy(user_m)
    user_m_right[:, 0] = 1.0 - user_m_right[:, 0]
    res_m_r, conf_m_r = classifier.classify(user_m_right, handedness="Right")
    assert res_m_r == "M", f"Right hand M should classify as M, got {res_m_r}"

    # 3. Open palm must NOT classify as M
    open_palm = np.zeros((21, 3), dtype=np.float32)
    open_palm[0] = [0, 0, 0]
    open_palm[5] = [0.15, -0.5, 0.0]
    open_palm[8] = [0.25, -0.95, 0.0]
    open_palm[12] = [0.0, -0.98, 0.0]
    open_palm[16] = [-0.25, -0.95, 0.0]
    open_palm[20] = [-0.40, -0.90, 0.0]
    open_palm[4] = [0.65, -0.45, 0.0]
    res_op, _ = classifier.classify(open_palm)
    assert res_op != "M", f"Open palm should not be M, got {res_op}"

    # 4. Real User M screenshot from media_1794033604878_1aaa728f.jpg
    w_u, h_u = 497.0, 543.0
    user_m_screenshot = np.zeros((21, 3), dtype=np.float32)
    user_m_screenshot[0]  = [326.6/w_u, 401.9/h_u, 0]
    user_m_screenshot[1]  = [218.3/w_u, 374.1/h_u, 0]
    user_m_screenshot[2]  = [245.7/w_u, 298.3/h_u, 0]
    user_m_screenshot[3]  = [227.4/w_u, 210.6/h_u, 0]
    user_m_screenshot[4]  = [226.1/w_u, 340.3/h_u, 0]
    user_m_screenshot[5]  = [58.2/w_u,  131.6/h_u, 0]
    user_m_screenshot[6]  = [50.7/w_u,  216.2/h_u, 0]
    user_m_screenshot[7]  = [116.4/w_u, 96.6/h_u,  0]
    user_m_screenshot[8]  = [64.2/w_u,  290.8/h_u, 0]
    user_m_screenshot[9]  = [137.3/w_u, 130.2/h_u, 0]
    user_m_screenshot[10] = [107.2/w_u, 207.6/h_u, 0]
    user_m_screenshot[11] = [178.2/w_u, 212.2/h_u, 0]
    user_m_screenshot[12] = [122.5/w_u, 296.6/h_u, 0]
    user_m_screenshot[13] = [208.0/w_u, 100.0/h_u, 0]
    user_m_screenshot[14] = [208.0/w_u, 109.0/h_u, 0]
    user_m_screenshot[15] = [167.7/w_u, 278.8/h_u, 0]
    user_m_screenshot[16] = [166.2/w_u, 295.4/h_u, 0]
    user_m_screenshot[17] = [356.5/w_u, 159.4/h_u, 0]
    user_m_screenshot[18] = [283.1/w_u, 214.6/h_u, 0]
    user_m_screenshot[19] = [284.5/w_u, 124.5/h_u, 0]
    user_m_screenshot[20] = [278.6/w_u, 145.0/h_u, 0]

    res_user_m, conf_user_m = classifier.classify(user_m_screenshot, handedness="Left")
    assert res_user_m == "M", f"User M screenshot should classify as M, got {res_user_m}"
    assert conf_user_m >= 0.95

    # Parity for User M screenshot
    user_m_screenshot_r = user_m_screenshot.copy()
    user_m_screenshot_r[:, 0] = 1.0 - user_m_screenshot_r[:, 0]
    res_user_m_r, conf_user_m_r = classifier.classify(user_m_screenshot_r, handedness="Right")
    assert res_user_m_r == "M", f"User M screenshot (Right) should classify as M, got {res_user_m_r}"
    assert conf_user_m_r >= 0.95

    # 5. Real User M screenshot from media_1794223340432_b2287891.jpg
    w2, h2 = 538.0, 512.0
    user_m2 = np.zeros((21, 3), dtype=np.float32)
    coords_m2 = {
        0: (185.9, 208.4), 1: (253.6, 237.4), 2: (309.1, 263.6), 3: (322.0, 274.0), 4: (335.9, 284.2),
        5: (128.7, 133.6), 6: (132.0, 227.4), 7: (149.9, 292.7), 8: (166.5, 342.8),
        9: (193.5, 111.7), 10: (184.4, 246.0), 11: (183.7, 333.3), 12: (187.6, 399.3),
        13: (265.7, 108.8), 14: (247.4, 255.1), 15: (226.5, 349.7), 16: (215.9, 415.5),
        17: (337.4, 133.7), 18: (323.9, 248.8), 19: (302.7, 343.8), 20: (292.6, 411.1),
    }
    for idx, (px, py) in coords_m2.items():
        user_m2[idx] = [px / w2, py / h2, 0.0]

    res_user_m2, conf_user_m2 = classifier.classify(user_m2, handedness="Right")
    assert res_user_m2 == "M", f"User M image 2 (Right) should classify as M, got {res_user_m2}"
    assert conf_user_m2 >= 0.95

    user_m2_l = user_m2.copy()
    user_m2_l[:, 0] = 1.0 - user_m2_l[:, 0]
    res_user_m2_l, conf_user_m2_l = classifier.classify(user_m2_l, handedness="Left")
    assert res_user_m2_l == "M", f"User M image 2 (Left) should classify as M, got {res_user_m2_l}"
    assert conf_user_m2_l >= 0.95




def test_strict_n_gesture():
    """Verifies that 2 fingers draped forward detects N, rejecting P (spread) and open palms."""
    classifier = GestureClassifier(mode="heuristic")

    w, h = 650.0, 652.0
    user_n = np.zeros((21, 3), dtype=np.float32)
    user_n[0] = [406.2/w, 74.0/h, 0]
    user_n[1] = [300.9/w, 104.5/h, 0]
    user_n[2] = [210.8/w, 168.9/h, 0]
    user_n[3] = [152.4/w, 241.0/h, 0]
    user_n[4] = [138.9/w, 307.0/h, 0]
    user_n[5] = [216.9/w, 237.7/h, 0]
    user_n[6] = [211.0/w, 375.6/h, 0]
    user_n[7] = [212.4/w, 444.8/h, 0]
    user_n[8] = [215.4/w, 501.1/h, 0]
    user_n[9] = [282.9/w, 253.4/h, 0]
    user_n[10] = [272.1/w, 395.0/h, 0]
    user_n[11] = [264.8/w, 477.8/h, 0]
    user_n[12] = [264.9/w, 538.1/h, 0]
    user_n[13] = [358.3/w, 260.0/h, 0]
    user_n[14] = [364.0/w, 340.2/h, 0]
    user_n[15] = [358.3/w, 300.1/h, 0]
    user_n[16] = [355.0/w, 256.3/h, 0]
    user_n[17] = [431.4/w, 247.7/h, 0]
    user_n[18] = [420.9/w, 296.2/h, 0]
    user_n[19] = [413.3/w, 260.3/h, 0]
    user_n[20] = [408.9/w, 225.6/h, 0]

    # 1. User N image must classify as N
    res_n, conf_n = classifier.classify(user_n, handedness="Left")
    assert res_n == "N", f"User N gesture should classify as N, got {res_n}"
    assert conf_n >= 0.95

    # 2. Right-hand parity
    user_n_right = np.copy(user_n)
    user_n_right[:, 0] = 1.0 - user_n_right[:, 0]
    res_n_r, conf_n_r = classifier.classify(user_n_right, handedness="Right")
    assert res_n_r == "N", f"Right hand N should classify as N, got {res_n_r}"

    # 3. Spread downward fingers (P sign) must NOT classify as N
    user_p = np.copy(user_n)
    user_p[8] = [150.0/w, 501.1/h, 0]
    user_p[12] = [330.0/w, 538.1/h, 0]
    user_p[4] = [240.0/w, 350.0/h, 0]
    res_p, _ = classifier.classify(user_p, handedness="Left")
    assert res_p == "P", f"Spread downward fingers should classify as P, got {res_p}"

    # 4. User N2 image (2-finger drape) must classify as N
    w2, h2 = 747.0, 612.0
    user_n2 = np.zeros((21, 3), dtype=np.float32)
    user_n2[0] = [399.0 / w2, 114.9 / h2, 0]
    user_n2[1] = [303.1 / w2, 147.3 / h2, 0]
    user_n2[2] = [248.9 / w2, 237.7 / h2, 0]
    user_n2[3] = [263.9 / w2, 326.2 / h2, 0]
    user_n2[4] = [324.1 / w2, 381.2 / h2, 0]
    user_n2[5] = [275.9 / w2, 227.2 / h2, 0]
    user_n2[6] = [282.2 / w2, 370.0 / h2, 0]
    user_n2[7] = [292.4 / w2, 434.9 / h2, 0]
    user_n2[8] = [301.5 / w2, 487.3 / h2, 0]
    user_n2[9] = [361.4 / w2, 223.7 / h2, 0]
    user_n2[10] = [356.8 / w2, 378.5 / h2, 0]
    user_n2[11] = [350.7 / w2, 459.8 / h2, 0]
    user_n2[12] = [351.1 / w2, 519.8 / h2, 0]
    user_n2[13] = [433.4 / w2, 225.6 / h2, 0]
    user_n2[14] = [398.9 / w2, 357.4 / h2, 0]
    user_n2[15] = [372.2 / w2, 332.6 / h2, 0]
    user_n2[16] = [367.6 / w2, 295.8 / h2, 0]
    user_n2[17] = [487.3 / w2, 232.4 / h2, 0]
    user_n2[18] = [436.3 / w2, 329.8 / h2, 0]
    user_n2[19] = [412.1 / w2, 311.1 / h2, 0]
    user_n2[20] = [411.0 / w2, 279.3 / h2, 0]

    res_n2, conf_n2 = classifier.classify(user_n2, handedness="Left")
    assert res_n2 == "N", f"User N2 gesture should classify as N, got {res_n2}"
    assert conf_n2 >= 0.95

    # Right-hand parity for user N2
    user_n2_r = np.copy(user_n2)
    user_n2_r[:, 0] = 1.0 - user_n2_r[:, 0]
    res_n2_r, conf_n2_r = classifier.classify(user_n2_r, handedness="Right")
    assert res_n2_r == "N", f"Right hand N2 should classify as N, got {res_n2_r}"

    # 5. Draped N with thumb emerging between middle & ring fingers
    w_m2, h_m2 = 538.0, 512.0
    user_n3 = np.zeros((21, 3), dtype=np.float32)
    coords_base = {
        0: (185.9, 208.4), 1: (253.6, 237.4), 2: (309.1, 263.6), 3: (322.0, 274.0), 4: (220.0, 260.0),
        5: (128.7, 133.6), 6: (132.0, 227.4), 7: (149.9, 292.7), 8: (166.5, 342.8),
        9: (193.5, 111.7), 10: (184.4, 246.0), 11: (183.7, 333.3), 12: (187.6, 399.3),
        13: (265.7, 108.8), 14: (247.4, 255.1), 15: (226.5, 349.7), 16: (240.0, 250.0),
        17: (337.4, 133.7), 18: (323.9, 248.8), 19: (302.7, 343.8), 20: (310.0, 240.0),
    }
    for idx, (px, py) in coords_base.items():
        user_n3[idx] = [px / w_m2, py / h_m2, 0.0]

    res_n3, conf_n3 = classifier.classify(user_n3, handedness="Right")
    assert res_n3 == "N", f"Draped N with thumb between middle and ring should classify as N, got {res_n3}"
    assert conf_n3 >= 0.95

    user_n3_l = user_n3.copy()
    user_n3_l[:, 0] = 1.0 - user_n3_l[:, 0]
    res_n3_l, conf_n3_l = classifier.classify(user_n3_l, handedness="Left")
    assert res_n3_l == "N", f"Draped N (Left) should classify as N, got {res_n3_l}"
    assert conf_n3_l >= 0.95




def test_strict_o_gesture():
    """Verifies that all 4 fingertips converging to thumb detects O, rejecting open palms and F."""
    classifier = GestureClassifier(mode="heuristic")

    w, h = 472.0, 520.0
    user_o = np.zeros((21, 3), dtype=np.float32)
    user_o[0] = [316.4/w, 479.2/h, 0]
    user_o[1] = [210.0/w, 428.4/h, 0]
    user_o[2] = [128.8/w, 348.9/h, 0]
    user_o[3] = [68.8/w, 295.4/h, 0]
    user_o[4] = [56.8/w, 234.2/h, 0]
    user_o[5] = [241.4/w, 221.7/h, 0]
    user_o[6] = [199.7/w, 124.1/h, 0]
    user_o[7] = [144.1/w, 140.6/h, 0]
    user_o[8] = [127.7/w, 190.5/h, 0]
    user_o[9] = [274.7/w, 217.9/h, 0]
    user_o[10] = [238.4/w, 111.6/h, 0]
    user_o[11] = [178.5/w, 154.1/h, 0]
    user_o[12] = [94.3/w, 190.2/h, 0]
    user_o[13] = [305.8/w, 221.9/h, 0]
    user_o[14] = [258.1/w, 149.3/h, 0]
    user_o[15] = [124.7/w, 132.0/h, 0]
    user_o[16] = [78.4/w, 177.7/h, 0]
    user_o[17] = [328.4/w, 238.6/h, 0]
    user_o[18] = [210.0/w, 102.2/h, 0]
    user_o[19] = [124.7/w, 132.0/h, 0]
    user_o[20] = [84.3/w, 165.3/h, 0]

    # 1. User O image must classify as O
    res_o, conf_o = classifier.classify(user_o, handedness="Left")
    assert res_o == "O", f"User O gesture should classify as O, got {res_o}"
    assert conf_o >= 0.95

    # 2. Right-hand parity
    user_o_right = np.copy(user_o)
    user_o_right[:, 0] = 1.0 - user_o_right[:, 0]
    res_o_r, conf_o_r = classifier.classify(user_o_right, handedness="Right")
    assert res_o_r == "O", f"Right hand O should classify as O, got {res_o_r}"

    # 3. Flat open palm must NOT classify as O
    open_palm = np.zeros((21, 3), dtype=np.float32)
    open_palm[0] = [0, 0, 0]
    open_palm[5] = [0.15, -0.5, 0.0]
    open_palm[8] = [0.25, -0.95, 0.0]
    open_palm[12] = [0.0, -0.98, 0.0]
    open_palm[16] = [-0.25, -0.95, 0.0]
    open_palm[20] = [-0.40, -0.90, 0.0]
    open_palm[4] = [0.65, -0.45, 0.0]
    res_op, _ = classifier.classify(open_palm)
    assert res_op != "O", f"Open palm should not be O, got {res_op}"




def test_strict_p_gesture():
    """Verifies that downward K-shape detects P, distinguishing P from C, N, and open palm."""
    classifier = GestureClassifier(mode="heuristic")

    w, h = 782.0, 491.0
    user_p = np.zeros((21, 3), dtype=np.float32)
    user_p[0] = [597.9 / w, 167.7 / h, 0]
    user_p[1] = [473.7 / w, 165.0 / h, 0]
    user_p[2] = [353.5 / w, 193.8 / h, 0]
    user_p[3] = [269.4 / w, 251.3 / h, 0]
    user_p[4] = [212.5 / w, 301.4 / h, 0]
    user_p[5] = [281.0 / w, 102.6 / h, 0]
    user_p[6] = [183.9 / w, 200.6 / h, 0]
    user_p[7] = [120.9 / w, 255.5 / h, 0]
    user_p[8] = [66.8 / w, 299.9 / h, 0]
    user_p[9] = [340.0 / w, 100.0 / h, 0]
    user_p[10] = [269.4 / w, 251.3 / h, 0]
    user_p[11] = [254.6 / w, 340.0 / h, 0]
    user_p[12] = [259.1 / w, 401.2 / h, 0]
    user_p[13] = [408.7 / w, 126.4 / h, 0]
    user_p[14] = [367.1 / w, 335.1 / h, 0]
    user_p[15] = [311.2 / w, 353.7 / h, 0]
    user_p[16] = [300.8 / w, 408.5 / h, 0]
    user_p[17] = [473.7 / w, 165.0 / h, 0]
    user_p[18] = [401.2 / w, 278.8 / h, 0]
    user_p[19] = [335.5 / w, 270.8 / h, 0]
    user_p[20] = [347.7 / w, 373.9 / h, 0]

    # 1. User P image must classify as P (heuristic and hybrid)
    res_p, conf_p = classifier.classify(user_p, handedness="Left")
    assert res_p == "P", f"User P gesture should classify as P, got {res_p}"
    assert conf_p >= 0.95

    clf_hybrid = GestureClassifier(mode="hybrid")
    res_ph, conf_ph = clf_hybrid.classify(user_p, handedness="Left")
    assert res_ph == "P", f"Hybrid classifier should classify as P, got {res_ph}"
    assert conf_ph >= 0.95

    # 2. Right-hand parity
    user_p_right = np.copy(user_p)
    user_p_right[:, 0] = 1.0 - user_p_right[:, 0]
    res_pr, conf_pr = classifier.classify(user_p_right, handedness="Right")
    assert res_pr == "P", f"Right hand P should classify as P, got {res_pr}"
    assert conf_pr >= 0.95

    # 3. User P from uploaded image (media_1794480375743_e12887cc.jpg)
    w_up, h_up = 872.0, 571.0
    user_p_up = np.zeros((21, 3), dtype=np.float32)
    user_p_up[0]  = [808.2 / w_up, 212.0 / h_up, 0.0]
    user_p_up[1]  = [653.3 / w_up, 274.0 / h_up, -0.02]
    user_p_up[2]  = [476.4 / w_up, 301.5 / h_up, -0.04]
    user_p_up[3]  = [452.6 / w_up, 365.0 / h_up, -0.05]
    user_p_up[4]  = [400.7 / w_up, 403.0 / h_up, -0.06]
    user_p_up[5]  = [446.7 / w_up, 129.1 / h_up, 0.0]
    user_p_up[6]  = [264.9 / w_up, 254.8 / h_up, 0.02]
    user_p_up[7]  = [178.3 / w_up, 293.3 / h_up, 0.04]
    user_p_up[8]  = [94.1 / w_up, 319.4 / h_up, 0.05]
    user_p_up[9]  = [518.5 / w_up, 125.1 / h_up, 0.0]
    user_p_up[10] = [398.5 / w_up, 160.2 / h_up, -0.05]
    user_p_up[11] = [388.1 / w_up, 412.8 / h_up, -0.03]
    user_p_up[12] = [396.8 / w_up, 488.8 / h_up, 0.0]
    user_p_up[13] = [598.1 / w_up, 138.9 / h_up, 0.0]
    user_p_up[14] = [543.8 / w_up, 270.1 / h_up, -0.05]
    user_p_up[15] = [587.2 / w_up, 305.5 / h_up, -0.03]
    user_p_up[16] = [619.0 / w_up, 318.1 / h_up, 0.0]
    user_p_up[17] = [653.3 / w_up, 274.0 / h_up, 0.0]
    user_p_up[18] = [556.0 / w_up, 327.7 / h_up, -0.05]
    user_p_up[19] = [521.3 / w_up, 315.0 / h_up, -0.03]
    user_p_up[20] = [629.3 / w_up, 297.6 / h_up, 0.0]

    res_pu, conf_pu = classifier.classify(user_p_up, handedness="Left")
    assert res_pu == "P", f"User uploaded P image should classify as P, got {res_pu}"
    assert conf_pu >= 0.95

    res_puh, conf_puh = clf_hybrid.classify(user_p_up, handedness="Left")
    assert res_puh == "P", f"User uploaded P hybrid should classify as P, got {res_puh}"
    assert conf_puh >= 0.95

    # Parity for uploaded P
    user_pu_r = np.copy(user_p_up)
    user_pu_r[:, 0] = 1.0 - user_pu_r[:, 0]
    res_pur, conf_pur = classifier.classify(user_pu_r, handedness="Right")
    assert res_pur == "P", f"Right hand uploaded P should classify as P, got {res_pur}"
    assert conf_pur >= 0.95

    # 4. Inverted peace sign (thumb on palm away from middle finger) must NOT classify as P
    inv_peace = np.copy(user_p_up)
    inv_peace[4] = [0.65, 0.25, 0.0]
    res_ip, _ = classifier.classify(inv_peace, handedness="Left")
    assert res_ip != "P", f"Inverted peace sign without thumb on middle joint must NOT be P, got {res_ip}"

    # 5. Flat open palm must NOT classify as P
    open_palm = np.zeros((21, 3), dtype=np.float32)
    open_palm[0] = [0, 0, 0]
    open_palm[5] = [0.15, -0.5, 0.0]
    open_palm[8] = [0.25, -0.95, 0.0]
    open_palm[12] = [0.0, -0.98, 0.0]
    open_palm[16] = [-0.25, -0.95, 0.0]
    open_palm[20] = [-0.40, -0.90, 0.0]
    open_palm[4] = [0.65, -0.45, 0.0]
    res_op, _ = classifier.classify(open_palm)
    assert res_op != "P", f"Open palm should not be P, got {res_op}"




def test_strict_q_gesture():
    """Verifies that downward index and thumb pinch detects Q, distinguishing Q from O, P, N, and open palm."""
    classifier = GestureClassifier(mode="heuristic")

    w, h = 672.0, 506.0
    user_q = np.zeros((21, 3), dtype=np.float32)
    user_q[0] = [603.0 / w, 338.1 / h, 0]
    user_q[1] = [456.0 / w, 368.8 / h, 0]
    user_q[2] = [337.5 / w, 351.9 / h, 0]
    user_q[3] = [253.8 / w, 364.1 / h, 0]
    user_q[4] = [187.9 / w, 382.5 / h, 0]
    user_q[5] = [377.9 / w, 215.5 / h, 0]
    user_q[6] = [274.8 / w, 308.1 / h, 0]
    user_q[7] = [228.0 / w, 365.4 / h, 0]
    user_q[8] = [190.5 / w, 412.8 / h, 0]
    user_q[9] = [363.2 / w, 163.1 / h, 0]
    user_q[10] = [231.2 / w, 280.3 / h, 0]
    user_q[11] = [184.6 / w, 362.7 / h, 0]
    user_q[12] = [146.4 / w, 426.4 / h, 0]
    user_q[13] = [337.8 / w, 142.2 / h, 0]
    user_q[14] = [214.3 / w, 264.8 / h, 0]
    user_q[15] = [168.9 / w, 349.7 / h, 0]
    user_q[16] = [141.0 / w, 406.1 / h, 0]
    user_q[17] = [337.8 / w, 142.2 / h, 0]
    user_q[18] = [214.3 / w, 264.8 / h, 0]
    user_q[19] = [168.9 / w, 349.7 / h, 0]
    user_q[20] = [128.7 / w, 424.3 / h, 0]

    # 1. User Q image must classify as Q (heuristic and hybrid)
    res_q, conf_q = classifier.classify(user_q, handedness="Left")
    assert res_q == "Q", f"User Q gesture should classify as Q, got {res_q}"
    assert conf_q >= 0.95

    clf_hybrid = GestureClassifier(mode="hybrid")
    res_qh, conf_qh = clf_hybrid.classify(user_q, handedness="Left")
    assert res_qh == "Q", f"Hybrid classifier should classify as Q, got {res_qh}"
    assert conf_qh >= 0.95

    # 2. Right-hand parity
    user_q_right = np.copy(user_q)
    user_q_right[:, 0] = 1.0 - user_q_right[:, 0]
    res_qr, conf_qr = classifier.classify(user_q_right, handedness="Right")
    assert res_qr == "Q", f"Right hand Q should classify as Q, got {res_qr}"
    assert conf_qr >= 0.95

    # 3. User Q from uploaded image (media_1794477743142_5905ce03.jpg)
    w_up, h_up = 628.0, 442.0
    user_q_uploaded = np.zeros((21, 3), dtype=np.float32)
    user_q_uploaded[0]  = [490.3 / w_up, 289.7 / h_up, 0.0]
    user_q_uploaded[1]  = [362.9 / w_up, 295.8 / h_up, -0.02]
    user_q_uploaded[2]  = [268.4 / w_up, 264.6 / h_up, -0.04]
    user_q_uploaded[3]  = [198.1 / w_up, 246.3 / h_up, -0.05]
    user_q_uploaded[4]  = [101.5 / w_up, 263.3 / h_up, -0.06]
    user_q_uploaded[5]  = [344.9 / w_up, 164.5 / h_up, 0.0]
    user_q_uploaded[6]  = [244.5 / w_up, 209.8 / h_up, 0.02]
    user_q_uploaded[7]  = [146.9 / w_up, 224.8 / h_up, 0.04]
    user_q_uploaded[8]  = [85.7 / w_up, 254.9 / h_up, 0.05]
    user_q_uploaded[9]  = [336.2 / w_up, 117.7 / h_up, 0.0]
    user_q_uploaded[10] = [208.3 / w_up, 174.3 / h_up, -0.05]
    user_q_uploaded[11] = [187.6 / w_up, 156.0 / h_up, -0.03]
    user_q_uploaded[12] = [150.2 / w_up, 241.8 / h_up, 0.0]
    user_q_uploaded[13] = [318.1 / w_up, 95.8 / h_up, 0.0]
    user_q_uploaded[14] = [208.4 / w_up, 174.3 / h_up, -0.05]
    user_q_uploaded[15] = [187.6 / w_up, 156.0 / h_up, -0.03]
    user_q_uploaded[16] = [152.9 / w_up, 264.9 / h_up, 0.0]
    user_q_uploaded[17] = [298.3 / w_up, 96.9 / h_up, 0.0]
    user_q_uploaded[18] = [208.4 / w_up, 174.3 / h_up, -0.05]
    user_q_uploaded[19] = [132.0 / w_up, 209.7 / h_up, -0.03]
    user_q_uploaded[20] = [100.1 / w_up, 247.2 / h_up, 0.0]

    res_qu, conf_qu = classifier.classify(user_q_uploaded, handedness="Left")
    assert res_qu == "Q", f"User uploaded Q image should classify as Q, got {res_qu}"
    assert conf_qu >= 0.95

    res_quh, conf_quh = clf_hybrid.classify(user_q_uploaded, handedness="Left")
    assert res_quh == "Q", f"User uploaded Q hybrid should classify as Q, got {res_quh}"
    assert conf_quh >= 0.95

    # Parity for uploaded Q
    user_qu_r = np.copy(user_q_uploaded)
    user_qu_r[:, 0] = 1.0 - user_qu_r[:, 0]
    res_qur, conf_qur = classifier.classify(user_qu_r, handedness="Right")
    assert res_qur == "Q", f"Right hand uploaded Q should classify as Q, got {res_qur}"
    assert conf_qur >= 0.95

    # 4. User Q from uploaded image (media_1794485506664_94a66ccf.png & media_1794485509933_def56660.jpg)
    w_new, h_new = 890.0, 606.0
    user_q_new = np.zeros((21, 3), dtype=np.float32)
    user_q_new[0]  = [666.9 / w_new, 380.3 / h_new, 0.0]
    user_q_new[1]  = [519.6 / w_new, 386.3 / h_new, 0.0]
    user_q_new[2]  = [391.9 / w_new, 346.8 / h_new, 0.0]
    user_q_new[3]  = [303.2 / w_new, 346.2 / h_new, 0.0]
    user_q_new[4]  = [237.8 / w_new, 340.6 / h_new, 0.0]
    user_q_new[5]  = [422.5 / w_new, 155.3 / h_new, 0.0]
    user_q_new[6]  = [296.8 / w_new, 224.8 / h_new, 0.0]
    user_q_new[7]  = [221.2 / w_new, 300.0 / h_new, 0.0]
    user_q_new[8]  = [165.8 / w_new, 366.6 / h_new, 0.0]
    user_q_new[9]  = [442.4 / w_new, 141.2 / h_new, 0.0]
    user_q_new[10] = [314.1 / w_new, 231.1 / h_new, 0.0]
    user_q_new[11] = [239.2 / w_new, 303.1 / h_new, 0.0]
    user_q_new[12] = [188.9 / w_new, 366.8 / h_new, 0.0]
    user_q_new[13] = [462.5 / w_new, 153.3 / h_new, 0.0]
    user_q_new[14] = [349.9 / w_new, 254.1 / h_new, 0.0]
    user_q_new[15] = [289.4 / w_new, 300.3 / h_new, 0.0]
    user_q_new[16] = [188.9 / w_new, 366.8 / h_new, 0.0]
    user_q_new[17] = [474.1 / w_new, 192.2 / h_new, 0.0]
    user_q_new[18] = [385.4 / w_new, 346.2 / h_new, 0.0]
    user_q_new[19] = [303.3 / w_new, 341.0 / h_new, 0.0]
    user_q_new[20] = [188.9 / w_new, 366.8 / h_new, 0.0]

    res_qn_l, conf_qn_l = classifier.classify(user_q_new, handedness="Left")
    assert res_qn_l == "Q", f"User Q new (Left) should classify as Q, got {res_qn_l}"
    assert conf_qn_l >= 0.95

    res_qn_h, conf_qn_h = clf_hybrid.classify(user_q_new, handedness="Left")
    assert res_qn_h == "Q", f"User Q new hybrid (Left) should classify as Q, got {res_qn_h}"
    assert conf_qn_h >= 0.95

    user_qn_r = np.copy(user_q_new)
    user_qn_r[:, 0] = 1.0 - user_qn_r[:, 0]
    res_qn_r, conf_qn_r = classifier.classify(user_qn_r, handedness="Right")
    assert res_qn_r == "Q", f"User Q new (Right) should classify as Q, got {res_qn_r}"
    assert conf_qn_r >= 0.95

    # 5. Flat open palm must NOT classify as Q
    open_palm = np.zeros((21, 3), dtype=np.float32)
    open_palm[0] = [0, 0, 0]
    open_palm[5] = [0.15, -0.5, 0.0]
    open_palm[8] = [0.25, -0.95, 0.0]
    open_palm[12] = [0.0, -0.98, 0.0]
    open_palm[16] = [-0.25, -0.95, 0.0]
    open_palm[20] = [-0.40, -0.90, 0.0]
    open_palm[4] = [0.65, -0.45, 0.0]
    res_op, _ = classifier.classify(open_palm)
    assert res_op != "Q", f"Open palm should not be Q, got {res_op}"




def test_strict_r_gesture():
    """Verifies that crossed index and middle fingers detect R, distinguishing R from U, V, K, H, B, and open palm."""
    classifier = GestureClassifier(mode="heuristic")
    clf_hybrid = GestureClassifier(mode="hybrid")

    # 1. User uploaded R gesture (media_1794632875900_b0f54415.png)
    w, h = 371.0, 601.0
    user_r = np.zeros((21, 3), dtype=np.float32)
    user_r[0] = [130.0 / w, 529.7 / h, 0.0]
    user_r[1] = [218.9 / w, 471.6 / h, -0.02]
    user_r[2] = [262.4 / w, 365.5 / h, -0.04]
    user_r[3] = [215.0 / w, 280.0 / h, -0.05]
    user_r[4] = [167.8 / w, 201.4 / h, -0.06]

    user_r[5] = [225.6 / w, 272.9 / h, -0.01]
    user_r[6] = [214.1 / w, 182.0 / h, -0.02]
    user_r[7] = [187.3 / w, 110.5 / h, -0.03]
    user_r[8] = [157.1 / w, 48.4 / h, -0.04]

    user_r[9] = [163.9 / w, 270.9 / h, 0.0]
    user_r[10] = [183.8 / w, 152.9 / h, 0.01]
    user_r[11] = [215.6 / w, 86.7 / h, 0.02]
    user_r[12] = [235.3 / w, 29.2 / h, 0.03]

    user_r[13] = [110.5 / w, 292.1 / h, 0.0]
    user_r[14] = [119.7 / w, 194.0 / h, 0.01]
    user_r[15] = [127.2 / w, 303.0 / h, 0.02]
    user_r[16] = [179.6 / w, 332.3 / h, 0.03]

    user_r[17] = [54.9 / w, 325.6 / h, 0.0]
    user_r[18] = [83.6 / w, 252.0 / h, 0.01]
    user_r[19] = [110.0 / w, 310.0 / h, 0.02]
    user_r[20] = [143.3 / w, 356.8 / h, 0.03]

    res_rl, conf_rl = classifier.classify(user_r, handedness="Left")
    assert res_rl == "R", f"User R (Left) should classify as R, got {res_rl}"
    assert conf_rl >= 0.95

    res_rl_h, conf_rl_h = clf_hybrid.classify(user_r, handedness="Left")
    assert res_rl_h == "R", f"User R (Left hybrid) should classify as R, got {res_rl_h}"
    assert conf_rl_h >= 0.95

    # Right hand parity
    user_r_right = np.copy(user_r)
    user_r_right[:, 0] = 1.0 - user_r_right[:, 0]
    res_rr, conf_rr = classifier.classify(user_r_right, handedness="Right")
    assert res_rr == "R", f"User R (Right) should classify as R, got {res_rr}"
    assert conf_rr >= 0.95

    res_rr_h, conf_rr_h = clf_hybrid.classify(user_r_right, handedness="Right")
    assert res_rr_h == "R", f"User R (Right hybrid) should classify as R, got {res_rr_h}"
    assert conf_rr_h >= 0.95

    # 2. Distinguish from U, V, K, H, B (they must NOT classify as R)
    from data.train_model import create_base_skeleton
    for other in ["U", "V", "K", "H", "B"]:
        skel = create_base_skeleton(other)
        pred, _ = classifier.classify(skel, handedness="Right")
        assert pred != "R", f"{other} should not classify as R, got {pred}"
        pred_h, _ = clf_hybrid.classify(skel, handedness="Right")
        assert pred_h != "R", f"{other} (hybrid) should not classify as R, got {pred_h}"

    # 3. Flat open palm must not classify as R
    open_palm = np.zeros((21, 3), dtype=np.float32)
    open_palm[0] = [0, 0, 0]
    open_palm[5] = [0.15, 0.5, 0.0]
    open_palm[8] = [0.25, 0.95, 0.0]
    open_palm[9] = [0.0, 0.52, 0.0]
    open_palm[12] = [0.0, 0.98, 0.0]
    open_palm[13] = [-0.15, 0.5, 0.0]
    open_palm[16] = [-0.25, 0.95, 0.0]
    open_palm[17] = [-0.30, 0.45, 0.0]
    open_palm[20] = [-0.40, 0.90, 0.0]
    open_palm[4] = [0.65, 0.45, 0.0]
    res_op, _ = classifier.classify(open_palm)
    assert res_op != "R", f"Open palm should not be R, got {res_op}"




def test_all_24_static_alphabets():
    """Verifies that all 24 static ASL letters are correctly classified in hybrid mode."""
    import csv
    classifier = GestureClassifier(mode="hybrid")
    samples = {}
    with open("data/asl_dataset.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            lbl = r[0]
            if lbl not in samples:
                samples[lbl] = np.array([float(x) for x in r[1:64]], dtype=np.float32).reshape(21, 3)

    for letter in sorted(samples.keys()):
        # Right hand test
        pred, conf = classifier.classify(samples[letter], handedness="Right")
        assert pred == letter, f"Right hand {letter} misclassified as {pred} (conf={conf:.2f})"
        assert conf >= 0.85, f"Confidence for {letter} too low: {conf:.2f}"

        # Left hand chirality test
        left_sample = samples[letter].copy()
        left_sample[:, 0] = -left_sample[:, 0]
        pred_l, conf_l = classifier.classify(left_sample, handedness="Left")
        assert pred_l == letter, f"Left hand {letter} misclassified as {pred_l} (conf={conf_l:.2f})"
        assert conf_l >= 0.85




def test_strict_g_gesture():
    """Verifies that ASL 'G' gesture is correctly classified across orientations and chirality, never confused with D or O."""
    from data.train_model import create_base_skeleton
    from scipy.spatial.transform import Rotation as R_scipy

    classifier = GestureClassifier(mode="hybrid")
    base_g = create_base_skeleton("G")

    # 1. Base upright and horizontal orientations
    pred_r, conf_r = classifier.classify(base_g, handedness="Right")
    assert pred_r == "G", f"Base G should classify as G, got {pred_r}"
    assert conf_r >= 0.85, f"Confidence too low: {conf_r}"

    # 2. Left hand parity
    base_g_l = base_g.copy()
    base_g_l[:, 0] = -base_g_l[:, 0]
    pred_l, conf_l = classifier.classify(base_g_l, handedness="Left")
    assert pred_l == "G", f"Left-hand G should classify as G, got {pred_l}"
    assert conf_l >= 0.85

    # 3. 3D rotations (roll, pitch, yaw) - G must never be misclassified as D or O
    for rx in [-20, 0, 20]:
        for ry in [-20, 0, 20]:
            for rz in [-25, 0, 25]:
                rot = R_scipy.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
                g_rot = (base_g @ rot.T).astype(np.float32)
                pred, conf = classifier.classify(g_rot, handedness="Right")
                assert pred == "G", f"Tilted G at ({rx},{ry},{rz}) misclassified as {pred} ({conf:.2f})"
                assert conf >= 0.80

    # 4. Natural horizontal hand ASL G gesture (hand horizontal, index extended along hand, thumb parallel)
    nat_g = np.zeros((21, 3), dtype=np.float32)
    nat_g[0] = [0.60, 0.50, 0.0]
    nat_g[1] = [0.56, 0.47, -0.01]; nat_g[2] = [0.51, 0.44, -0.01]; nat_g[3] = [0.44, 0.43, -0.01]; nat_g[4] = [0.36, 0.43, -0.01]
    nat_g[5] = [0.46, 0.46, 0.0];   nat_g[6] = [0.39, 0.46, -0.01]; nat_g[7] = [0.32, 0.46, -0.01]; nat_g[8] = [0.25, 0.46, -0.01]
    nat_g[9] = [0.45, 0.50, 0.0];   nat_g[10] = [0.41, 0.50, 0.03]; nat_g[11] = [0.43, 0.50, 0.04]; nat_g[12] = [0.45, 0.50, 0.02]
    nat_g[13] = [0.46, 0.54, 0.0];  nat_g[14] = [0.42, 0.54, 0.03]; nat_g[15] = [0.44, 0.54, 0.04]; nat_g[16] = [0.46, 0.54, 0.02]
    nat_g[17] = [0.47, 0.58, 0.0];  nat_g[18] = [0.44, 0.58, 0.03]; nat_g[19] = [0.45, 0.58, 0.04]; nat_g[20] = [0.47, 0.58, 0.02]

    pred_nat, conf_nat = classifier.classify(nat_g, handedness="Right")
    assert pred_nat == "G", f"Natural horizontal G should classify as G, got {pred_nat} ({conf_nat:.2f})"
    assert conf_nat >= 0.92, f"Confidence on natural G too low: {conf_nat:.2f}"

    # Chirality parity on natural G
    nat_g_l = nat_g.copy()
    nat_g_l[:, 0] = 1.0 - nat_g_l[:, 0]
    pred_nat_l, conf_nat_l = classifier.classify(nat_g_l, handedness="Left")
    assert pred_nat_l == "G", f"Natural horizontal left-hand G should classify as G, got {pred_nat_l} ({conf_nat_l:.2f})"
    assert conf_nat_l >= 0.92




def test_strict_i_vs_j_distinction():
    """Verifies that ASL 'I' and 'J' are correctly recognized and never confused, with 100% left/right parity."""
    from data.train_model import create_base_skeleton
    classifier = GestureClassifier()

    # 1. Base upright 'I'
    base_i = create_base_skeleton("I")
    pred_i_r, conf_i_r = classifier.classify(base_i, handedness="Right")
    assert pred_i_r == "I" and conf_i_r >= 0.92, f"Base I should classify as I, got {pred_i_r} ({conf_i_r})"

    # Left-hand parity for base I
    base_i_l = base_i.copy()
    base_i_l[:, 0] = -base_i_l[:, 0]
    pred_i_l, conf_i_l = classifier.classify(base_i_l, handedness="Left")
    assert pred_i_l == "I" and conf_i_l >= 0.92, f"Left-hand base I should classify as I, got {pred_i_l} ({conf_i_l})"

    # 2. User Image 1 (Real ASL I)
    w1, h1 = 577.0, 632.0
    pts1 = np.zeros((21, 3), dtype=np.float32)
    pts1[0] = [294.56/w1, 510.30/h1, 0]
    pts1[1] = [196.87/w1, 472.47/h1, 0]; pts1[2] = [128.05/w1, 372.99/h1, 0]; pts1[3] = [132.87/w1, 275.65/h1, 0]; pts1[4] = [148.78/w1, 285.55/h1, 0]
    pts1[5] = [179.01/w1, 174.53/h1, 0]; pts1[6] = [115.93/w1, 190.65/h1, 0]; pts1[7] = [129.21/w1, 247.54/h1, 0]; pts1[8] = [191.21/w1, 221.16/h1, 0]
    pts1[9] = [246.55/w1, 169.92/h1, 0]; pts1[10] = [177.65/w1, 240.22/h1, 0]; pts1[11] = [154.38/w1, 269.32/h1, 0]; pts1[12] = [182.19/w1, 273.80/h1, 0]
    pts1[13] = [275.12/w1, 247.66/h1, 0]; pts1[14] = [217.17/w1, 253.20/h1, 0]; pts1[15] = [214.99/w1, 290.00/h1, 0]; pts1[16] = [214.99/w1, 317.69/h1, 0]
    pts1[17] = [342.72/w1, 248.99/h1, 0]; pts1[18] = [344.27/w1, 160.08/h1, 0]; pts1[19] = [330.50/w1, 98.76/h1, 0]; pts1[20] = [311.18/w1, 34.65/h1, 0]

    p1_l, c1_l = classifier.classify(pts1, handedness="Left")
    assert p1_l == "I" and c1_l >= 0.92, f"User Image 1 should classify as I, got {p1_l} ({c1_l})"

    pts1_r = pts1.copy()
    pts1_r[:, 0] = 1.0 - pts1_r[:, 0]
    p1_r, c1_r = classifier.classify(pts1_r, handedness="Right")
    assert p1_r == "I" and c1_r >= 0.92, f"User Image 1 right parity should classify as I, got {p1_r} ({c1_r})"

    # 3. User Image 2 (Real ASL J)
    w2, h2 = 673.0, 495.0
    pts2 = np.zeros((21, 3), dtype=np.float32)
    pts2[0] = [522.2/w2, 369.5/h2, 0]
    pts2[1] = [505.6/w2, 253.9/h2, 0]; pts2[2] = [451.5/w2, 186.7/h2, 0]; pts2[3] = [390.1/w2, 166.3/h2, 0]; pts2[4] = [301.5/w2, 281.7/h2, 0]
    pts2[17] = [254.9/w2, 321.1/h2, 0]; pts2[18] = [169.5/w2, 287.4/h2, 0]; pts2[19] = [120.0/w2, 271.5/h2, 0]; pts2[20] = [81.4/w2, 258.4/h2, 0]
    pts2[5] = [342.1/w2, 115.0/h2, 0]; pts2[6] = [304.6/w2, 144.7/h2, 0]; pts2[7] = [289.8/w2, 206.0/h2, 0]; pts2[8] = [318.0/w2, 221.9/h2, 0]
    pts2[9] = [281.8/w2, 112.0/h2, 0]; pts2[10] = [292.5/w2, 174.8/h2, 0]; pts2[11] = [267.3/w2, 248.3/h2, 0]; pts2[12] = [342.3/w2, 184.4/h2, 0]
    pts2[13] = [259.4/w2, 176.4/h2, 0]; pts2[14] = [236.7/w2, 249.5/h2, 0]; pts2[15] = [268.7/w2, 270.0/h2, 0]; pts2[16] = [340.5/w2, 168.2/h2, 0]

    p2_l, c2_l = classifier.classify(pts2, handedness="Left")
    assert p2_l == "J" and c2_l >= 0.92, f"User Image 2 should classify as J, got {p2_l} ({c2_l})"

    pts2_r = pts2.copy()
    pts2_r[:, 0] = 1.0 - pts2_r[:, 0]
    p2_r, c2_r = classifier.classify(pts2_r, handedness="Right")
    assert p2_r == "J" and c2_r >= 0.92, f"User Image 2 right parity should classify as J, got {p2_r} ({c2_r})"

    # Image 2 must never classify as I
    assert p2_l != "I" and p2_r != "I", "J gesture must never classify as I"

    # 4. Latest User Image 1 (media_1794559828895_162f4e66.jpg - ASL I)
    w_new1, h_new1 = 496.0, 618.0
    user_i_new = np.zeros((21, 3), dtype=np.float32)
    user_i_new[0]  = [264.6 / w_new1, 494.7 / h_new1, 0.0]
    user_i_new[1]  = [177.9 / w_new1, 450.4 / h_new1, 0.0]
    user_i_new[2]  = [125.0 / w_new1, 355.5 / h_new1, 0.0]
    user_i_new[3]  = [149.5 / w_new1, 270.1 / h_new1, 0.0]
    user_i_new[4]  = [217.4 / w_new1, 268.0 / h_new1, 0.0]
    user_i_new[5]  = [126.9 / w_new1, 251.4 / h_new1, 0.0]
    user_i_new[6]  = [110.0 / w_new1, 200.0 / h_new1, 0.0]
    user_i_new[7]  = [118.2 / w_new1, 273.1 / h_new1, 0.0]
    user_i_new[8]  = [137.9 / w_new1, 237.9 / h_new1, 0.0]
    user_i_new[9]  = [184.3 / w_new1, 262.7 / h_new1, 0.0]
    user_i_new[10] = [175.0 / w_new1, 200.0 / h_new1, 0.0]
    user_i_new[11] = [175.0 / w_new1, 250.0 / h_new1, 0.0]
    user_i_new[12] = [187.9 / w_new1, 240.2 / h_new1, 0.0]
    user_i_new[13] = [245.9 / w_new1, 264.3 / h_new1, 0.0]
    user_i_new[14] = [240.0 / w_new1, 200.0 / h_new1, 0.0]
    user_i_new[15] = [230.4 / w_new1, 275.5 / h_new1, 0.0]
    user_i_new[16] = [204.6 / w_new1, 196.2 / h_new1, 0.0]
    user_i_new[17] = [307.3 / w_new1, 270.4 / h_new1, 0.0]
    user_i_new[18] = [314.3 / w_new1, 154.5 / h_new1, 0.0]
    user_i_new[19] = [320.0 / w_new1, 110.0 / h_new1, 0.0]
    user_i_new[20] = [320.6 / w_new1, 68.4 / h_new1, 0.0]

    p_in_l, c_in_l = classifier.classify(user_i_new, handedness="Left")
    assert p_in_l == "I" and c_in_l >= 0.92, f"Latest User Image 1 should classify as I, got {p_in_l}"

    user_i_new_r = np.copy(user_i_new)
    user_i_new_r[:, 0] = 1.0 - user_i_new_r[:, 0]
    p_in_r, c_in_r = classifier.classify(user_i_new_r, handedness="Right")
    assert p_in_r == "I" and c_in_r >= 0.92, f"Latest User Image 1 right parity should classify as I, got {p_in_r}"

    # 5. Latest User Image 2 (media_1794559841102_6b951183.jpg - ASL J)
    w_new2, h_new2 = 715.0, 481.0
    user_j_new = np.zeros((21, 3), dtype=np.float32)
    user_j_new[0]  = [568.5 / w_new2, 442.3 / h_new2, 0.0]
    user_j_new[1]  = [552.6 / w_new2, 288.6 / h_new2, 0.0]
    user_j_new[2]  = [506.1 / w_new2, 197.4 / h_new2, 0.0]
    user_j_new[3]  = [383.9 / w_new2, 216.6 / h_new2, 0.0]
    user_j_new[4]  = [284.5 / w_new2, 241.2 / h_new2, 0.0]
    user_j_new[5]  = [465.9 / w_new2, 107.6 / h_new2, 0.0]
    user_j_new[6]  = [310.8 / w_new2, 88.5 / h_new2, 0.0]
    user_j_new[7]  = [317.0 / w_new2, 135.3 / h_new2, 0.0]
    user_j_new[8]  = [318.9 / w_new2, 210.8 / h_new2, 0.0]
    user_j_new[9]  = [426.7 / w_new2, 154.5 / h_new2, 0.0]
    user_j_new[10] = [360.0 / w_new2, 120.0 / h_new2, 0.0]
    user_j_new[11] = [355.0 / w_new2, 150.0 / h_new2, 0.0]
    user_j_new[12] = [352.5 / w_new2, 160.0 / h_new2, 0.0]
    user_j_new[13] = [383.5 / w_new2, 222.0 / h_new2, 0.0]
    user_j_new[14] = [370.0 / w_new2, 150.0 / h_new2, 0.0]
    user_j_new[15] = [365.0 / w_new2, 160.0 / h_new2, 0.0]
    user_j_new[16] = [366.5 / w_new2, 152.8 / h_new2, 0.0]
    user_j_new[17] = [342.7 / w_new2, 292.3 / h_new2, 0.0]
    user_j_new[18] = [245.8 / w_new2, 220.7 / h_new2, 0.0]
    user_j_new[19] = [149.4 / w_new2, 204.2 / h_new2, 0.0]
    user_j_new[20] = [98.6 / w_new2, 179.1 / h_new2, 0.0]

    p_jn_l, c_jn_l = classifier.classify(user_j_new, handedness="Left")
    assert p_jn_l == "J" and c_jn_l >= 0.92, f"Latest User Image 2 should classify as J, got {p_jn_l}"

    user_j_new_r = np.copy(user_j_new)
    user_j_new_r[:, 0] = 1.0 - user_j_new_r[:, 0]
    p_jn_r, c_jn_r = classifier.classify(user_j_new_r, handedness="Right")
    assert p_jn_r == "J" and c_jn_r >= 0.92, f"Latest User Image 2 right parity should classify as J, got {p_jn_r}"
    assert p_jn_l != "I" and p_jn_r != "I", "Latest J gesture must never classify as I"




def test_strict_s_vs_t_gesture():
    """Verifies that ASL 'S' and 'T' are correctly distinguished and never confused across left/right hands."""
    from data.train_model import create_base_skeleton
    classifier = GestureClassifier()

    # 1. Base S skeleton
    base_s = create_base_skeleton("S")
    pred_s, conf_s = classifier.classify(base_s, handedness="Right")
    assert pred_s == "S" and conf_s >= 0.92, f"Base S should classify as S, got {pred_s} ({conf_s})"

    # Left-hand parity for base S
    base_s_l = base_s.copy()
    base_s_l[:, 0] = -base_s_l[:, 0]
    pred_sl, conf_sl = classifier.classify(base_s_l, handedness="Left")
    assert pred_sl == "S" and conf_sl >= 0.92, f"Left base S should classify as S, got {pred_sl} ({conf_sl})"

    # 2. Base T skeleton
    base_t = create_base_skeleton("T")
    pred_t, conf_t = classifier.classify(base_t, handedness="Right")
    assert pred_t == "T" and conf_t >= 0.92, f"Base T should classify as T, got {pred_t} ({conf_t})"

    # Left-hand parity for base T
    base_t_l = base_t.copy()
    base_t_l[:, 0] = -base_t_l[:, 0]
    pred_tl, conf_tl = classifier.classify(base_t_l, handedness="Left")
    assert pred_tl == "T" and conf_tl >= 0.92, f"Left base T should classify as T, got {pred_tl} ({conf_tl})"

    # 3. User S image landmarks (media_1794030996918_a0ea140c.jpg)
    w, h = 453.0, 477.0
    pts_user_s = np.zeros((21, 3), dtype=np.float32)
    pts_user_s[0] = [291.3/w, 379.6/h, 0]
    pts_user_s[1] = [194.9/w, 357.6/h, 0]; pts_user_s[2] = [100.5/w, 280.4/h, 0]; pts_user_s[3] = [74.7/w, 190.9/h, 0]; pts_user_s[4] = [138.2/w, 161.8/h, 0]
    pts_user_s[5] = [106.3/w, 112.1/h, 0]; pts_user_s[6] = [125.1/w, 175.2/h, 0]; pts_user_s[7] = [179.3/w, 162.5/h, 0]; pts_user_s[8] = [140.8/w, 202.8/h, 0]
    pts_user_s[9] = [163.6/w, 89.1/h, 0]; pts_user_s[10] = [179.3/w, 162.5/h, 0]; pts_user_s[11] = [190.6/w, 180.0/h, 0]; pts_user_s[12] = [190.6/w, 204.1/h, 0]
    pts_user_s[13] = [220.4/w, 74.7/h, 0]; pts_user_s[14] = [230.8/w, 153.1/h, 0]; pts_user_s[15] = [236.7/w, 180.0/h, 0]; pts_user_s[16] = [236.7/w, 202.8/h, 0]
    pts_user_s[17] = [268.3/w, 134.4/h, 0]; pts_user_s[18] = [278.8/w, 87.6/h, 0]; pts_user_s[19] = [297.0/w, 157.6/h, 0]; pts_user_s[20] = [272.8/w, 177.8/h, 0]

    p_us, c_us = classifier.classify(pts_user_s, handedness="Left")
    assert p_us == "S" and c_us >= 0.92, f"User S image should classify as S, got {p_us} ({c_us})"
    assert p_us != "T", "User S image must NEVER classify as T"

    # Right hand parity for user S image
    pts_user_s_r = pts_user_s.copy()
    pts_user_s_r[:, 0] = 1.0 - pts_user_s_r[:, 0]
    p_usr, c_usr = classifier.classify(pts_user_s_r, handedness="Right")
    assert p_usr == "S" and c_usr >= 0.92, f"User S image (Right hand) should classify as S, got {p_usr} ({c_usr})"
    assert p_usr != "T", "User S image (Right hand) must NEVER classify as T"

    # 4. User T image landmarks (media_1795414443135_8c114941.jpg)
    w_t, h_t = 610.0, 547.0
    pts_user_t = np.zeros((21, 3), dtype=np.float32)
    pts_user_t[0] = [335.4 / w_t, 506.1 / h_t, 0.0]
    pts_user_t[1] = [222.9 / w_t, 448.0 / h_t, -0.02]; pts_user_t[2] = [142.0 / w_t, 307.9 / h_t, -0.04]; pts_user_t[3] = [148.0 / w_t, 185.0 / h_t, -0.05]; pts_user_t[4] = [151.0 / w_t, 67.2 / h_t, -0.08]
    pts_user_t[5] = [147.9 / w_t, 178.6 / h_t, 0.0]; pts_user_t[6] = [83.6 / w_t, 82.3 / h_t, -0.03]; pts_user_t[7] = [82.2 / w_t, 220.2 / h_t, -0.04]; pts_user_t[8] = [106.1 / w_t, 244.6 / h_t, -0.05]
    pts_user_t[9] = [248.3 / w_t, 182.9 / h_t, 0.0]; pts_user_t[10] = [199.0 / w_t, 90.6 / h_t, -0.03]; pts_user_t[11] = [197.6 / w_t, 226.6 / h_t, -0.04]; pts_user_t[12] = [221.2 / w_t, 252.9 / h_t, -0.05]
    pts_user_t[13] = [345.6 / w_t, 201.7 / h_t, 0.0]; pts_user_t[14] = [301.0 / w_t, 141.7 / h_t, -0.03]; pts_user_t[15] = [286.0 / w_t, 273.7 / h_t, -0.04]; pts_user_t[16] = [305.3 / w_t, 290.4 / h_t, -0.05]
    pts_user_t[17] = [440.2 / w_t, 233.7 / h_t, 0.0]; pts_user_t[18] = [383.4 / w_t, 191.2 / h_t, -0.03]; pts_user_t[19] = [364.1 / w_t, 276.7 / h_t, -0.04]; pts_user_t[20] = [384.9 / w_t, 291.6 / h_t, -0.05]

    p_ut, c_ut = classifier.classify(pts_user_t, handedness="Left")
    assert p_ut == "T" and c_ut >= 0.92, f"User T image should classify as T, got {p_ut} ({c_ut})"
    assert p_ut != "A", "User T image must NEVER classify as A"
    assert p_ut != "S", "User T image must NEVER classify as S"

    # Right hand parity for user T image
    pts_user_t_r = pts_user_t.copy()
    pts_user_t_r[:, 0] = 1.0 - pts_user_t_r[:, 0]
    p_utr, c_utr = classifier.classify(pts_user_t_r, handedness="Right")
    assert p_utr == "T" and c_utr >= 0.92, f"User T image (Right hand) should classify as T, got {p_utr} ({c_utr})"
    assert p_utr != "A", "User T image (Right hand) must NEVER classify as A"
    assert p_utr != "S", "User T image (Right hand) must NEVER classify as S"




def test_strict_u_vs_v_gesture():
    """Verifies that ASL 'U' and 'V' are correctly distinguished and never confused across left/right hands."""
    from data.train_model import create_base_skeleton
    classifier = GestureClassifier()

    # 1. Base U skeleton
    base_u = create_base_skeleton("U")
    pred_u, conf_u = classifier.classify(base_u, handedness="Right")
    assert pred_u == "U" and conf_u >= 0.92, f"Base U should classify as U, got {pred_u} ({conf_u})"

    # Left-hand parity for base U
    base_u_l = base_u.copy()
    base_u_l[:, 0] = -base_u_l[:, 0]
    pred_ul, conf_ul = classifier.classify(base_u_l, handedness="Left")
    assert pred_ul == "U" and conf_ul >= 0.92, f"Left base U should classify as U, got {pred_ul} ({conf_ul})"

    # 2. Base V skeleton
    base_v = create_base_skeleton("V")
    pred_v, conf_v = classifier.classify(base_v, handedness="Right")
    assert pred_v == "V" and conf_v >= 0.92, f"Base V should classify as V, got {pred_v} ({conf_v})"

    # Left-hand parity for base V
    base_v_l = base_v.copy()
    base_v_l[:, 0] = -base_v_l[:, 0]
    pred_vl, conf_vl = classifier.classify(base_v_l, handedness="Left")
    assert pred_vl == "V" and conf_vl >= 0.92, f"Left base V should classify as V, got {pred_vl} ({conf_vl})"

    # 3. User U image landmarks (media_1795415993378_51ddc483.jpg)
    w_u, h_u = 326.0, 657.0
    pts_user_u = np.zeros((21, 3), dtype=np.float32)
    pts_user_u[0] = [197.1 / w_u, 578.3 / h_u, 0.0]
    pts_user_u[1] = [128.0 / w_u, 527.3 / h_u, -0.02]; pts_user_u[2] = [95.1 / w_u, 438.1 / h_u, -0.04]; pts_user_u[3] = [135.6 / w_u, 355.9 / h_u, -0.05]; pts_user_u[4] = [187.9 / w_u, 303.4 / h_u, -0.07]
    pts_user_u[5] = [60.6 / w_u, 317.2 / h_u, 0.0]; pts_user_u[6] = [55.9 / w_u, 205.5 / h_u, -0.03]; pts_user_u[7] = [57.5 / w_u, 137.9 / h_u, -0.04]; pts_user_u[8] = [59.0 / w_u, 81.7 / h_u, -0.05]
    pts_user_u[9] = [129.7 / w_u, 309.7 / h_u, 0.0]; pts_user_u[10] = [115.8 / w_u, 182.6 / h_u, -0.03]; pts_user_u[11] = [111.4 / w_u, 107.7 / h_u, -0.04]; pts_user_u[12] = [106.7 / w_u, 41.6 / h_u, -0.05]
    pts_user_u[13] = [246.5 / w_u, 317.3 / h_u, 0.0]; pts_user_u[14] = [197.0 / w_u, 331.0 / h_u, -0.03]; pts_user_u[15] = [186.5 / w_u, 348.6 / h_u, -0.04]; pts_user_u[16] = [174.3 / w_u, 397.0 / h_u, -0.05]
    pts_user_u[17] = [256.7 / w_u, 366.2 / h_u, 0.0]; pts_user_u[18] = [223.8 / w_u, 367.9 / h_u, -0.03]; pts_user_u[19] = [199.6 / w_u, 274.8 / h_u, -0.04]; pts_user_u[20] = [207.6 / w_u, 404.8 / h_u, -0.05]

    p_uu, c_uu = classifier.classify(pts_user_u, handedness="Left")
    assert p_uu == "U" and c_uu >= 0.92, f"User U image should classify as U, got {p_uu} ({c_uu})"
    assert p_uu != "V", "User U image must NEVER classify as V"
    assert p_uu != "R", "User U image must NEVER classify as R"
    assert p_uu != "K", "User U image must NEVER classify as K"
    assert p_uu != "None", "User U image must not be rejected as None"

    # Right hand parity for user U image
    pts_user_u_r = pts_user_u.copy()
    pts_user_u_r[:, 0] = 1.0 - pts_user_u_r[:, 0]
    p_uur, c_uur = classifier.classify(pts_user_u_r, handedness="Right")
    assert p_uur == "U" and c_uur >= 0.92, f"User U image (Right hand) should classify as U, got {p_uur} ({c_uur})"
    assert p_uur != "V", "User U image (Right hand) must NEVER classify as V"
    assert p_uur != "R", "User U image (Right hand) must NEVER classify as R"
    assert p_uur != "K", "User U image (Right hand) must NEVER classify as K"
    assert p_uur != "None", "User U image (Right hand) must not be rejected as None"

    # 4. User U image 2 landmarks (media_1795417498374_16958d73.jpg)
    pts_user_u2 = np.array([
        [0.650383, 0.717960, 0.000000],
        [0.612158, 0.684640, -0.004805],
        [0.596749, 0.597977, -0.001657],
        [0.606209, 0.530229, -0.001705],
        [0.625468, 0.494005, -0.000744],
        [0.616627, 0.559826, 0.022702],
        [0.618788, 0.512842, 0.017711],
        [0.622478, 0.480959, 0.013205],
        [0.624880, 0.463651, 0.010398],
        [0.640399, 0.559077, 0.019861],
        [0.638972, 0.502263, 0.010589],
        [0.639622, 0.473819, 0.003306],
        [0.637761, 0.459068, 0.000431],
        [0.663250, 0.564983, 0.013333],
        [0.655313, 0.515584, -0.005981],
        [0.644585, 0.561497, -0.013004],
        [0.635735, 0.613989, -0.011228],
        [0.689440, 0.573765, 0.006378],
        [0.677622, 0.526382, -0.012079],
        [0.662652, 0.531607, -0.019740],
        [0.647251, 0.550849, -0.021296],
    ], dtype=np.float32)

    p_uu2, c_uu2 = classifier.classify(pts_user_u2, handedness="Left")
    assert p_uu2 == "U" and c_uu2 >= 0.92, f"User U image 2 should classify as U, got {p_uu2} ({c_uu2})"
    assert p_uu2 != "K", "User U image 2 must NEVER classify as K"
    assert p_uu2 != "V", "User U image 2 must NEVER classify as V"
    assert p_uu2 != "R", "User U image 2 must NEVER classify as R"
    assert p_uu2 != "None", "User U image 2 must not be rejected as None"

    # Right hand parity for user U image 2
    pts_user_u2_r = pts_user_u2.copy()
    pts_user_u2_r[:, 0] = 1.0 - pts_user_u2_r[:, 0]
    p_uu2_r, c_uu2_r = classifier.classify(pts_user_u2_r, handedness="Right")
    assert p_uu2_r == "U" and c_uu2_r >= 0.92, f"User U image 2 (Right hand) should classify as U, got {p_uu2_r} ({c_uu2_r})"
    assert p_uu2_r != "K", "User U image 2 (Right hand) must NEVER classify as K"
    assert p_uu2_r != "V", "User U image 2 (Right hand) must NEVER classify as V"
    assert p_uu2_r != "R", "User U image 2 (Right hand) must NEVER classify as R"
    assert p_uu2_r != "None", "User U image 2 (Right hand) must not be rejected as None"

    # Real User U image 3 landmarks (media_1795428461295_bd0f8d04.jpg)
    pts_user_u3 = np.array([
        [0.6433, 0.6931, -0.0000],
        [0.6122, 0.6568, -0.0106],
        [0.6002, 0.5766, -0.0116],
        [0.6110, 0.5187, -0.0143],
        [0.6310, 0.4917, -0.0158],
        [0.6238, 0.5195,  0.0166],
        [0.6244, 0.4716,  0.0120],
        [0.6205, 0.4510,  0.0042],
        [0.6180, 0.4379, -0.0020],
        [0.6430, 0.5188,  0.0155],
        [0.6467, 0.4583,  0.0124],
        [0.6449, 0.4259,  0.0046],
        [0.6437, 0.4037, -0.0024],
        [0.6587, 0.5297,  0.0104],
        [0.6674, 0.4688,  0.0009],
        [0.6604, 0.4654, -0.0079],
        [0.6524, 0.4800, -0.0129],
        [0.6756, 0.5481,  0.0037],
        [0.6803, 0.4944, -0.0059],
        [0.6739, 0.4785, -0.0124],
        [0.6655, 0.4795, -0.0168],
    ], dtype=np.float32)

    p_uu3, c_uu3 = classifier.classify(pts_user_u3, handedness="Left")
    assert p_uu3 == "U" and c_uu3 >= 0.92, f"User U image 3 should classify as U, got {p_uu3} ({c_uu3})"
    assert p_uu3 != "K", "User U image 3 must NEVER classify as K"
    assert p_uu3 != "H", "User U image 3 must NEVER classify as H"
    assert p_uu3 != "N", "User U image 3 must NEVER classify as N"
    assert p_uu3 != "None", "User U image 3 must not be rejected as None"

    # Right hand parity for user U image 3
    pts_user_u3_r = pts_user_u3.copy()
    pts_user_u3_r[:, 0] = 1.0 - pts_user_u3_r[:, 0]
    p_uu3_r, c_uu3_r = classifier.classify(pts_user_u3_r, handedness="Right")
    assert p_uu3_r == "U" and c_uu3_r >= 0.92, f"User U image 3 (Right hand) should classify as U, got {p_uu3_r} ({c_uu3_r})"
    assert p_uu3_r != "K" and p_uu3_r != "H" and p_uu3_r != "N"

    # H, U, N mutual exclusivity tests:
    # 1. Base U (upright): NEVER H, NEVER N
    p_u, _ = classifier.classify(base_u, handedness="Right")
    assert p_u == "U"
    assert p_u != "H" and p_u != "N"

    # 2. Base H (horizontal): NEVER U, NEVER N
    base_h = create_base_skeleton("H")
    p_h, _ = classifier.classify(base_h, handedness="Right")
    assert p_h == "H"
    assert p_h != "U" and p_h != "N"

    # 3. Base N (downward): NEVER U, NEVER H
    base_n = create_base_skeleton("N")
    p_n, _ = classifier.classify(base_n, handedness="Right")
    assert p_n == "N"
    assert p_n != "U" and p_n != "H"




def test_strict_e_gesture():
    """Verifies that all 4 fingers curled down with thumb tucked underneath detects E with 100% left/right parity."""
    classifier = GestureClassifier(mode="hybrid")

    # Real User E image landmarks (media_1794222218533_be53b607.jpg)
    w, h = 623.0, 463.0
    pts_user_e = np.zeros((21, 3), dtype=np.float32)
    pts_user_e[0] = [382 / w, 416 / h, 0]
    pts_user_e[1] = [270 / w, 393 / h, 0]
    pts_user_e[2] = [168 / w, 324 / h, 0]
    pts_user_e[3] = [132 / w, 242 / h, 0]
    pts_user_e[4] = [201 / w, 227 / h, 0]  # Thumb tip tucked under index
    pts_user_e[5] = [238 / w, 162 / h, 0]
    pts_user_e[6] = [222 / w, 39 / h, 0]
    pts_user_e[7] = [216 / w, 96 / h, 0]
    pts_user_e[8] = [227 / w, 148 / h, 0]  # Index tip curled
    pts_user_e[9] = [307 / w, 150 / h, 0]
    pts_user_e[10] = [292 / w, 8 / h, 0]
    pts_user_e[11] = [284 / w, 86 / h, 0]
    pts_user_e[12] = [293 / w, 141 / h, 0]  # Middle tip curled
    pts_user_e[13] = [372 / w, 153 / h, 0]
    pts_user_e[14] = [363 / w, 24 / h, 0]
    pts_user_e[15] = [348 / w, 93 / h, 0]
    pts_user_e[16] = [353 / w, 154 / h, 0]  # Ring tip curled
    pts_user_e[17] = [442 / w, 173 / h, 0]
    pts_user_e[18] = [437 / w, 69 / h, 0]
    pts_user_e[19] = [420 / w, 96 / h, 0]
    pts_user_e[20] = [419 / w, 138 / h, 0]  # Pinky tip curled

    # 1. Left hand (as in user image)
    pred_l, conf_l = classifier.classify(pts_user_e, handedness="Left")
    assert pred_l == "E" and conf_l >= 0.95, f"User E image should classify as E, got {pred_l} ({conf_l:.2f})"

    # 2. Right hand parity (mirrored)
    pts_user_e_r = pts_user_e.copy()
    pts_user_e_r[:, 0] = 1.0 - pts_user_e_r[:, 0]
    pred_r, conf_r = classifier.classify(pts_user_e_r, handedness="Right")
    assert pred_r == "E" and conf_r >= 0.95, f"User E image (Right hand) should classify as E, got {pred_r} ({conf_r:.2f})"

    # 3. Must NEVER confuse with S, T, A, or D
    assert pred_l not in {"S", "T", "A", "D"}, f"User E should not be confused with {pred_l}"




def test_strict_x_gesture():
    """Verifies that index hooked back gesture classifies as X with 100% left/right parity and zero false positives."""
    from data.train_model import create_base_skeleton

    classifier_h = GestureClassifier(mode="heuristic")
    classifier_m = GestureClassifier(mode="hybrid")

    # Real User X image landmarks from media_1795418505332_2f96935b.jpg (resolution: 395x640)
    w, h = 395.0, 640.0
    pts_user_x = np.zeros((21, 3), dtype=np.float32)
    pts_user_x[0]  = [229.4 / w, 603.7 / h, 0.0]
    pts_user_x[1]  = [121.5 / w, 487.3 / h, -0.02]
    pts_user_x[2]  = [95.9 / w, 364.8 / h, -0.04]
    pts_user_x[3]  = [103.4 / w, 280.9 / h, -0.05]
    pts_user_x[4]  = [121.4 / w, 210.4 / h, -0.07]
    pts_user_x[5]  = [247.1 / w, 257.6 / h, 0.0]
    pts_user_x[6]  = [205.6 / w, 151.2 / h, -0.03]
    pts_user_x[7]  = [144.2 / w, 96.1 / h, -0.04]
    pts_user_x[8]  = [91.4 / w, 57.5 / h, -0.05]
    pts_user_x[9]  = [296.7 / w, 271.2 / h, 0.0]
    pts_user_x[10] = [136.5 / w, 199.5 / h, -0.03]
    pts_user_x[11] = [84.0 / w, 287.3 / h, -0.04]
    pts_user_x[12] = [111.7 / w, 335.9 / h, -0.05]
    pts_user_x[13] = [309.2 / w, 304.0 / h, 0.0]
    pts_user_x[14] = [141.0 / w, 237.8 / h, -0.03]
    pts_user_x[15] = [108.2 / w, 323.7 / h, -0.04]
    pts_user_x[16] = [148.2 / w, 365.0 / h, -0.05]
    pts_user_x[17] = [294.0 / w, 348.0 / h, 0.0]
    pts_user_x[18] = [162.1 / w, 284.8 / h, -0.03]
    pts_user_x[19] = [134.8 / w, 335.9 / h, -0.04]
    pts_user_x[20] = [168.8 / w, 365.0 / h, -0.05]

    # 1. Left hand (user image)
    pred_hl, conf_hl = classifier_h.classify(pts_user_x, handedness="Left")
    assert pred_hl == "X", f"User X image (heuristic) should classify as X, got {pred_hl} ({conf_hl:.2f})"
    assert conf_hl >= 0.95

    pred_ml, conf_ml = classifier_m.classify(pts_user_x, handedness="Left")
    assert pred_ml == "X", f"User X image (hybrid) should classify as X, got {pred_ml} ({conf_ml:.2f})"
    assert conf_ml >= 0.95

    # 2. Right hand parity (mirrored)
    pts_user_x_r = pts_user_x.copy()
    pts_user_x_r[:, 0] = 1.0 - pts_user_x_r[:, 0]

    pred_hr, conf_hr = classifier_h.classify(pts_user_x_r, handedness="Right")
    assert pred_hr == "X", f"User X image Right hand (heuristic) should classify as X, got {pred_hr} ({conf_hr:.2f})"
    assert conf_hr >= 0.95

    pred_mr, conf_mr = classifier_m.classify(pts_user_x_r, handedness="Right")
    assert pred_mr == "X", f"User X image Right hand (hybrid) should classify as X, got {pred_mr} ({conf_mr:.2f})"
    assert conf_mr >= 0.95

    # 3. Base skeleton X
    base_x = create_base_skeleton("X")
    pred_bx, conf_bx = classifier_m.classify(base_x, handedness="Right")
    assert pred_bx == "X", f"Base skeleton X should classify as X, got {pred_bx} ({conf_bx:.2f})"
    assert conf_bx >= 0.95

    # 4. Strict exclusion: must never classify as P, D, L, or fist letters
    assert pred_hl not in {"P", "D", "L", "A", "S", "T", "E", "M", "N"}, f"Must not be confused with {pred_hl}"
    assert pred_ml not in {"P", "D", "L", "A", "S", "T", "E", "M", "N"}, f"Must not be confused with {pred_ml}"



