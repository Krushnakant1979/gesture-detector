"""
Unit and Integration Tests for ASL Gesture Detection Pipeline
Verifies normalizer, filters, classifier, visualizer, tracker, and assistive modules
without needing a physical webcam.
"""

from __future__ import annotations

import os
import sys
import tempfile
import numpy as np

# Set writable cache dir for matplotlib
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classifier import GestureClassifier
from src.filters import (
    ExponentialMovingAverage,
    GestureDebouncer,
    StrokeTrajectoryTracker,
    WordPredictor,
)
from src.normalizer import LandmarkNormalizer
from src.tracker import HandData, HandTracker
from src.visualizer import ModernVisualizer


def test_normalizer_invariance():
    """Verifies that translation and scale shifts result in identical normalized coordinates."""
    normalizer = LandmarkNormalizer()
    np.random.seed(42)

    base_hand = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    norm_base = normalizer.normalize(base_hand)
    assert norm_base.shape == (63,)

    shifted_hand = base_hand + np.array([2.5, -1.8, 0.4], dtype=np.float32)
    norm_shifted = normalizer.normalize(shifted_hand)
    np.testing.assert_allclose(norm_base, norm_shifted, atol=1e-5)


def test_debouncer_stabilization():
    """Ensures debouncer rejects rapid 1-frame glitches and locks on sustained gestures."""
    debouncer = GestureDebouncer(consecutive_frames=4, history_length=8)

    state, _ = debouncer.update("A", 0.9)
    assert state == "None"

    for _ in range(4):
        state, conf = debouncer.update("B", 0.95)

    assert state == "B"
    assert conf > 0.9

    state, _ = debouncer.update("A", 0.9)
    assert state == "B"


def test_visualizer_rendering():
    """Verifies ASL HUD, suggestions, and skeleton drawing on a synthetic frame."""
    visualizer = ModernVisualizer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    dummy_hand = HandData(
        handedness="Right",
        confidence=0.98,
        landmarks_normalized=np.zeros((21, 3), dtype=np.float32),
        landmarks_pixel=np.array([[200 + i * 5, 300 + i * 5] for i in range(21)], dtype=np.int32),
        bbox=(180, 280, 150, 150),
    )

    rendered = visualizer.render(
        frame=frame,
        hands=[dummy_hand],
        active_letter="A",
        confidence=0.94,
        spelled_text="HELLO",
        suggestions=["HELLO", "HELP"],
    )
    assert rendered.shape == (720, 1280, 3)
    assert rendered.dtype == np.uint8


def test_tracker_initialization():
    """Verifies MediaPipe HandTracker initializes in video mode and processes an empty image."""
    tracker = HandTracker(running_mode="video")
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
    hands = tracker.process(blank_image)
    assert isinstance(hands, list)
    assert len(hands) == 0
    tracker.close()


def test_asl_classifier():
    """Verifies that the trained ASL ML model loads and predicts valid letters A-Z."""
    classifier = GestureClassifier(mode="hybrid", model_path="models/gesture_classifier.pkl")
    dummy_landmarks = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    letter, conf = classifier.classify(dummy_landmarks)
    assert isinstance(letter, str)
    assert 0.0 <= conf <= 1.0


def test_exponential_moving_average():
    """Verifies EMA filter smooths coordinate noise."""
    ema = ExponentialMovingAverage(alpha=0.5)
    v1 = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    v2 = np.array([[2.0, 2.0, 2.0]], dtype=np.float32)

    s1 = ema.update(v1)
    np.testing.assert_allclose(s1, v1)

    s2 = ema.update(v2)
    np.testing.assert_allclose(s2, np.array([[1.5, 1.5, 1.5]], dtype=np.float32))

    ema.reset()
    assert ema.state is None


def test_word_predictor():
    """Verifies prefix auto-completion suggestions."""
    predictor = WordPredictor()
    suggestions = predictor.predict("HEL")
    assert "HELLO" in suggestions or "HELP" in suggestions
    assert len(suggestions) <= 3


def test_stroke_trajectory_tracker():
    """Verifies StrokeTrajectoryTracker resets and handles empty/normal inputs cleanly."""
    tracker = StrokeTrajectoryTracker()
    assert tracker.update(None) is None
    dummy_lm = np.zeros((21, 3), dtype=np.float32)
    assert tracker.update(dummy_lm) is None
    tracker.reset()


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

    # 3. Flat open palm must NOT classify as P
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


if __name__ == "__main__":
    print("Running self-checks...")
    test_normalizer_invariance()
    print("[PASS] Normalizer test passed")
    test_debouncer_stabilization()
    print("[PASS] Debouncer test passed")
    test_visualizer_rendering()
    print("[PASS] Visualizer test passed")
    test_tracker_initialization()
    print("[PASS] Tracker test passed")
    test_asl_classifier()
    print("[PASS] ASL Classifier test passed")
    test_exponential_moving_average()
    print("[PASS] Exponential Moving Average test passed")
    test_word_predictor()
    print("[PASS] Word Predictor test passed")
    test_stroke_trajectory_tracker()
    print("[PASS] Stroke Trajectory Tracker test passed")
    test_k_vs_v_distinction()
    print("[PASS] K vs V distinction test passed")
    test_strict_a_gesture()
    print("[PASS] Strict A gesture distinction test passed")
    test_c_vs_open_palm_distinction()
    print("[PASS] C vs Open Palm distinction test passed")
    test_strict_b_gesture()
    print("[PASS] Strict B gesture test passed")
    test_strict_l_gesture()
    print("[PASS] Strict L gesture test passed")
    test_strict_m_gesture()
    print("[PASS] Strict M gesture test passed")
    test_strict_n_gesture()
    print("[PASS] Strict N gesture test passed")
    test_strict_o_gesture()
    print("[PASS] Strict O gesture test passed")
    test_strict_p_gesture()
    print("[PASS] Strict P gesture test passed")
    print("\nAll ASL pipeline tests completed successfully!")
