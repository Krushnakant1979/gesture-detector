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


def test_asl_classifier():
    """Verifies that the trained ASL ML model loads and predicts valid letters A-Z."""
    classifier = GestureClassifier(mode="hybrid", model_path="models/gesture_classifier.pkl")
    dummy_landmarks = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    letter, conf = classifier.classify(dummy_landmarks)
    assert isinstance(letter, str)
    assert 0.0 <= conf <= 1.0




def test_stroke_trajectory_tracker():
    """Verifies StrokeTrajectoryTracker resets and handles empty/normal inputs cleanly."""
    tracker = StrokeTrajectoryTracker()
    assert tracker.update(None) is None
    dummy_lm = np.zeros((21, 3), dtype=np.float32)
    assert tracker.update(dummy_lm) is None
    tracker.reset()




def test_lightgbm_and_lstm():
    """Verifies that LightGBM classifier loads and LSTMR sequence recognizer executes."""
    # LightGBM classifier
    if os.path.exists("models/lightgbm_classifier.pkl"):
        classifier = GestureClassifier(mode="ml", model_path="models/lightgbm_classifier.pkl")
        assert classifier.ml_model is not None
        dummy_landmarks = np.zeros((21, 3), dtype=np.float32)
        letter, conf = classifier.classify(dummy_landmarks)
        assert isinstance(letter, str)
        assert isinstance(conf, float)

    # LSTMR sequence recognizer
    lstm = LSTMRecurrentClassifier(input_size=63, hidden_size=48, window_size=16)
    dummy_feat = np.random.randn(63).astype(np.float32)
    for _ in range(16):
        res = lstm.update(dummy_feat)
    assert lstm.sequence_buffer.maxlen == 16
    lstm.reset()
    assert len(lstm.sequence_buffer) == 0




def test_dynamic_j_and_z_strokes():
    """Verifies temporal dynamic stroke recognition and stationary hand rejection."""
    from data.train_model import create_base_skeleton
    normalizer = LandmarkNormalizer()
    classifier = GestureClassifier(mode="hybrid")
    tracker = StrokeTrajectoryTracker()
    lstm = LSTMRecurrentClassifier()

    # 1. Stationary hand must NOT trigger J or Z
    base_i = create_base_skeleton("I")
    c_i = normalizer.normalize(base_i, canonical=True, handedness="Right")
    for _ in range(20):
        assert lstm.update(c_i) is None, "Stationary I must not trigger dynamic sign"

    base_d = create_base_skeleton("D")
    c_d = normalizer.normalize(base_d, canonical=True, handedness="Right")
    for _ in range(20):
        assert lstm.update(c_d) is None, "Stationary D must not trigger dynamic sign"

    # 2. Dynamic J stroke
    j_detected = None
    for t in range(20):
        frame_lm = base_i.copy()
        if t < 12:
            frame_lm[20, 1] += t * 0.02
            frame_lm[20, 0] -= t * 0.005
        else:
            frame_lm[20, 1] += 12 * 0.02 - (t - 12) * 0.015
            frame_lm[20, 0] -= 12 * 0.005 + (t - 12) * 0.02
        m = normalizer.analyze_hand(frame_lm, "Right")
        res = tracker.update(frame_lm, m)
        if res:
            j_detected = res
    assert j_detected is not None and j_detected[0] == "J", f"Expected J stroke, got {j_detected}"

    # 3. Dynamic Z stroke
    tracker.reset()
    z_detected = None
    for t in range(25):
        frame_lm = base_d.copy()
        if t < 8:
            frame_lm[8, 0] += t * 0.025
        elif t < 16:
            frame_lm[8, 0] += 8 * 0.025 - (t - 8) * 0.03
            frame_lm[8, 1] += (t - 8) * 0.025
        else:
            frame_lm[8, 0] += 8 * 0.025 - 8 * 0.03 + (t - 16) * 0.03
            frame_lm[8, 1] += 8 * 0.025
        m = normalizer.analyze_hand(frame_lm, "Right")
        res = tracker.update(frame_lm, m)
        if res:
            z_detected = res
    assert z_detected is not None and z_detected[0] == "Z", f"Expected Z stroke, got {z_detected}"

    # 4. Circular motion Z stroke with extended index finger (clockwise)
    tracker.reset()
    circle_z_detected = None
    for t in range(20):
        frame_lm = base_d.copy()
        angle = 2.0 * np.pi * (t / 19.0)
        frame_lm[8, 0] += 0.08 * np.cos(angle)
        frame_lm[8, 1] += 0.08 * np.sin(angle)
        m = normalizer.analyze_hand(frame_lm, "Right")
        res = tracker.update(frame_lm, m)
        if res:
            circle_z_detected = res
            pred, conf = classifier.classify(frame_lm, handedness="Right", dynamic_gesture=res[0], precomputed_m=m)
            assert pred == "Z" and conf >= 0.95, f"Classifier must return Z on dynamic circle stroke, got {pred}"
    assert circle_z_detected is not None and circle_z_detected[0] == "Z", f"Expected circular Z stroke, got {circle_z_detected}"

    # 5. Counter-clockwise circular motion Z stroke
    tracker.reset()
    ccw_z_detected = None
    for t in range(20):
        frame_lm = base_d.copy()
        angle = -2.0 * np.pi * (t / 19.0)
        frame_lm[8, 0] += 0.08 * np.cos(angle)
        frame_lm[8, 1] += 0.08 * np.sin(angle)
        m = normalizer.analyze_hand(frame_lm, "Right")
        res = tracker.update(frame_lm, m)
        if res:
            ccw_z_detected = res
            pred, conf = classifier.classify(frame_lm, handedness="Right", dynamic_gesture=res[0], precomputed_m=m)
            assert pred == "Z" and conf >= 0.95
    assert ccw_z_detected is not None and ccw_z_detected[0] == "Z", f"Expected CCW circular Z stroke, got {ccw_z_detected}"




def test_unknown_gesture_rejection():
    """Verifies that invalid or ambiguous hand shapes do not produce false alphabet predictions."""
    classifier = GestureClassifier(mode="hybrid")

    # Degenerate hand (all zeros or tiny span)
    zeros = np.zeros((21, 3), dtype=np.float32)
    pred, conf = classifier.classify(zeros)
    assert pred == "None" or conf < 0.65

    # Random noise hand
    np.random.seed(99)
    noise_hand = np.random.uniform(-0.1, 0.1, (21, 3)).astype(np.float32)
    pred_n, conf_n = classifier.classify(noise_hand)
    assert pred_n == "None" or conf_n < 0.70

    # Downward dangling/claw random hand (media_1794225829454_535fd193.jpg)
    # Must reject false 'M', 'C', or any alphabet with < 0.92 confidence / "None"
    random_claw = np.array([
        [0.2551, 0.3869, 0.0000],
        [0.2965, 0.5234, -0.0204],
        [0.3767, 0.6329, -0.0303],
        [0.4356, 0.7190, -0.0389],
        [0.4813, 0.8011, -0.0459],
        [0.4590, 0.5098, -0.0155],
        [0.5425, 0.6550, -0.0404],
        [0.5890, 0.7593, -0.0607],
        [0.6224, 0.8403, -0.0734],
        [0.4633, 0.4922, -0.0191],
        [0.5554, 0.6483, -0.0421],
        [0.6032, 0.7633, -0.0628],
        [0.6368, 0.8564, -0.0764],
        [0.4524, 0.4900, -0.0261],
        [0.5382, 0.6360, -0.0473],
        [0.5827, 0.7453, -0.0620],
        [0.6155, 0.8293, -0.0713],
        [0.4336, 0.4985, -0.0347],
        [0.4931, 0.6136, -0.0509],
        [0.5299, 0.6996, -0.0580],
        [0.5578, 0.7693, -0.0619],
    ], dtype=np.float32)
    pred_claw_r, conf_claw_r = classifier.classify(random_claw, handedness="Right")
    assert pred_claw_r == "None" or conf_claw_r < 0.92, f"Random downward claw hand must be rejected, got {pred_claw_r} ({conf_claw_r:.2f})"
    assert pred_claw_r != "M", f"Random dangling hand must NOT classify as M"
    assert pred_claw_r != "C", f"Random dangling hand must NOT classify as C"

    # Left-hand parity check
    random_claw_l = random_claw.copy()
    random_claw_l[:, 0] = 1.0 - random_claw_l[:, 0]
    pred_claw_l, conf_claw_l = classifier.classify(random_claw_l, handedness="Left")
    assert pred_claw_l == "None" or conf_claw_l < 0.92, f"Random downward claw left hand must be rejected, got {pred_claw_l}"
    assert pred_claw_l != "M"
    assert pred_claw_l != "C"




def test_lstmr_dynamic_j_and_z_trajectories():
    """Verifies LSTMR sequence recognition on full strokes and rejection of incomplete or stationary gestures."""
    from data.train_model import create_base_skeleton
    normalizer = LandmarkNormalizer()
    lstm = LSTMRecurrentClassifier(model_path="models/lstm_gesture_model.npz", window_size=20)
    base_i = create_base_skeleton("I")
    base_d = create_base_skeleton("D")
    c_i = normalizer.normalize(base_i, canonical=True)
    c_d = normalizer.normalize(base_d, canonical=True)

    # 1. Full dynamic J sequence (downward plunge + upward curved hook)
    lstm.reset()
    j_res = None
    for t in range(20):
        feat = np.zeros(63, dtype=np.float32)
        if t < 12:
            p = t / 12.0
            feat[0] = -p * 0.02
            feat[1] = p * 0.22
        else:
            p = (t - 12) / 8.0
            feat[0] = -0.02 - p * 0.08
            feat[1] = 0.22 - p * 0.08
        feat[3:] = c_i[3:]
        # Fingertip position in image space
        raw_lm = base_i.copy()
        raw_lm[20, 0] += feat[0]
        raw_lm[20, 1] += feat[1]
        res = lstm.update(feat, raw_landmarks=raw_lm)
        if res:
            j_res = res
    assert j_res is not None and j_res[0] == "J", f"Expected dynamic J recognition, got {j_res}"
    assert j_res[1] >= 0.80, f"J confidence too low: {j_res[1]:.2f}"

    # 2. Full dynamic Z sequence (3-phase zigzag: right, diagonal down-left, right)
    lstm.reset()
    z_res = None
    for t in range(20):
        feat = np.zeros(63, dtype=np.float32)
        if t < 7:
            p = t / 7.0
            feat[0] = p * 0.18
            feat[1] = 0.0
        elif t < 14:
            p = (t - 7) / 7.0
            feat[0] = (1.0 - p) * 0.18
            feat[1] = p * 0.18
        else:
            p = (t - 14) / 6.0
            feat[0] = p * 0.18
            feat[1] = 0.18
        feat[3:] = c_d[3:]
        raw_lm = base_d.copy()
        raw_lm[8, 0] += feat[0]
        raw_lm[8, 1] += feat[1]
        res = lstm.update(feat, raw_landmarks=raw_lm)
        if res:
            z_res = res
    assert z_res is not None and z_res[0] == "Z", f"Expected dynamic Z recognition, got {z_res}"
    assert z_res[1] >= 0.80, f"Z confidence too low: {z_res[1]:.2f}"

    # 3. Incomplete J sequence (only moved down, never hooked back up) -> Must NOT recognize J
    lstm.reset()
    incomplete_j_res = None
    for t in range(20):
        feat = np.zeros(63, dtype=np.float32)
        p = min(1.0, t / 12.0)
        feat[0] = 0.0
        feat[1] = p * 0.10
        feat[3:] = c_i[3:]
        raw_lm = base_i.copy()
        raw_lm[20, 1] += feat[1]
        res = lstm.update(feat, raw_landmarks=raw_lm)
        if res:
            incomplete_j_res = res
    assert incomplete_j_res is None, f"Incomplete J stroke should be rejected, got {incomplete_j_res}"

    # 4. Incomplete Z sequence (only moved right, never completed zigzag) -> Must NOT recognize Z
    lstm.reset()
    incomplete_z_res = None
    for t in range(20):
        feat = np.zeros(63, dtype=np.float32)
        p = min(1.0, t / 8.0)
        feat[0] = p * 0.12
        feat[1] = 0.0
        feat[3:] = c_d[3:]
        raw_lm = base_d.copy()
        raw_lm[8, 0] += feat[0]
        res = lstm.update(feat, raw_landmarks=raw_lm)
        if res:
            incomplete_z_res = res
    assert incomplete_z_res is None, f"Incomplete Z stroke should be rejected, got {incomplete_z_res}"

    # 5. Stationary hand (holding position with slight tremor) -> Must return None
    lstm.reset()
    for _ in range(20):
        feat = np.zeros(63, dtype=np.float32)
        feat[0] = np.random.randn() * 0.003
        feat[1] = np.random.randn() * 0.003
        feat[3:] = c_i[3:]
        assert lstm.update(feat) is None, "Stationary hand must not trigger dynamic sign"




def test_full_alphabet_classification_report():
    """Generates a complete precision, recall, F1, and false-positive evaluation across all 26 ASL alphabets."""
    from collections import defaultdict
    from data.train_model import create_base_skeleton, STATIC_ASL_CLASSES

    classifier = GestureClassifier(mode="hybrid")
    all_classes = STATIC_ASL_CLASSES + ["J", "Z"]

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    correct_cnt = 0
    total_cnt = 0

    # 1. Test 24 static alphabets with variations
    for letter in STATIC_ASL_CLASSES:
        base = create_base_skeleton(letter)
        # Test Right and Left
        for handedness in ["Right", "Left"]:
            lm = base.copy()
            if handedness == "Left":
                lm[:, 0] = -lm[:, 0]
            pred, conf = classifier.classify(lm, handedness=handedness)
            total_cnt += 1
            if pred == letter:
                tp[letter] += 1
                correct_cnt += 1
            else:
                fn[letter] += 1
                if pred != "None":
                    fp[pred] += 1

    # 2. Test Dynamic J and Z
    tracker = StrokeTrajectoryTracker()
    base_i = create_base_skeleton("I")
    base_d = create_base_skeleton("D")
    normalizer = LandmarkNormalizer()

    # Dynamic J test
    tracker.reset()
    j_pred = None
    for t in range(20):
        f = base_i.copy()
        if t < 12:
            f[20, 1] += t * 0.02
        else:
            f[20, 1] += 12 * 0.02 - (t - 12) * 0.015
            f[20, 0] -= (t - 12) * 0.02
        m = normalizer.analyze_hand(f, "Right")
        stroke = tracker.update(f, m)
        pred, _ = classifier.classify(f, handedness="Right", dynamic_gesture=stroke[0] if stroke else None)
        if pred == "J":
            j_pred = pred
    total_cnt += 1
    if j_pred == "J":
        tp["J"] += 1
        correct_cnt += 1
    else:
        fn["J"] += 1

    # Dynamic Z test
    tracker.reset()
    z_pred = None
    for t in range(25):
        f = base_d.copy()
        if t < 8:
            f[8, 0] += t * 0.025
        elif t < 16:
            f[8, 0] += 8 * 0.025 - (t - 8) * 0.03
            f[8, 1] += (t - 8) * 0.025
        else:
            f[8, 0] += 8 * 0.025 - 8 * 0.03 + (t - 16) * 0.03
            f[8, 1] += 8 * 0.025
        m = normalizer.analyze_hand(f, "Right")
        stroke = tracker.update(f, m)
        pred, _ = classifier.classify(f, handedness="Right", dynamic_gesture=stroke[0] if stroke else None)
        if pred == "Z":
            z_pred = pred
    total_cnt += 1
    if z_pred == "Z":
        tp["Z"] += 1
        correct_cnt += 1
    else:
        fn["Z"] += 1

    # 3. Test Negative / Unknown Gestures (Open palm, noise) -> Compute FPR
    negative_tests = 20
    fp_neg = 0
    # Open palm
    open_palm = np.zeros((21, 3), dtype=np.float32)
    open_palm[0] = [0, 0, 0]
    for idx, tip in [(5, 8), (9, 12), (13, 16), (17, 20)]:
        open_palm[idx] = [0.1 * (idx - 9), -0.5, 0.0]
        open_palm[tip] = [0.12 * (idx - 9), -0.95, 0.0]
    open_palm[4] = [0.65, -0.45, 0.0]

    for _ in range(negative_tests):
        noise = open_palm + np.random.uniform(-0.01, 0.01, (21, 3)).astype(np.float32)
        p, _ = classifier.classify(noise, handedness="Right")
        if p != "None":
            fp_neg += 1

    fpr = (fp_neg / negative_tests) * 100

    print("\n" + "=" * 65)
    print(" COMPLETE ASL ALPHABET EVALUATION REPORT (ALL 26 ALPHABETS)")
    print("=" * 65)
    print(f"{'Alphabet':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Status'}")
    print("-" * 65)

    precisions = []
    recalls = []
    f1s = []

    for letter in all_classes:
        t_pos = tp[letter]
        f_pos = fp[letter]
        f_neg = fn[letter]

        prec = t_pos / (t_pos + f_pos) if (t_pos + f_pos) > 0 else 0.0
        rec = t_pos / (t_pos + f_neg) if (t_pos + f_neg) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        status = "PASSED" if f1 >= 0.90 else "NEEDS REVIEW"
        print(f"{letter:<10} {prec * 100:>8.1f}%   {rec * 100:>8.1f}%   {f1 * 100:>8.1f}%    {status}")

    avg_prec = np.mean(precisions) * 100
    avg_rec = np.mean(recalls) * 100
    avg_f1 = np.mean(f1s) * 100

    print("-" * 65)
    print(f"Overall Accuracy:       {correct_cnt}/{total_cnt} ({correct_cnt/total_cnt*100:.2f}%)")
    print(f"Macro Average Precision: {avg_prec:.2f}%")
    print(f"Macro Average Recall:    {avg_rec:.2f}%")
    print(f"Macro Average F1-Score:  {avg_f1:.2f}%")
    print(f"False-Positive Rate:     {fpr:.2f}% on unknown/open-palm poses")
    print("=" * 65)

    assert correct_cnt == total_cnt, f"Expected 100% alphabet pass rate, got {correct_cnt}/{total_cnt}"
    assert fpr == 0.0, f"Expected 0.0% false positive rate on negative poses, got {fpr:.2f}%"



