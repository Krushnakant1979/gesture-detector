"""
Model Training Script for ASL Alphabet Classification
Supports training:
- High-Performance LightGBM Gradient Boosted Decision Forest (`models/lightgbm_classifier.pkl`)
- Random Forest Ensemble (`models/gesture_classifier.pkl`)
- LSTMR Recurrent Gesture Sequence Recognizer (`models/lstm_gesture_model.npz`)
Supports both synthetic anatomical patterns and real-world recorded landmark datasets.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lstm_recognizer import LSTMRecurrentClassifier
from src.normalizer import LandmarkNormalizer

STATIC_ASL_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
]


def create_base_skeleton(cls: str) -> np.ndarray:
    """
    Returns 21 3D landmarks for ASL static sign 'cls' in anatomical meters.
    Wrist at (0, 0, 0), fingers pointing generally +Y in neutral hand frame.
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    # Wrist (0)
    lm[0] = [0.0, 0.0, 0.0]

    # Base palm MCPs
    lm[5]  = [-0.035, 0.16, 0.005]  # Index MCP
    lm[9]  = [0.0,    0.17, 0.0]    # Middle MCP
    lm[13] = [0.032,  0.16, -0.005] # Ring MCP
    lm[17] = [0.055,  0.14, -0.01]  # Pinky MCP

    def extend_finger(mcp_idx: int, dir_vec: np.ndarray, l1=0.045, l2=0.035, l3=0.030):
        u = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)
        mcp = lm[mcp_idx]
        pip = mcp + u * l1
        dip = pip + u * l2
        tip = dip + u * l3
        lm[mcp_idx + 1] = pip
        lm[mcp_idx + 2] = dip
        lm[mcp_idx + 3] = tip

    def curl_finger(mcp_idx: int, tight: bool = True):
        mcp = lm[mcp_idx]
        pip = mcp + [0.0, 0.025, 0.020]
        dip = mcp + [0.0, 0.005, 0.030]
        tip = mcp + [0.0, -0.015, 0.015] if tight else mcp + [0.0, -0.005, 0.025]
        lm[mcp_idx + 1] = pip
        lm[mcp_idx + 2] = dip
        lm[mcp_idx + 3] = tip

    def arch_finger(mcp_idx: int, forward_c: float = 0.045):
        mcp = lm[mcp_idx]
        pip = mcp + [0.0, 0.030, forward_c]
        dip = mcp + [0.0, 0.015, forward_c * 1.5]
        tip = mcp + [0.0, -0.010, forward_c * 1.6]
        lm[mcp_idx + 1] = pip
        lm[mcp_idx + 2] = dip
        lm[mcp_idx + 3] = tip

    # Default thumb base
    lm[1] = [-0.03, 0.04, 0.005]
    lm[2] = [-0.05, 0.08, 0.015]

    if cls == "A":
        # All 4 fingers clenched into fist, thumb upright snug alongside index
        curl_finger(5); curl_finger(9); curl_finger(13); curl_finger(17)
        lm[3] = [-0.045, 0.12, 0.015]
        lm[4] = [-0.040, 0.16, 0.015]
    elif cls == "B":
        # 4 fingers extended straight up together, thumb folded flat across palm
        extend_finger(5,  [-0.005, 1.0, 0.0])
        extend_finger(9,  [ 0.0,   1.0, 0.0])
        extend_finger(13, [ 0.005, 1.0, 0.0])
        extend_finger(17, [ 0.010, 1.0, 0.0])
        lm[3] = [-0.02, 0.10, 0.025]
        lm[4] = [ 0.01, 0.11, 0.025]
    elif cls == "C":
        # All 4 fingers arched in C curve, thumb curved opposite forming lower arc
        arch_finger(5, 0.040); arch_finger(9, 0.042); arch_finger(13, 0.040); arch_finger(17, 0.038)
        lm[2] = [-0.04, 0.07, 0.02]
        lm[3] = [-0.02, 0.09, 0.05]
        lm[4] = [ 0.00, 0.08, 0.07]
    elif cls == "D":
        # Index straight up, middle/ring/pinky curled, middle touching thumb
        extend_finger(5, [0.0, 1.0, 0.0])
        curl_finger(9); curl_finger(13); curl_finger(17)
        lm[3] = [-0.02, 0.10, 0.03]
        lm[4] = [ 0.00, 0.12, 0.03]
        lm[12] = [0.00, 0.12, 0.03]
    elif cls == "E":
        # All 4 fingers curled down tightly, thumb folded horizontally under fingertips
        curl_finger(5); curl_finger(9); curl_finger(13); curl_finger(17)
        lm[3] = [-0.02, 0.09, 0.025]
        lm[4] = [ 0.02, 0.09, 0.025]
    elif cls == "F":
        # Index and thumb form circle, middle, ring, pinky straight up and spread
        extend_finger(9,  [-0.05, 1.0, 0.0])
        extend_finger(13, [ 0.00, 1.0, 0.0])
        extend_finger(17, [ 0.06, 1.0, 0.0])
        arch_finger(5, 0.035)
        lm[3] = [-0.03, 0.11, 0.04]
        lm[4] = [-0.02, 0.14, 0.05]
        lm[8] = [-0.02, 0.14, 0.05]
    elif cls == "G":
        # Index extended horizontally, thumb parallel above, others curled
        extend_finger(5, [1.0, 0.05, 0.0])
        curl_finger(9); curl_finger(13); curl_finger(17)
        lm[2] = [-0.02, 0.10, 0.01]
        lm[3] = [ 0.01, 0.13, 0.01]
        lm[4] = [ 0.05, 0.13, 0.01]
    elif cls == "H":
        # Index and middle extended horizontally together, thumb tucked, others curled
        extend_finger(5, [1.0, 0.05, 0.0])
        extend_finger(9, [1.0, 0.00, 0.0])
        curl_finger(13); curl_finger(17)
        lm[3] = [-0.02, 0.09, 0.02]
        lm[4] = [ 0.00, 0.09, 0.02]
    elif cls == "I":
        # Pinky extended straight up, index/middle/ring curled, thumb over fingers
        extend_finger(17, [0.0, 1.0, 0.0])
        curl_finger(5); curl_finger(9); curl_finger(13)
        lm[3] = [-0.02, 0.10, 0.03]
        lm[4] = [ 0.01, 0.10, 0.03]
    elif cls == "K":
        # Index straight up, middle up/forward, thumb elevated between index and middle
        extend_finger(5, [-0.04, 1.0, 0.0])
        extend_finger(9, [ 0.04, 0.95, 0.15])
        curl_finger(13); curl_finger(17)
        lm[3] = [-0.01, 0.14, 0.05]
        lm[4] = [ 0.00, 0.18, 0.08]
    elif cls == "L":
        # Index straight up, thumb extended horizontally at 90 deg, others curled
        extend_finger(5, [0.0, 1.0, 0.0])
        curl_finger(9); curl_finger(13); curl_finger(17)
        lm[2] = [-0.06, 0.06, 0.0]
        lm[3] = [-0.10, 0.07, 0.0]
        lm[4] = [-0.14, 0.07, 0.0]
    elif cls == "M":
        # 4 fingers curled over thumb, thumb tip emerges between ring & pinky
        curl_finger(5); curl_finger(9); curl_finger(13); curl_finger(17)
        lm[3] = [0.01, 0.11, 0.02]
        lm[4] = [0.04, 0.12, 0.025]
    elif cls == "N":
        # 4 fingers curled over thumb, thumb tip emerges between middle and ring
        curl_finger(5); curl_finger(9); curl_finger(13); curl_finger(17)
        lm[3] = [-0.01, 0.11, 0.02]
        lm[4] = [ 0.02, 0.12, 0.025]
    elif cls == "O":
        # All 4 fingertips curve forward to meet thumb tip in an open circle
        arch_finger(5, 0.055); arch_finger(9, 0.055); arch_finger(13, 0.055); arch_finger(17, 0.050)
        lm[2] = [-0.03, 0.08, 0.03]
        lm[3] = [-0.01, 0.12, 0.06]
        lm[4] = [ 0.00, 0.14, 0.08]
        lm[8]  = [ 0.00, 0.14, 0.08]
        lm[12] = [ 0.00, 0.14, 0.08]
        lm[16] = [ 0.01, 0.13, 0.08]
        lm[20] = [ 0.02, 0.13, 0.07]
    elif cls == "P":
        # Downward K: hand directed down, index forward, middle down, thumb near middle PIP
        extend_finger(5, [0.0, 0.5, 0.8])
        extend_finger(9, [0.0, -0.8, 0.4])
        curl_finger(13); curl_finger(17)
        lm[3] = [-0.01, 0.10, 0.04]
        lm[4] = [ 0.00, 0.12, 0.05]
    elif cls == "Q":
        # Downward G: hand directed down, index and thumb pointing down together
        extend_finger(5, [0.1, -1.0, 0.0])
        curl_finger(9); curl_finger(13); curl_finger(17)
        lm[2] = [-0.02, 0.08, 0.01]
        lm[3] = [ 0.00, 0.03, 0.01]
        lm[4] = [ 0.01, -0.03, 0.01]
    elif cls == "R":
        # Index and middle extended up and crossed, others curled, thumb across
        extend_finger(5, [ 0.18, 1.0, -0.015])
        extend_finger(9, [-0.18, 1.0,  0.015])
        curl_finger(13); curl_finger(17)
        lm[3] = [-0.02, 0.10, 0.02]
        lm[4] = [ 0.01, 0.10, 0.02]
    elif cls == "S":
        # Fist with thumb wrapped across front of index and middle
        curl_finger(5); curl_finger(9); curl_finger(13); curl_finger(17)
        lm[2] = [-0.03, 0.09, 0.03]
        lm[3] = [ 0.00, 0.11, 0.035]
        lm[4] = [ 0.02, 0.12, 0.035]
    elif cls == "T":
        # Fist with thumb tip tucked between index and middle
        curl_finger(5); curl_finger(9); curl_finger(13); curl_finger(17)
        lm[2] = [-0.03, 0.09, 0.02]
        lm[3] = [-0.02, 0.12, 0.03]
        lm[4] = [-0.01, 0.15, 0.035]
    elif cls == "U":
        # Index and middle straight up touching together, others curled
        extend_finger(5, [ 0.18, 1.0, 0.0])
        extend_finger(9, [-0.10, 1.0, 0.0])
        curl_finger(13); curl_finger(17)
        lm[3] = [-0.02, 0.10, 0.02]
        lm[4] = [ 0.01, 0.10, 0.02]
    elif cls == "V":
        # Index and middle straight up spread apart (peace sign), others curled
        extend_finger(5, [-0.08, 0.98, 0.0])
        extend_finger(9, [ 0.08, 0.98, 0.0])
        curl_finger(13); curl_finger(17)
        lm[3] = [-0.02, 0.10, 0.02]
        lm[4] = [ 0.01, 0.10, 0.02]
    elif cls == "W":
        # Index, middle, ring straight up and spread, pinky curled with thumb
        extend_finger(5,  [-0.15, 0.98, 0.0])
        extend_finger(9,  [ 0.00, 1.00, 0.0])
        extend_finger(13, [ 0.15, 0.98, 0.0])
        curl_finger(17)
        lm[3] = [0.01, 0.09, 0.02]
        lm[4] = [0.03, 0.10, 0.02]
    elif cls == "X":
        # Index hooked (PIP up, tip down), others curled, thumb tucked
        mcp = lm[5]
        lm[6] = mcp + [0.0, 0.040, 0.01]
        lm[7] = lm[6] + [0.0, 0.005, 0.03]
        lm[8] = lm[7] + [0.0, -0.025, 0.01]
        curl_finger(9); curl_finger(13); curl_finger(17)
        lm[3] = [-0.02, 0.09, 0.02]
        lm[4] = [ 0.00, 0.09, 0.02]
    elif cls == "Y":
        # Thumb and pinky extended wide, index, middle, ring curled
        extend_finger(17, [0.30, 0.90, 0.0])
        curl_finger(5); curl_finger(9); curl_finger(13)
        lm[2] = [-0.06, 0.07, 0.0]
        lm[3] = [-0.10, 0.09, 0.0]
        lm[4] = [-0.14, 0.11, 0.0]
    else:
        curl_finger(5); curl_finger(9); curl_finger(13); curl_finger(17)

    return lm


def generate_asl_dataset(output_csv: str, samples_per_class: int = 120) -> None:
    """
    Generates a calibrated synthetic dataset based on anatomical landmark patterns
    for ASL alphabet static letters A through Y with 3D canonical coordinate invariance.
    """
    print(f"[INFO] Generating calibrated canonical ASL dataset ({len(STATIC_ASL_CLASSES)} static classes)...")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    np.random.seed(42)

    normalizer = LandmarkNormalizer()
    X = []
    y = []

    for cls in STATIC_ASL_CLASSES:
        base = create_base_skeleton(cls)
        for _ in range(samples_per_class):
            scale = float(np.random.uniform(0.88, 1.12))
            lm = base * scale

            noise = np.random.normal(0.0, 0.003, lm.shape).astype(np.float32)
            noise[0] = 0.0
            lm = lm + noise

            euler = np.random.uniform(-25, 25, 3)
            rot = R_scipy.from_euler("xyz", euler, degrees=True).as_matrix().astype(np.float32)
            lm = lm @ rot.T

            handedness = "Right" if np.random.rand() > 0.5 else "Left"
            if handedness == "Left":
                lm[:, 0] = -lm[:, 0]

            c_lm, _ = normalizer.canonical_align(lm, handedness=handedness)
            X.append(c_lm.flatten())
            y.append(cls)

    header = "label," + ",".join([f"f_{i}" for i in range(63)])
    lines = [header]
    for label, feat in zip(y, X):
        row = f"{label}," + ",".join([f"{v:.6f}" for v in feat])
        lines.append(row)

    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[SUCCESS] Calibrated ASL dataset saved with {len(X)} samples across {len(STATIC_ASL_CLASSES)} static letters.")


def load_dataset(dataset_path: str, real_world_csv: str = "data/real_world_landmarks.csv"):
    """Loads dataset from synthetic and/or real-world landmark recordings in canonical space."""
    if not os.path.exists(dataset_path):
        generate_asl_dataset(dataset_path)

    data = []
    labels = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for parts in reader:
            if len(parts) >= 64:
                lbl = parts[0].strip().upper()
                # Exclude dynamic J and Z from static ML model training
                if lbl in STATIC_ASL_CLASSES:
                    labels.append(lbl)
                    data.append([float(x) for x in parts[1:64]])

    if os.path.exists(real_world_csv):
        real_count = 0
        try:
            with open(real_world_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lbl = row.get("label", "").strip().upper()
                    if lbl in STATIC_ASL_CLASSES and "c0" in row:
                        c_feats = [float(row[f"c{i}"]) for i in range(63)]
                        labels.append(lbl)
                        data.append(c_feats)
                        real_count += 1
            print(f"[INFO] Merged {real_count} real-world captured landmark frames from '{real_world_csv}'.")
        except Exception as e:
            print(f"[WARN] Error loading real-world dataset: {e}")

    X = np.array(data, dtype=np.float32)
    y = np.array(labels)
    return X, y


def generate_dynamic_dataset(
    num_samples_per_class: int = 150,
    seq_len: int = 20,
    real_world_csv: str = "data/dynamic_sequences.csv",
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generates balanced dynamic sequence dataset for J, Z, and STATIC gestures.
    Also merges real-world recorded sequences from real_world_csv if available.
    """
    normalizer = LandmarkNormalizer()
    base_i = create_base_skeleton("I")
    base_d = create_base_skeleton("D")
    base_skeletons = [create_base_skeleton(c) for c in STATIC_ASL_CLASSES]

    c_i = normalizer.normalize(base_i, canonical=True)
    c_d = normalizer.normalize(base_d, canonical=True)
    canonical_bases = [normalizer.normalize(b, canonical=True) for b in base_skeletons]

    X_seqs: List[np.ndarray] = []
    y_labels: List[str] = []

    # 1. J sequences (Pinky downward plunge followed by curved hook upward/left)
    for _ in range(num_samples_per_class):
        seq = np.zeros((seq_len, 63), dtype=np.float32)
        speed = np.random.uniform(0.75, 1.25)
        scale = np.random.uniform(0.85, 1.15)
        # Random initial wrist offset
        w0 = np.random.uniform(-0.1, 0.1, 3)

        for t in range(seq_len):
            if t < int(seq_len * 0.6):
                p = t / (seq_len * 0.6)
                dx = -p * 0.02 * speed * scale
                dy = p * 0.22 * speed * scale
            else:
                p = (t - seq_len * 0.6) / (seq_len * 0.4)
                dx = (-0.02 - p * 0.08) * speed * scale
                dy = (0.22 - p * 0.08) * speed * scale

            seq[t, 0] = dx + np.random.randn() * 0.003
            seq[t, 1] = dy + np.random.randn() * 0.003
            seq[t, 2] = np.random.randn() * 0.003
            # Canonical hand shape: 'I' (pinky upright)
            seq[t, 3:] = c_i[3:] + np.random.randn(60) * 0.004

        X_seqs.append(seq)
        y_labels.append("J")

    # 2. Z sequences (Index finger 3-phase stroke: right, diagonal down-left, right)
    for _ in range(num_samples_per_class):
        seq = np.zeros((seq_len, 63), dtype=np.float32)
        speed = np.random.uniform(0.75, 1.25)
        scale = np.random.uniform(0.85, 1.15)

        for t in range(seq_len):
            if t < int(seq_len * 0.35):
                p = t / (seq_len * 0.35)
                dx = p * 0.18 * speed * scale
                dy = 0.0
            elif t < int(seq_len * 0.70):
                p = (t - seq_len * 0.35) / (seq_len * 0.35)
                dx = (1.0 - p) * 0.18 * speed * scale
                dy = p * 0.18 * speed * scale
            else:
                p = (t - seq_len * 0.70) / (seq_len * 0.30)
                dx = p * 0.18 * speed * scale
                dy = 0.18 * speed * scale

            seq[t, 0] = dx + np.random.randn() * 0.003
            seq[t, 1] = dy + np.random.randn() * 0.003
            seq[t, 2] = np.random.randn() * 0.003
            # Canonical hand shape: 'D' (index upright)
            seq[t, 3:] = c_d[3:] + np.random.randn(60) * 0.004

        X_seqs.append(seq)
        y_labels.append("Z")

    # 3. STATIC / Negative sequences (stationary hands, tremors, and incomplete partial strokes)
    for i in range(num_samples_per_class):
        seq = np.zeros((seq_len, 63), dtype=np.float32)
        base = canonical_bases[i % len(canonical_bases)]

        if i % 3 == 0:
            # Fully stationary hand with physiological micro-tremor
            for t in range(seq_len):
                seq[t, 0] = np.random.randn() * 0.006
                seq[t, 1] = np.random.randn() * 0.006
                seq[t, 2] = np.random.randn() * 0.006
                seq[t, 3:] = base[3:] + np.random.randn(60) * 0.004
        elif i % 3 == 1:
            # Incomplete J stroke (only moved down, never hooked) -> Must be rejected as STATIC
            for t in range(seq_len):
                p = min(1.0, t / (seq_len * 0.6))
                seq[t, 0] = np.random.randn() * 0.004
                seq[t, 1] = p * 0.10 + np.random.randn() * 0.004
                seq[t, 3:] = c_i[3:] + np.random.randn(60) * 0.004
        else:
            # Incomplete Z stroke (only moved right, never completed zigzag) -> Must be rejected as STATIC
            for t in range(seq_len):
                p = min(1.0, t / (seq_len * 0.4))
                seq[t, 0] = p * 0.12 + np.random.randn() * 0.004
                seq[t, 1] = np.random.randn() * 0.004
                seq[t, 3:] = c_d[3:] + np.random.randn(60) * 0.004

        X_seqs.append(seq)
        y_labels.append("STATIC")

    # Merge real-world dynamic sequences if available
    if os.path.exists(real_world_csv):
        try:
            seq_dict: Dict[str, List[np.ndarray]] = {}
            seq_labels: Dict[str, str] = {}
            with open(real_world_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = row.get("seq_id", "")
                    lbl = row.get("label", "").strip().upper()
                    if sid and lbl in ("J", "Z", "STATIC"):
                        feats = [float(row[f"c{i}"]) for i in range(63)]
                        seq_dict.setdefault(sid, []).append(np.array(feats, dtype=np.float32))
                        seq_labels[sid] = lbl

            loaded_count = 0
            for sid, frames in seq_dict.items():
                if len(frames) >= 8:
                    arr = np.array(frames, dtype=np.float32)
                    if len(arr) != seq_len:
                        # Resample to seq_len
                        idxs = np.linspace(0, len(arr) - 1, seq_len).astype(int)
                        arr = arr[idxs]
                    X_seqs.append(arr)
                    y_labels.append(seq_labels[sid])
                    loaded_count += 1
            print(f"[INFO] Merged {loaded_count} real-world dynamic sequences from '{real_world_csv}'.")
        except Exception as e:
            print(f"[WARN] Error loading real-world dynamic sequences: {e}")

    return X_seqs, y_labels


def train_models(
    dataset_path: str = "data/asl_dataset.csv",
    rf_output_path: str = "models/gesture_classifier.pkl",
    lgbm_output_path: str = "models/lightgbm_classifier.pkl",
    lstm_output_path: str = "models/lstm_gesture_model.npz",
    model_type: str = "all",
    regenerate_dataset: bool = False,
) -> None:
    if regenerate_dataset or not os.path.exists(dataset_path):
        generate_asl_dataset(dataset_path)

    X, y = load_dataset(dataset_path)
    unique_classes = sorted(np.unique(y))
    print(f"[INFO] Dataset ready: {len(X)} samples across {len(unique_classes)} classes: {unique_classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 1. Train Random Forest
    if model_type in ("rf", "all"):
        print("\n[INFO] Training Random Forest ASL Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        print(f"[RESULTS - Random Forest] Evaluation Accuracy: {acc_rf * 100:.2f}%")

        os.makedirs(os.path.dirname(rf_output_path), exist_ok=True)
        with open(rf_output_path, "wb") as f:
            pickle.dump(rf_model, f)
        print(f"[SUCCESS] Random Forest saved to '{rf_output_path}'")

    # 2. Train LightGBM Classifier
    if model_type in ("lightgbm", "all"):
        if HAS_LIGHTGBM:
            print("\n[INFO] Training LightGBM Gradient Boosted Decision Forest...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.08,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_model.fit(X_train, y_train)
            y_pred_lgb = lgb_model.predict(X_test)
            acc_lgb = accuracy_score(y_test, y_pred_lgb)
            print(f"[RESULTS - LightGBM] Evaluation Accuracy: {acc_lgb * 100:.2f}%")

            print("\n" + "=" * 60)
            print(" LIGHTGBM FULL CLASSIFICATION REPORT")
            print("=" * 60)
            print(classification_report(y_test, y_pred_lgb, digits=3))

            cm = confusion_matrix(y_test, y_pred_lgb, labels=unique_classes)
            confused_pairs = []
            for i in range(len(unique_classes)):
                for j in range(len(unique_classes)):
                    if i != j and cm[i, j] > 0:
                        confused_pairs.append((unique_classes[i], unique_classes[j], int(cm[i, j])))

            if confused_pairs:
                print(f"[DIAGNOSIS] Confused Alphabet Pairs (total {len(confused_pairs)}):")
                for true_cls, pred_cls, cnt in sorted(confused_pairs, key=lambda x: x[2], reverse=True):
                    print(f"  True [{true_cls}] -> Mispredicted as [{pred_cls}] ({cnt} samples)")
            else:
                print("[DIAGNOSIS] Zero confusion pairs detected! All alphabet classes separated perfectly.")

            os.makedirs(os.path.dirname(lgbm_output_path), exist_ok=True)
            with open(lgbm_output_path, "wb") as f:
                pickle.dump(lgb_model, f)
            print(f"[SUCCESS] Best validated LightGBM Model saved to '{lgbm_output_path}'")
        else:
            print("[WARN] LightGBM not installed, skipping LightGBM training.")

    # 3. Train and Export LSTMR Dynamic Recognizer
    if model_type in ("lstm", "all"):
        print("\n[INFO] Training LSTMR Recurrent Gesture Recognizer on Dynamic Sequences...")
        X_dyn, y_dyn = generate_dynamic_dataset(num_samples_per_class=150, seq_len=20)
        lstm_model = LSTMRecurrentClassifier(
            input_size=63,
            hidden_size=48,
            classes=["J", "Z", "STATIC"],
            window_size=20,
        )
        acc_lstm = lstm_model.fit(X_dyn, y_dyn, epochs=30, lr=0.005, verbose=True)
        print(f"[RESULTS - LSTMR] Evaluation Accuracy: {acc_lstm * 100:.2f}%")
        lstm_model.save(lstm_output_path)
        print(f"[SUCCESS] Trained LSTMR Model saved to '{lstm_output_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASL Alphabet Classifier Models")
    parser.add_argument("--data", type=str, default="data/asl_dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--model", type=str, choices=["all", "lightgbm", "rf", "lstm"], default="all", help="Model type to train")
    parser.add_argument("--rf-output", type=str, default="models/gesture_classifier.pkl", help="Output RF model path")
    parser.add_argument("--lgbm-output", type=str, default="models/lightgbm_classifier.pkl", help="Output LightGBM model path")
    parser.add_argument("--lstm-output", type=str, default="models/lstm_gesture_model.npz", help="Output LSTMR weights path")
    parser.add_argument("--regenerate", action="store_true", help="Force regenerate canonical dataset")
    args = parser.parse_args()

    train_models(
        dataset_path=args.data,
        rf_output_path=args.rf_output,
        lgbm_output_path=args.lgbm_output,
        lstm_output_path=args.lstm_output,
        model_type=args.model,
        regenerate_dataset=args.regenerate,
    )


if __name__ == "__main__":
    main()
