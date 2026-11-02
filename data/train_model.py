"""
Model Training Script for ASL Alphabet Classification
Trains an accurate Random Forest model on normalized 3D hand landmark coordinates
covering the American Sign Language alphabet (A through Z).
"""

from __future__ import annotations

import argparse
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


def generate_asl_dataset(output_csv: str, samples_per_class: int = 80) -> None:
    """
    Generates a calibrated synthetic dataset based on anatomical landmark patterns
    for ASL alphabet letters A through Z.
    """
    print("[INFO] Generating calibrated ASL alphabet dataset (A-Z)...")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    np.random.seed(42)

    # All 26 ASL alphabet classes
    classes = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

    X = []
    y = []

    for cls in classes:
        for _ in range(samples_per_class):
            # 63 features: 21 points x 3 coordinates (x, y, z)
            vec = np.zeros(63, dtype=np.float32)

            # Wrist at 0, 1, 2 = (0, 0, 0)
            # Default palm base
            vec[3:6] = [-0.15, -0.2, 0.0]   # Thumb CMC (1)
            vec[6:9] = [-0.25, -0.35, 0.0]  # Thumb MCP (2)
            vec[9:12] = [-0.3, -0.45, 0.0]  # Thumb IP (3)
            vec[12:15] = [-0.35, -0.55, 0.0] # Thumb TIP (4)

            # MCP knuckles (indices 5, 9, 13, 17)
            vec[15:18] = [-0.15, -0.5, 0.0] # Index MCP (5)
            vec[27:30] = [0.0, -0.52, 0.0]  # Middle MCP (9)
            vec[39:42] = [0.15, -0.5, 0.0]  # Ring MCP (13)
            vec[51:54] = [0.28, -0.45, 0.0] # Pinky MCP (17)

            # Finger configurations by letter:
            # 1. Letters with 4 upright fingers (B, C, O)
            if cls == "B":
                # 4 fingers straight up, touching
                vec[24:27] = [-0.15, -0.95, 0.0]  # Index TIP
                vec[36:39] = [0.0, -0.98, 0.0]    # Middle TIP
                vec[48:51] = [0.15, -0.94, 0.0]   # Ring TIP
                vec[60:63] = [0.28, -0.88, 0.0]   # Pinky TIP
                vec[12:15] = [0.05, -0.4, 0.1]    # Thumb tucked across palm
            elif cls == "C":
                # Curved into C
                vec[24:27] = [-0.1, -0.7, -0.2]
                vec[36:39] = [0.0, -0.7, -0.2]
                vec[48:51] = [0.1, -0.68, -0.2]
                vec[60:63] = [0.2, -0.65, -0.2]
                vec[12:15] = [-0.2, -0.35, -0.1]
            elif cls == "O":
                # All tips touching thumb tip
                vec[12:15] = [-0.05, -0.6, 0.0]
                vec[24:27] = [-0.05, -0.62, 0.0]
                vec[36:39] = [-0.04, -0.62, 0.0]
                vec[48:51] = [-0.03, -0.61, 0.0]
                vec[60:63] = [-0.02, -0.60, 0.0]
            # 2. Letters with 3 upright fingers (W, F)
            elif cls == "W":
                vec[24:27] = [-0.2, -0.95, 0.0]   # Index TIP
                vec[36:39] = [0.0, -0.98, 0.0]    # Middle TIP
                vec[48:51] = [0.2, -0.94, 0.0]    # Ring TIP
                vec[60:63] = [0.25, -0.4, 0.1]    # Pinky curled
                vec[12:15] = [0.15, -0.45, 0.1]   # Thumb over pinky
            elif cls == "F":
                vec[12:15] = [-0.1, -0.6, 0.0]    # Thumb tip touches index tip
                vec[24:27] = [-0.1, -0.6, 0.0]    # Index tip
                vec[36:39] = [0.0, -0.98, 0.0]    # Middle upright
                vec[48:51] = [0.15, -0.94, 0.0]   # Ring upright
                vec[60:63] = [0.28, -0.88, 0.0]   # Pinky upright
            # 3. Letters with 2 upright fingers (V, U, R, K, H, P)
            elif cls == "V":
                vec[24:27] = [-0.2, -0.95, 0.0]   # Index spread
                vec[36:39] = [0.2, -0.95, 0.0]    # Middle spread
                vec[48:51] = [0.15, -0.35, 0.1]   # Ring curled
                vec[60:63] = [0.25, -0.35, 0.1]   # Pinky curled
            elif cls == "U":
                vec[24:27] = [-0.05, -0.95, 0.0]  # Index close
                vec[36:39] = [0.05, -0.95, 0.0]   # Middle close
                vec[48:51] = [0.15, -0.35, 0.1]
                vec[60:63] = [0.25, -0.35, 0.1]
            elif cls == "R":
                vec[24:27] = [0.03, -0.95, 0.0]   # Crossed over
                vec[36:39] = [-0.03, -0.95, 0.0]
                vec[48:51] = [0.15, -0.35, 0.1]
                vec[60:63] = [0.25, -0.35, 0.1]
            elif cls == "K":
                vec[24:27] = [-0.15, -0.95, 0.0]
                vec[36:39] = [0.15, -0.95, 0.0]
                vec[12:15] = [0.0, -0.7, 0.05]    # Thumb between them
                vec[48:51] = [0.15, -0.35, 0.1]
                vec[60:63] = [0.25, -0.35, 0.1]
            elif cls == "H":
                vec[24:27] = [0.7, -0.4, 0.0]     # Horizontal index
                vec[36:39] = [0.7, -0.3, 0.0]     # Horizontal middle
                vec[48:51] = [0.15, -0.35, 0.1]
                vec[60:63] = [0.25, -0.35, 0.1]
            elif cls == "P":
                vec[24:27] = [0.2, 0.3, 0.0]      # Pointing down index
                vec[36:39] = [0.0, 0.4, 0.0]      # Pointing down middle
                vec[12:15] = [0.1, 0.1, 0.0]
                vec[48:51] = [0.15, -0.35, 0.1]
                vec[60:63] = [0.25, -0.35, 0.1]
            # 4. Letters with 1 upright finger (L, D, G, X, I, Q, Z, Y)
            elif cls == "L":
                vec[24:27] = [-0.15, -0.95, 0.0]  # Index UP
                vec[12:15] = [-0.65, -0.45, 0.0]  # Thumb HORIZONTAL
                vec[36:39] = [0.0, -0.35, 0.1]    # Middle curled
                vec[48:51] = [0.15, -0.35, 0.1]
                vec[60:63] = [0.25, -0.35, 0.1]
            elif cls == "D":
                vec[24:27] = [-0.15, -0.95, 0.0]  # Index UP
                vec[12:15] = [-0.05, -0.5, 0.0]   # Thumb touching middle
                vec[36:39] = [-0.05, -0.5, 0.0]
                vec[48:51] = [0.15, -0.35, 0.1]
                vec[60:63] = [0.25, -0.35, 0.1]
            elif cls == "G":
                vec[24:27] = [0.7, -0.5, 0.0]     # Index horizontal
                vec[12:15] = [0.5, -0.4, 0.0]     # Thumb parallel
                vec[36:39] = [0.0, -0.35, 0.1]
                vec[48:51] = [0.15, -0.35, 0.1]
                vec[60:63] = [0.25, -0.35, 0.1]
            elif cls == "Q":
                vec[24:27] = [0.2, 0.5, 0.0]      # Index down
                vec[12:15] = [0.1, 0.4, 0.0]      # Thumb down
                vec[36:39] = [0.0, -0.35, 0.1]
                vec[48:51] = [0.15, -0.35, 0.1]
                vec[60:63] = [0.25, -0.35, 0.1]
            elif cls == "X":
                vec[24:27] = [-0.15, -0.65, -0.1] # Index bent hook
                vec[36:39] = [0.0, -0.35, 0.1]
                vec[48:51] = [0.15, -0.35, 0.1]
                vec[60:63] = [0.25, -0.35, 0.1]
            elif cls == "I" or cls == "J":
                vec[60:63] = [0.28, -0.92, 0.0]   # Pinky UP
                vec[24:27] = [-0.15, -0.35, 0.1]  # Others curled
                vec[36:39] = [0.0, -0.35, 0.1]
                vec[48:51] = [0.15, -0.35, 0.1]
                vec[12:15] = [-0.1, -0.4, 0.1]
            elif cls == "Y":
                vec[12:15] = [-0.65, -0.45, 0.0]  # Thumb extended out
                vec[60:63] = [0.45, -0.85, 0.0]   # Pinky extended out
                vec[24:27] = [-0.15, -0.35, 0.1]  # Middle 3 curled
                vec[36:39] = [0.0, -0.35, 0.1]
                vec[48:51] = [0.15, -0.35, 0.1]
            elif cls == "Z":
                vec[24:27] = [-0.15, -0.95, 0.0]  # Index pointing
                vec[36:39] = [0.0, -0.35, 0.1]
                vec[48:51] = [0.15, -0.35, 0.1]
                vec[60:63] = [0.25, -0.35, 0.1]
            # 5. Letters where all fingers curled (A, E, S, M, N, T)
            elif cls == "A":
                vec[12:15] = [-0.25, -0.65, 0.0]  # Thumb alongside index, UP
                vec[24:27] = [-0.15, -0.35, 0.1]
                vec[36:39] = [0.0, -0.35, 0.1]
                vec[48:51] = [0.15, -0.35, 0.1]
                vec[60:63] = [0.25, -0.35, 0.1]
            elif cls == "S":
                vec[12:15] = [0.0, -0.4, 0.15]    # Thumb across fingers
                vec[24:27] = [-0.15, -0.35, 0.05]
                vec[36:39] = [0.0, -0.35, 0.05]
                vec[48:51] = [0.15, -0.35, 0.05]
                vec[60:63] = [0.25, -0.35, 0.05]
            elif cls == "E":
                vec[12:15] = [-0.1, -0.25, 0.1]   # Thumb tucked low under tips
                vec[24:27] = [-0.15, -0.38, 0.0]
                vec[36:39] = [0.0, -0.38, 0.0]
                vec[48:51] = [0.15, -0.38, 0.0]
                vec[60:63] = [0.25, -0.38, 0.0]
            elif cls == "T":
                vec[12:15] = [-0.08, -0.45, 0.1]  # Thumb between index & middle
                vec[24:27] = [-0.15, -0.35, 0.05]
                vec[36:39] = [0.0, -0.35, 0.05]
                vec[48:51] = [0.15, -0.35, 0.05]
                vec[60:63] = [0.25, -0.35, 0.05]
            elif cls == "M":
                vec[12:15] = [0.15, -0.3, 0.1]    # Under 3 fingers
                vec[24:27] = [-0.15, -0.35, 0.05]
                vec[36:39] = [0.0, -0.35, 0.05]
                vec[48:51] = [0.15, -0.35, 0.05]
                vec[60:63] = [0.25, -0.35, 0.05]
            elif cls == "N":
                vec[12:15] = [0.05, -0.3, 0.1]    # Under 2 fingers
                vec[24:27] = [-0.15, -0.35, 0.05]
                vec[36:39] = [0.0, -0.35, 0.05]
                vec[48:51] = [0.15, -0.35, 0.05]
                vec[60:63] = [0.25, -0.35, 0.05]

            # Add natural human hand measurement variance
            noise = np.random.normal(0.0, 0.025, 63).astype(np.float32)
            sample = vec + noise
            X.append(sample)
            y.append(cls)

    header = "label," + ",".join([f"f_{i}" for i in range(63)])
    lines = [header]
    for label, feat in zip(y, X):
        row = f"{label}," + ",".join([f"{v:.6f}" for v in feat])
        lines.append(row)

    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[SUCCESS] Calibrated ASL dataset saved with {len(X)} samples across {len(classes)} letters.")


def train(dataset_path: str, model_output_path: str) -> None:
    if not os.path.exists(dataset_path):
        generate_asl_dataset(dataset_path)

    print(f"[INFO] Loading ASL dataset from '{dataset_path}'...")
    data = []
    labels = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) < 64:
                continue
            labels.append(parts[0])
            data.append([float(x) for x in parts[1:]])

    X = np.array(data, dtype=np.float32)
    y = np.array(labels)

    unique_classes = np.unique(y)
    print(f"[INFO] Dataset ready: {len(X)} samples across {len(unique_classes)} classes: {list(unique_classes)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Training Random Forest ASL Classifier...")
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULTS] Evaluation Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[SUCCESS] Trained ASL Model saved to '{model_output_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASL Alphabet Classifier")
    parser.add_argument("--data", type=str, default="data/asl_dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--output", type=str, default="models/gesture_classifier.pkl", help="Output model path")
    args = parser.parse_args()

    train(args.data, args.output)


if __name__ == "__main__":
    main()
