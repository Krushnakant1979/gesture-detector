"""
Hand Tracker Module using MediaPipe Tasks HandLandmarker
Extracts 21 3D hand landmarks, handedness, and bounding boxes in real time.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import urllib.request
from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task"))


@dataclass
class HandData:
    """Holds structured information about a detected hand."""
    handedness: str  # "Left" or "Right"
    confidence: float
    landmarks_normalized: np.ndarray  # Shape (21, 3) in [0.0, 1.0] relative to frame
    landmarks_pixel: np.ndarray       # Shape (21, 2) in integer image pixels
    bbox: Tuple[int, int, int, int]    # (x_min, y_min, width, height) in pixels


class HandTracker:
    """Wrapper around MediaPipe HandLandmarker Tasks API with legacy fallback support."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        box_padding: int = 20,
    ) -> None:
        self.box_padding = box_padding
        self.model_path = model_path
        self._ensure_model_exists(self.model_path)

        # Initialize MediaPipe Tasks HandLandmarker
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = HandLandmarker.create_from_options(options)
        logger.info("MediaPipe HandLandmarker Tasks API initialized successfully.")

    @staticmethod
    def _ensure_model_exists(model_path: str) -> None:
        """Downloads official hand_landmarker.task model file if missing."""
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            logger.info("Downloading MediaPipe hand_landmarker model to %s...", model_path)
            urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, model_path)
            logger.info("Model download complete.")

    def process(self, frame_bgr: np.ndarray) -> List[HandData]:
        """
        Processes a BGR image frame and extracts hand landmark data.

        Args:
            frame_bgr: Input video frame in OpenCV BGR format.

        Returns:
            List of HandData objects for all detected hands.
        """
        h, w, _ = frame_bgr.shape

        # MediaPipe requires RGB format
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.detector.detect(mp_image)

        detected_hands: List[HandData] = []
        if not result or not result.hand_landmarks:
            return detected_hands

        for idx, landmarks in enumerate(result.hand_landmarks):
            handedness_label = "Unknown"
            confidence = 0.0

            if result.handedness and idx < len(result.handedness):
                categories = result.handedness[idx]
                if categories:
                    handedness_label = categories[0].category_name or categories[0].display_name or "Unknown"
                    confidence = float(categories[0].score)

            # Extract 21 points
            coords_norm = np.zeros((21, 3), dtype=np.float32)
            coords_pixel = np.zeros((21, 2), dtype=np.int32)

            for i, lm in enumerate(landmarks):
                coords_norm[i] = [lm.x, lm.y, lm.z]
                coords_pixel[i] = [int(lm.x * w), int(lm.y * h)]

            # Compute bounding box with padding
            x_min = max(0, int(np.min(coords_pixel[:, 0])) - self.box_padding)
            y_min = max(0, int(np.min(coords_pixel[:, 1])) - self.box_padding)
            x_max = min(w, int(np.max(coords_pixel[:, 0])) + self.box_padding)
            y_max = min(h, int(np.max(coords_pixel[:, 1])) + self.box_padding)
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            detected_hands.append(
                HandData(
                    handedness=handedness_label,
                    confidence=confidence,
                    landmarks_normalized=coords_norm,
                    landmarks_pixel=coords_pixel,
                    bbox=bbox,
                )
            )

        return detected_hands

    def close(self) -> None:
        """Closes the MediaPipe detector instance."""
        if hasattr(self, "detector") and self.detector:
            self.detector.close()
