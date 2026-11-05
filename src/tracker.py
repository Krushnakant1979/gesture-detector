"""
Hand Tracker Module using MediaPipe Tasks HandLandmarker
Extracts 21 3D hand landmarks, handedness, and bounding boxes in real time.
Supports RunningMode.VIDEO for low-latency temporal landmark tracking and RunningMode.IMAGE for static evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import time
import urllib.request
from typing import List, Optional, Tuple, Union
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
    """Wrapper around MediaPipe HandLandmarker Tasks API with video temporal tracking."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        num_hands: int = 2,
        min_detection_confidence: float = 0.65,
        min_presence_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
        box_padding: int = 20,
        running_mode: str = "video",
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        self.box_padding = box_padding
        self.model_path = model_path
        self.running_mode_str = running_mode.lower()
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size) if self.enable_clahe else None
        self._ensure_model_exists(self.model_path)

        # Initialize MediaPipe Tasks HandLandmarker
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        self._is_video_mode = (self.running_mode_str == "video")
        mode = RunningMode.VIDEO if self._is_video_mode else RunningMode.IMAGE

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = HandLandmarker.create_from_options(options)
        self._last_timestamp_ms: int = 0
        logger.info("MediaPipe HandLandmarker initialized in %s mode.", "VIDEO" if self._is_video_mode else "IMAGE")

    @staticmethod
    def _ensure_model_exists(model_path: str) -> None:
        """Downloads official hand_landmarker.task model file if missing."""
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            logger.info("Downloading MediaPipe hand_landmarker model to %s...", model_path)
            urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, model_path)
            logger.info("Model download complete.")

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Enhances localized contrast and edge definition in low-light and backlit environments
        using Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel (LAB space).
        """
        if not self.enable_clahe or self.clahe is None or frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr

        try:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            cl = self.clahe.apply(l_chan)
            enhanced_lab = cv2.merge((cl, a_chan, b_chan))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.debug("CLAHE enhancement failed, falling back to raw frame: %s", e)
            return frame_bgr

    def process(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> List[HandData]:
        """
        Processes a BGR image frame and extracts hand landmark data.

        Args:
            frame_bgr: Input video frame in OpenCV BGR format.
            timestamp_ms: Monotonic millisecond timestamp (used in VIDEO mode).

        Returns:
            List of HandData objects for all detected hands.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w, _ = frame_bgr.shape

        # Enhance frame contrast under poor/backlit lighting if CLAHE is enabled
        processed_bgr = self.preprocess_frame(frame_bgr) if self.enable_clahe else frame_bgr

        # MediaPipe requires RGB format
        frame_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._is_video_mode:
            if timestamp_ms is None:
                now_ms = int(time.time() * 1000)
                if now_ms <= self._last_timestamp_ms:
                    now_ms = self._last_timestamp_ms + 1
                self._last_timestamp_ms = now_ms
                timestamp_ms = now_ms
            elif timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
                self._last_timestamp_ms = timestamp_ms
            else:
                self._last_timestamp_ms = timestamp_ms

            result = self.detector.detect_for_video(mp_image, timestamp_ms)
        else:
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
            try:
                self.detector.close()
            except Exception:
                pass
