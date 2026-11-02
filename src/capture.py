"""
Threaded Camera Capture Module
Provides non-blocking, zero-latency frame acquisition from webcams.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ThreadedCamera:
    """Runs video capture in a dedicated thread to eliminate buffer latency."""

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.device_id = device_id
        self.target_width = width
        self.target_height = height
        self.target_fps = fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.is_running: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> ThreadedCamera:
        """Initializes the video capture device and starts the worker thread."""
        logger.info("Initializing camera (Device ID: %d)...", self.device_id)

        # On Windows, try cv2.CAP_DSHOW for faster startup, fallback to default
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        if not self.cap or not self.cap.isOpened():
            logger.warning("DSHOW backend failed; falling back to default backend.")
            self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device {self.device_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Prime the first frame
        success, first_frame = self.cap.read()
        if not success or first_frame is None:
            self.cap.release()
            raise RuntimeError(f"Camera opened, but failed to read initial frame from device {self.device_id}")

        with self.lock:
            self.frame = first_frame
            self.is_running = True

        self.thread = threading.Thread(target=self._capture_loop, name="CameraWorkerThread", daemon=True)
        self.thread.start()
        logger.info("Camera stream started successfully.")
        return self

    def _capture_loop(self) -> None:
        """Continually grabs the latest frame and stores it."""
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                break

            success, frame = self.cap.read()
            if success and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the most recent frame thread-safely."""
        with self.lock:
            if not self.is_running or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self) -> None:
        """Stops the worker thread and releases video capture hardware."""
        logger.info("Stopping camera stream...")
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.cap and self.cap.isOpened():
            self.cap.release()

        self.cap = None
        self.frame = None
        logger.info("Camera stream released.")

    def __enter__(self) -> ThreadedCamera:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

