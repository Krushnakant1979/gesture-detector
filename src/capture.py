"""
Threaded Camera Capture Module with Robust Reconnect
Provides non-blocking, zero-latency frame acquisition from webcams with automatic reconnect.
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
    """Runs video capture in a dedicated thread with auto-reconnection watchdog."""

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        reconnect_timeout_sec: float = 2.0,
    ) -> None:
        self.device_id = device_id
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.reconnect_timeout_sec = reconnect_timeout_sec

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.is_running: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        self._last_successful_frame_time = time.time()

    def _open_device(self, dev_id: int) -> Optional[cv2.VideoCapture]:
        """Attempts to open camera device with DSHOW backend first, then standard."""
        logger.info("Opening camera device %d...", dev_id)
        cap = cv2.VideoCapture(dev_id, cv2.CAP_DSHOW)
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(dev_id)

        if cap and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            return cap
        return None

    def start(self) -> ThreadedCamera:
        """Initializes the video capture device and starts the worker thread."""
        self.cap = self._open_device(self.device_id)

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device {self.device_id}")

        # Prime the first frame
        success, first_frame = self.cap.read()
        if not success or first_frame is None:
            self.cap.release()
            raise RuntimeError(f"Camera opened, but failed to read initial frame from device {self.device_id}")

        with self.lock:
            self.frame = first_frame
            self.is_running = True
            self._last_successful_frame_time = time.time()

        self.thread = threading.Thread(target=self._capture_loop, name="CameraWorkerThread", daemon=True)
        self.thread.start()
        logger.info("Camera stream started successfully on device %d.", self.device_id)
        return self

    def _attempt_reconnect(self) -> bool:
        """Attempts to reconnect to the camera if disconnected."""
        logger.warning("Camera watchdog triggered: attempting reconnection...")
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        time.sleep(0.5)

        # Try designated device ID first, fallback to device 0
        candidate_ids = [self.device_id]
        if self.device_id != 0:
            candidate_ids.append(0)

        for dev_id in candidate_ids:
            new_cap = self._open_device(dev_id)
            if new_cap and new_cap.isOpened():
                success, frame = new_cap.read()
                if success and frame is not None:
                    self.cap = new_cap
                    with self.lock:
                        self.frame = frame
                        self._last_successful_frame_time = time.time()
                    logger.info("Successfully reconnected to camera device %d.", dev_id)
                    return True
                else:
                    new_cap.release()

        logger.error("Reconnection attempt failed. Will retry...")
        return False

    def _capture_loop(self) -> None:
        """Continually grabs the latest frame and stores it with watchdog reconnection."""
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                self._attempt_reconnect()
                time.sleep(0.5)
                continue

            success, frame = self.cap.read()
            now = time.time()

            if success and frame is not None:
                with self.lock:
                    self.frame = frame
                    self._last_successful_frame_time = now
            else:
                # Check if timeout exceeded
                if now - self._last_successful_frame_time > self.reconnect_timeout_sec:
                    self._attempt_reconnect()
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
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = None
        self.frame = None
        logger.info("Camera stream released.")

    def __enter__(self) -> ThreadedCamera:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
