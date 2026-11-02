"""
Visualizer Module for ASL Alphabet Recognition
Renders high-fps, modern HUD graphics, illuminated ASL letter badges, finger state monitors,
and interactive text spelling displays using OpenCV.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.tracker import HandData


# Standard 21-point hand joint connectivity
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base cross-connections
    (5, 9), (9, 13), (13, 17),
]


class ModernVisualizer:
    """Renders semi-transparent HUD elements, illuminated ASL character badges, and word speller."""

    def __init__(
        self,
        theme_color: Tuple[int, int, int] = (0, 230, 118),     # Vibrant Green (BGR)
        secondary_color: Tuple[int, int, int] = (255, 171, 0), # Cyan / Yellow
        draw_landmarks: bool = True,
        show_fps: bool = True,
        show_bounding_box: bool = True,
    ) -> None:
        self.theme_color = theme_color
        self.secondary_color = secondary_color
        self.draw_landmarks = draw_landmarks
        self.show_fps = show_fps
        self.show_bounding_box = show_bounding_box

        # FPS calculation state
        self.prev_time = time.time()
        self.smoothed_fps = 30.0

    def update_fps(self) -> float:
        """Calculates smoothed frame rate."""
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            current_fps = 1.0 / dt
            self.smoothed_fps = 0.9 * self.smoothed_fps + 0.1 * current_fps
        return self.smoothed_fps

    def draw_skeleton(self, frame: np.ndarray, landmarks_px: np.ndarray) -> None:
        """Draws bone connections and joint circles."""
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            pt1 = tuple(landmarks_px[p1_idx])
            pt2 = tuple(landmarks_px[p2_idx])
            cv2.line(frame, pt1, pt2, (220, 220, 220), 2, cv2.LINE_AA)

        tips = {4, 8, 12, 16, 20}
        for idx, pt in enumerate(landmarks_px):
            coord = tuple(pt)
            if idx in tips:
                cv2.circle(frame, coord, 7, self.secondary_color, -1, cv2.LINE_AA)
                cv2.circle(frame, coord, 9, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, coord, 4, self.theme_color, -1, cv2.LINE_AA)

    def draw_corner_rect(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        length: int = 18,
        thickness: int = 2,
    ) -> None:
        """Draws clean corner brackets around the hand."""
        x, y, w, h = bbox
        cv2.line(frame, (x, y), (x + length, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + length), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness, cv2.LINE_AA)

    def render(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        active_letter: str,
        confidence: float,
        spelled_text: str = "",
        debug_info: Optional[Dict[str, any]] = None,
    ) -> np.ndarray:
        """Renders the comprehensive ASL Alphabet HUD."""
        fps = self.update_fps()
        h, w, _ = frame.shape

        # 1. Hand Skeleton & Brackets
        for hand in hands:
            if self.draw_landmarks:
                self.draw_skeleton(frame, hand.landmarks_pixel)

            if self.show_bounding_box:
                self.draw_corner_rect(frame, hand.bbox, self.theme_color, length=20, thickness=2)
                bx, by, _, _ = hand.bbox
                label_text = f"{hand.handedness} ({int(hand.confidence * 100)}%)"
                cv2.putText(
                    frame,
                    label_text,
                    (bx, max(22, by - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )

        # 2. Top Header HUD Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 54), (16, 16, 16), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        fps_color = (0, 255, 128) if fps >= 24 else ((0, 215, 255) if fps >= 15 else (0, 0, 255))
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "ASL Fingerspelling Studio", (170, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        controls_tip = "[Space]: Add Letter  |  [C]: Clear  |  [Q]: Quit"
        cv2.putText(frame, controls_tip, (w - 440, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        # 3. Prominent ASL Alphabet Badge (Top-Left under Header)
        is_active = active_letter not in ("None", "-", "Unknown")
        badge_w, badge_h = 210, 125
        badge_x, badge_y = 20, 70

        badge_overlay = frame.copy()
        cv2.rectangle(badge_overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (20, 20, 20), -1)
        cv2.addWeighted(badge_overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (60, 60, 60), 1)

        cv2.putText(frame, "DETECTED LETTER", (badge_x + 18, badge_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

        # Large Letter Character
        disp_char = active_letter if is_active else "-"
        char_color = self.theme_color if is_active else (100, 100, 100)
        cv2.putText(frame, disp_char, (badge_x + 80, badge_y + 82), cv2.FONT_HERSHEY_DUPLEX, 1.8, char_color, 3, cv2.LINE_AA)

        # Confidence Bar in Badge
        bar_x = badge_x + 18
        bar_y = badge_y + 102
        bar_w = badge_w - 36
        bar_h = 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 45), -1)
        if is_active:
            fill_w = int(bar_w * np.clip(confidence, 0.0, 1.0))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), self.theme_color, -1)
            cv2.putText(frame, f"{int(confidence * 100)}%", (badge_x + badge_w - 48, badge_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # 4. Finger State Status Indicators (Below Badge)
        if debug_info:
            status_y = badge_y + badge_h + 15
            finger_names = [("T", debug_info.get("thumb_out", False) or debug_info.get("thumb_side", False)),
                            ("I", debug_info.get("ext_index", False)),
                            ("M", debug_info.get("ext_middle", False)),
                            ("R", debug_info.get("ext_ring", False)),
                            ("P", debug_info.get("ext_pinky", False))]

            cv2.putText(frame, "Fingers:", (badge_x, status_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
            for i, (fn, state) in enumerate(finger_names):
                f_box_x = badge_x + 65 + i * 28
                col = self.theme_color if state else (70, 70, 70)
                cv2.rectangle(frame, (f_box_x, status_y), (f_box_x + 22, status_y + 20), col, -1 if state else 1)
                text_col = (0, 0, 0) if state else (160, 160, 160)
                cv2.putText(frame, fn, (f_box_x + 6, status_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1, cv2.LINE_AA)

        # 5. Bottom Spelled Word / Sentence Bar
        bar_bottom_h = 65
        bar_bottom_y = h - bar_bottom_h - 15
        bar_bottom_x = 20
        bar_bottom_w = w - 40

        b_overlay = frame.copy()
        cv2.rectangle(b_overlay, (bar_bottom_x, bar_bottom_y), (bar_bottom_x + bar_bottom_w, bar_bottom_y + bar_bottom_h), (18, 18, 18), -1)
        cv2.addWeighted(b_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (bar_bottom_x, bar_bottom_y), (bar_bottom_x + bar_bottom_w, bar_bottom_y + bar_bottom_h), (55, 55, 55), 1)

        cv2.putText(frame, "SPELLED TEXT:", (bar_bottom_x + 18, bar_bottom_y + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 2, cv2.LINE_AA)

        # Spelled output text
        display_spelled = spelled_text if spelled_text else "Show an ASL sign & press SPACE to append..."
        spelled_color = (255, 255, 255) if spelled_text else (110, 110, 110)
        cv2.putText(frame, display_spelled, (bar_bottom_x + 195, bar_bottom_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, spelled_color, 2, cv2.LINE_AA)

        return frame
