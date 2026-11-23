"""
Visualizer Module for ASL Alphabet Recognition Studio
Renders high-fps, modern HUD graphics, illuminated ASL letter badges, finger state monitors,
gesture hold progress rings, interactive word editing buttons, and real-time TTS status indicators.
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

    FINGER_TIPS = {4, 8, 12, 16, 20}

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

        # UI interactive button coordinates cache
        self.button_boxes: Dict[str, Tuple[int, int, int, int]] = {}

    def update_fps(self) -> float:
        """Calculates smoothed frame rate."""
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            current_fps = 1.0 / dt
            self.smoothed_fps = 0.9 * self.smoothed_fps + 0.1 * current_fps
        return self.smoothed_fps

    @staticmethod
    def _segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
        """Determines if segment (p1, p2) and segment (p3, p4) strictly intersect in 2D."""
        def ccw(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def draw_skeleton(self, frame: np.ndarray, landmarks_px: np.ndarray) -> None:
        """
        Draws bone connections and joint circles with topological sanity checking.
        Suppresses spurious cross-palm connection lines that slice through fingers
        when MediaPipe landmark regression collapses during forward drapes (e.g. M and N).
        """
        digit_bones = [
            (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
        ]
        pts = landmarks_px.astype(np.float32)

        for p1_idx, p2_idx in HAND_CONNECTIONS:
            if p1_idx == 0 or p2_idx == 0:
                seg_a, seg_b = pts[p1_idx], pts[p2_idx]
                crosses_digit = False
                for f1, f2 in digit_bones:
                    if len({p1_idx, p2_idx, f1, f2}) < 4:
                        continue
                    if self._segments_intersect(seg_a, seg_b, pts[f1], pts[f2]):
                        crosses_digit = True
                        break
                if crosses_digit:
                    continue

            pt1 = (int(landmarks_px[p1_idx][0]), int(landmarks_px[p1_idx][1]))
            pt2 = (int(landmarks_px[p2_idx][0]), int(landmarks_px[p2_idx][1]))
            cv2.line(frame, pt1, pt2, (220, 220, 220), 2, cv2.LINE_AA)

        for idx, pt in enumerate(landmarks_px):
            coord = (int(pt[0]), int(pt[1]))
            if idx in self.FINGER_TIPS:
                cv2.circle(frame, coord, 7, self.secondary_color, -1, cv2.LINE_AA)
                cv2.circle(frame, coord, 9, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, coord, 4, self.theme_color, -1, cv2.LINE_AA)

    @staticmethod
    def draw_alpha_rect(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        alpha: float,
    ) -> None:
        """Draws a semi-transparent filled rectangle using zero-copy local ROI slicing."""
        x, y, w, h = bbox
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            return
        sub = frame[y1:y2, x1:x2]
        overlay = np.full(sub.shape, color, dtype=np.uint8)
        cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, sub)

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

    def get_button_at_pos(self, px: int, py: int) -> Optional[str]:
        """Returns button name if mouse coordinates (px, py) fall inside a button."""
        for name, (bx, by, bw, bh) in self.button_boxes.items():
            if bx <= px <= bx + bw and by <= py <= by + bh:
                return name
        return None

    def render(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        active_letter: str,
        confidence: float,
        spelled_text: str = "",
        debug_info: Optional[Dict[str, any]] = None,
        suggestions: Optional[List[str]] = None,
        system_state: str = "WAITING",
        current_word: str = "",
        letter_sequence: Optional[List[str]] = None,
        hold_progress: float = 0.0,
        last_spoken_word: str = "",
        tts_status: str = "IDLE",
        active_control: str = "NONE",
        is_steady: bool = True,
    ) -> np.ndarray:
        """Renders the comprehensive ASL Alphabet and Word Formation HUD."""
        fps = self.update_fps()
        h, w, _ = frame.shape

        # 1. Hand Skeleton & Bounding Brackets
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
        self.draw_alpha_rect(frame, (0, 0, w, 54), (16, 16, 16), 0.75)
        fps_color = (0, 255, 128) if fps >= 24 else ((0, 215, 255) if fps >= 15 else (0, 0, 255))
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "ASL Fingerspelling Studio", (170, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        controls_tip = "[Hold Sign]: Add  |  [👍 Thumbs-Up]: Speak Word  |  [Del/Backspace]  |  [C]: Clear"
        cv2.putText(frame, controls_tip, (max(500, w - 660), 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

        # 3. Prominent ASL Alphabet / Control Badge (Top-Left)
        badge_w, badge_h = 240, 130
        badge_x, badge_y = 20, 70
        self.draw_alpha_rect(frame, (badge_x, badge_y, badge_w, badge_h), (20, 20, 20), 0.82)
        cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (60, 60, 60), 1)

        is_tu = (active_control == "THUMBS_UP")
        is_active = bool(
            not is_tu
            and active_letter not in ("None", "-", "Unknown", "Unknown Hand Gesture", "")
            and confidence >= 0.92
        )

        if is_tu:
            cv2.putText(frame, "CONTROL GESTURE", (badge_x + 18, badge_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "CONFIRM", (badge_x + 35, badge_y + 75), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 128), 2, cv2.LINE_AA)
            cv2.putText(frame, "(Thumbs-Up)", (badge_x + 55, badge_y + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1, cv2.LINE_AA)
        elif is_active:
            cv2.putText(frame, "DETECTED LETTER", (badge_x + 18, badge_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
            cv2.putText(frame, active_letter, (badge_x + 95, badge_y + 82), cv2.FONT_HERSHEY_DUPLEX, 1.8, self.theme_color, 3, cv2.LINE_AA)
            # Accuracy Percentage
            cv2.putText(frame, f"{int(confidence * 100)}%", (badge_x + badge_w - 52, badge_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        elif len(hands) > 0 and not is_steady:
            cv2.putText(frame, "DETECTED LETTER", (badge_x + 18, badge_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
            motion_color = (0, 215, 255)
            cv2.putText(frame, "HOLD STEADY", (badge_x + 18, badge_y + 72), cv2.FONT_HERSHEY_DUPLEX, 0.82, motion_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "Hold sign steady to detect", (badge_x + 18, badge_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
        elif len(hands) > 0:
            cv2.putText(frame, "DETECTED LETTER", (badge_x + 18, badge_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
            warn_color = (0, 165, 255)
            cv2.putText(frame, "Unknown Hand", (badge_x + 18, badge_y + 60), cv2.FONT_HERSHEY_DUPLEX, 0.65, warn_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "Gesture", (badge_x + 18, badge_y + 85), cv2.FONT_HERSHEY_DUPLEX, 0.65, warn_color, 2, cv2.LINE_AA)
            sub_text = f"Accuracy: {int(confidence * 100)}% (< 92%)" if confidence > 0 else "Accuracy < 92%"
            cv2.putText(frame, sub_text, (badge_x + 18, badge_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "DETECTED LETTER", (badge_x + 18, badge_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
            cv2.putText(frame, "-", (badge_x + 105, badge_y + 82), cv2.FONT_HERSHEY_DUPLEX, 1.8, (100, 100, 100), 3, cv2.LINE_AA)
            cv2.putText(frame, "No Hand Detected", (badge_x + 45, badge_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 130, 130), 1, cv2.LINE_AA)

        # Confidence Bar in Badge
        bar_x = badge_x + 18
        bar_y = badge_y + 112
        bar_w = badge_w - 36
        bar_h = 7
        if is_active or is_tu:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 45), -1)
            fill_w = int(bar_w * np.clip(confidence, 0.0, 1.0))
            col = (0, 255, 128) if is_tu else self.theme_color
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), col, -1)

        # 4. Finger State Status Indicators (Below Badge)
        if debug_info:
            status_y = badge_y + badge_h + 12
            finger_names = [
                ("T", debug_info.get("thumb_extended_out", False) or debug_info.get("is_thumb_pointing_up", False)),
                ("I", debug_info.get("ext_index", False)),
                ("M", debug_info.get("ext_middle", False)),
                ("R", debug_info.get("ext_ring", False)),
                ("P", debug_info.get("ext_pinky", False)),
            ]
            cv2.putText(frame, "Fingers:", (badge_x, status_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
            for i, (fn, state) in enumerate(finger_names):
                f_box_x = badge_x + 65 + i * 28
                col = self.theme_color if state else (70, 70, 70)
                cv2.rectangle(frame, (f_box_x, status_y), (f_box_x + 22, status_y + 20), col, -1 if state else 1)
                text_col = (0, 0, 0) if state else (160, 160, 160)
                cv2.putText(frame, fn, (f_box_x + 6, status_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1, cv2.LINE_AA)

        # 5. System State & Gesture Hold Progress Banner (Top-Right under Header)
        state_w, state_h = 300, 85
        state_x = w - state_w - 20
        state_y = 70
        self.draw_alpha_rect(frame, (state_x, state_y, state_w, state_h), (22, 22, 22), 0.85)
        cv2.rectangle(frame, (state_x, state_y), (state_x + state_w, state_y + state_h), (60, 60, 60), 1)

        cv2.putText(frame, "SYSTEM STATUS", (state_x + 16, state_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)

        state_color_map = {
            "WAITING": ((130, 130, 130), "WAITING FOR GESTURE"),
            "COLLECTING": ((255, 200, 0), f"HOLDING SIGN '{active_letter}'"),
            "CONFIRMING": ((0, 230, 118), "CONFIRMING (THUMBS-UP)"),
            "COOLDOWN": ((0, 165, 255), "EVENT COMMITTED"),
            "SPEAKING": ((220, 100, 255), f"SPEAKING: '{last_spoken_word}'"),
        }
        st_color, st_label = state_color_map.get(system_state, ((150, 150, 150), system_state))

        # Status Pill indicator
        cv2.circle(frame, (state_x + 24, state_y + 44), 6, st_color, -1, cv2.LINE_AA)
        cv2.putText(frame, st_label, (state_x + 38, state_y + 48), cv2.FONT_HERSHEY_DUPLEX, 0.48, st_color, 1, cv2.LINE_AA)

        # Hold Progress Bar
        prog_bar_x = state_x + 16
        prog_bar_y = state_y + 63
        prog_bar_w = state_w - 32
        prog_bar_h = 9
        cv2.rectangle(frame, (prog_bar_x, prog_bar_y), (prog_bar_x + prog_bar_w, prog_bar_y + prog_bar_h), (40, 40, 40), -1)

        if hold_progress > 0.0:
            fill_prog = int(prog_bar_w * np.clip(hold_progress, 0.0, 1.0))
            prog_col = (0, 230, 118) if system_state == "CONFIRMING" else (255, 190, 0)
            cv2.rectangle(frame, (prog_bar_x, prog_bar_y), (prog_bar_x + fill_prog, prog_bar_y + prog_bar_h), prog_col, -1)
            cv2.putText(frame, f"{int(hold_progress * 100)}%", (state_x + state_w - 45, state_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, prog_col, 1, cv2.LINE_AA)

        # 6. Suggestions Strip (if enabled)
        bar_bottom_h = 75
        bar_bottom_y = h - bar_bottom_h - 15
        bar_bottom_x = 20
        bar_bottom_w = w - 40

        if suggestions:
            sug_y = bar_bottom_y - 28
            self.draw_alpha_rect(frame, (bar_bottom_x, sug_y, bar_bottom_w, 24), (24, 24, 24), 0.80)
            sug_text = "Suggestions [TAB to complete]: " + "   |   ".join(suggestions)
            cv2.putText(frame, sug_text, (bar_bottom_x + 12, sug_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

        # 7. Interactive Control Action Buttons (Above bottom bar)
        btn_y = bar_bottom_y - 36 if not suggestions else sug_y - 36
        btn_h = 28
        btn_spacing = 10

        buttons = [
            ("DELETE", "[ DEL ]", (0, 140, 255), 90),
            ("CLEAR", "[ CLEAR ]", (50, 50, 220), 95),
            ("REPEAT", "[ REPEAT ]", (200, 100, 255), 105),
            ("CONFIRM", "[ CONFIRM 👍 ]", (0, 200, 100), 140),
        ]

        total_btn_w = sum(b[3] for b in buttons) + btn_spacing * (len(buttons) - 1)
        start_btn_x = w - total_btn_w - 25

        self.button_boxes.clear()
        curr_x = start_btn_x
        for b_id, b_text, b_col, b_w in buttons:
            self.button_boxes[b_id] = (curr_x, btn_y, b_w, btn_h)
            self.draw_alpha_rect(frame, (curr_x, btn_y, b_w, btn_h), (25, 25, 25), 0.85)
            cv2.rectangle(frame, (curr_x, btn_y), (curr_x + b_w, btn_y + btn_h), b_col, 1)
            cv2.putText(frame, b_text, (curr_x + 10, btn_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, b_col, 1, cv2.LINE_AA)
            curr_x += b_w + btn_spacing

        # Last Spoken Status Tag (left of buttons)
        if last_spoken_word:
            tts_tag = f"SPOKEN: \"{last_spoken_word}\" ({tts_status})"
            self.draw_alpha_rect(frame, (bar_bottom_x, btn_y, 350, btn_h), (20, 20, 20), 0.80)
            cv2.putText(frame, tts_tag, (bar_bottom_x + 10, btn_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 255, 100), 1, cv2.LINE_AA)

        # 8. Bottom Word Sequence & Formed Word Display Bar
        self.draw_alpha_rect(frame, (bar_bottom_x, bar_bottom_y, bar_bottom_w, bar_bottom_h), (18, 18, 18), 0.88)
        cv2.rectangle(frame, (bar_bottom_x, bar_bottom_y), (bar_bottom_x + bar_bottom_w, bar_bottom_y + bar_bottom_h), (55, 55, 55), 1)

        display_word = current_word if current_word else ""

        if display_word:
            # Formed word actively being constructed
            cv2.putText(frame, "CURRENT WORD:", (bar_bottom_x + 18, bar_bottom_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 170, 170), 1, cv2.LINE_AA)
            cv2.putText(frame, display_word, (bar_bottom_x + 155, bar_bottom_y + 36), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 200), 2, cv2.LINE_AA)

            # Sequence details (e.g. K -> A -> J -> A -> L)
            if letter_sequence and len(letter_sequence) > 1:
                seq_text = "Sequence: " + " -> ".join(letter_sequence)
                cv2.putText(frame, seq_text, (bar_bottom_x + 18, bar_bottom_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 220, 255), 1, cv2.LINE_AA)
            elif spelled_text:
                cv2.putText(frame, f"Sentence History: {spelled_text}", (bar_bottom_x + 18, bar_bottom_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
        elif spelled_text:
            # Word buffer is empty, show previously confirmed words
            cv2.putText(frame, "LAST CONFIRMED:", (bar_bottom_x + 18, bar_bottom_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 170, 170), 1, cv2.LINE_AA)
            cv2.putText(frame, spelled_text, (bar_bottom_x + 175, bar_bottom_y + 36), cv2.FONT_HERSHEY_DUPLEX, 1.0, (100, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Hold an ASL sign to start new word  |  Click [CLEAR] to reset", (bar_bottom_x + 18, bar_bottom_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "CURRENT WORD:", (bar_bottom_x + 18, bar_bottom_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140, 140, 140), 1, cv2.LINE_AA)
            cv2.putText(frame, "Hold an ASL sign to add letters  |  Show 👍 to confirm & speak", (bar_bottom_x + 155, bar_bottom_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (110, 110, 110), 1, cv2.LINE_AA)
            cv2.putText(frame, "Supports repeated letters (e.g. HELLO, BOOK, APPLE) via neutral transitions", (bar_bottom_x + 18, bar_bottom_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 130, 130), 1, cv2.LINE_AA)

        return frame
