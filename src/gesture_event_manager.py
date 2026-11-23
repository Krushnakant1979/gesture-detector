"""
Gesture Event Manager for ASL Fingerspelling Studio.
Manages temporal stability, hold duration, cooldowns, neutral-pose detection,
and discrete event dispatching for alphabet sequence collection and control gestures.
"""

from __future__ import annotations

from enum import Enum
import logging
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.gesture_commands import ControlGestureRecognizer, ControlGestureType

logger = logging.getLogger(__name__)


class SystemState(str, Enum):
    WAITING = "WAITING"
    COLLECTING = "COLLECTING"
    CONFIRMING = "CONFIRMING"
    COOLDOWN = "COOLDOWN"
    SPEAKING = "SPEAKING"


class EventType(str, Enum):
    NONE = "NONE"
    LETTER = "LETTER"
    CONFIRM = "CONFIRM"
    DELETE = "DELETE"
    CLEAR = "CLEAR"


class GestureEvent:
    """Represents a discrete gesture event produced by the event manager."""

    def __init__(
        self,
        event_type: EventType,
        value: str = "",
        confidence: float = 0.0,
        progress: float = 0.0,
        state: SystemState = SystemState.WAITING,
    ) -> None:
        self.event_type = event_type
        self.value = value
        self.confidence = confidence
        self.progress = progress
        self.state = state

    def __repr__(self) -> str:
        return f"GestureEvent({self.event_type.value}, value='{self.value}', conf={self.confidence:.2f}, prog={self.progress:.2f}, state={self.state.value})"


class GestureEventManager:
    """
    Stateful Gesture Event Processor.
    Translates continuous frame-by-frame predictions into clean, discrete spelling events:
    - Verifies prediction stability over a configurable hold duration.
    - Enforces neutral-pose / hand-transition requirements between consecutive letters (solves 'HELLO').
    - Dispatches Thumbs-Up word confirmation events.
    - Prevents duplicate letters from continuous frames.
    """

    def __init__(
        self,
        hold_frames: int = 18,              # Frames gesture must be steadily held (~0.60s at 30 FPS)
        confirm_hold_frames: int = 14,      # Frames thumbs-up must be held for confirmation (~0.47s)
        cooldown_frames: int = 12,          # Cooldown frames after committing an event (~0.40s)
        neutral_frames_required: int = 8,   # Frames of neutral/open pose required to reset repeat lockout (~0.27s)
        min_confidence: float = 0.92,       # Calibrated confidence gate
    ) -> None:
        self.hold_frames = max(2, hold_frames)
        self.confirm_hold_frames = max(2, confirm_hold_frames)
        self.cooldown_frames = max(1, cooldown_frames)
        self.neutral_frames_required = max(1, neutral_frames_required)
        self.min_confidence = float(min_confidence)

        self.control_recognizer = ControlGestureRecognizer(min_confidence=self.min_confidence)

        # State machine variables
        self.current_state: SystemState = SystemState.WAITING
        self.candidate_letter: str = ""
        self.candidate_hold_count: int = 0
        self.last_committed_letter: str = ""
        self.cooldown_counter: int = 0
        self.neutral_counter: int = 0
        self.is_neutral_active: bool = True  # Starts true so first gesture is immediately accepted

        # Thumbs-up confirmation state
        self.control_candidate: ControlGestureType = ControlGestureType.NONE
        self.control_hold_count: int = 0
        self.has_confirmed_current_word: bool = False

    def process_frame(
        self,
        raw_letter: str,
        confidence: float,
        landmarks: Optional[np.ndarray],
        debug_info: Optional[Dict[str, Any]],
        handedness: str = "Right",
        is_tts_speaking: bool = False,
        is_steady: bool = True,
    ) -> GestureEvent:
        """
        Processes a single webcam frame and returns a discrete GestureEvent.
        """
        # If TTS is actively speaking, system state is SPEAKING
        if is_tts_speaking:
            self.current_state = SystemState.SPEAKING
            return GestureEvent(
                EventType.NONE,
                state=SystemState.SPEAKING,
                progress=0.0,
            )

        # Handle cooldown countdown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.current_state = SystemState.COOLDOWN
            return GestureEvent(
                EventType.NONE,
                state=SystemState.COOLDOWN,
                progress=0.0,
            )

        # If the hand is in motion (moving between gestures or shaking),
        # do not accumulate hold frames or commit gestures until steady
        if not is_steady:
            self.candidate_letter = ""
            self.candidate_hold_count = 0
            self.control_candidate = ControlGestureType.NONE
            self.control_hold_count = 0
            self.neutral_counter = 0  # Moving hand cannot count toward neutral reset
            self.current_state = SystemState.WAITING
            return GestureEvent(
                EventType.NONE,
                state=SystemState.WAITING,
                progress=0.0,
            )

        # 1. Check for Control Gestures (THUMBS_UP / THUMBS_DOWN) FIRST
        # Control gestures are non-alphabet gestures and take precedence over letter recognition.
        if landmarks is not None and debug_info is not None:
            ctrl_type, ctrl_conf = self.control_recognizer.recognize(
                landmarks, debug_info, handedness=handedness
            )

            if ctrl_type == ControlGestureType.THUMBS_UP:
                self.neutral_counter = 0
                self.candidate_letter = ""
                self.candidate_hold_count = 0

                # Guard against repeated continuous confirmations
                if self.has_confirmed_current_word:
                    self.current_state = SystemState.WAITING
                    return GestureEvent(
                        EventType.NONE,
                        state=SystemState.WAITING,
                        progress=1.0,
                    )

                if self.control_candidate == ControlGestureType.THUMBS_UP:
                    self.control_hold_count += 1
                else:
                    self.control_candidate = ControlGestureType.THUMBS_UP
                    self.control_hold_count = 1

                progress = min(1.0, self.control_hold_count / self.confirm_hold_frames)
                self.current_state = SystemState.CONFIRMING

                if self.control_hold_count >= self.confirm_hold_frames:
                    self.has_confirmed_current_word = True
                    self.control_hold_count = 0
                    self.cooldown_counter = self.cooldown_frames
                    self.is_neutral_active = False
                    return GestureEvent(
                        EventType.CONFIRM,
                        value="CONFIRM",
                        confidence=ctrl_conf,
                        progress=1.0,
                        state=SystemState.CONFIRMING,
                    )

                return GestureEvent(
                    EventType.NONE,
                    value="THUMBS_UP",
                    confidence=ctrl_conf,
                    progress=progress,
                    state=SystemState.CONFIRMING,
                )

            elif ctrl_type == ControlGestureType.THUMBS_DOWN:
                self.neutral_counter = 0
                self.candidate_letter = ""
                self.candidate_hold_count = 0
                if self.control_candidate == ControlGestureType.THUMBS_DOWN:
                    self.control_hold_count += 1
                else:
                    self.control_candidate = ControlGestureType.THUMBS_DOWN
                    self.control_hold_count = 1

                progress = min(1.0, self.control_hold_count / self.confirm_hold_frames)
                if self.control_hold_count >= self.confirm_hold_frames:
                    self.control_hold_count = 0
                    self.cooldown_counter = self.cooldown_frames
                    self.is_neutral_active = False
                    return GestureEvent(
                        EventType.DELETE,
                        value="DELETE",
                        confidence=ctrl_conf,
                        progress=1.0,
                        state=SystemState.WAITING,
                    )
                return GestureEvent(
                    EventType.NONE,
                    value="THUMBS_DOWN",
                    confidence=ctrl_conf,
                    progress=progress,
                    state=SystemState.WAITING,
                )

        # Reset control hold if hand is not in control gesture
        self.control_candidate = ControlGestureType.NONE
        self.control_hold_count = 0

        # 2. Detect Neutral Hand Pose or No Hand (for Alphabet signs)
        # Neutral occurs when: no hand in frame, confidence < 0.92, or flat open palm
        is_neutral = (
            landmarks is None
            or debug_info is None
            or confidence < self.min_confidence
            or raw_letter in ("None", "-", "Unknown", "Unknown Hand Gesture", "")
            or (debug_info and debug_info.get("is_flat_open_hand", False))
        )

        if is_neutral:
            self.neutral_counter += 1
            if self.neutral_counter >= self.neutral_frames_required:
                self.is_neutral_active = True
                self.last_committed_letter = ""  # Clears repeat lockout
                self.has_confirmed_current_word = False
                self.candidate_letter = ""
                self.candidate_hold_count = 0
                self.control_candidate = ControlGestureType.NONE
                self.control_hold_count = 0
            self.current_state = SystemState.WAITING
            return GestureEvent(
                EventType.NONE,
                state=SystemState.WAITING,
                progress=0.0,
            )

        # Reset neutral counter since hand is actively posing an alphabet letter
        self.neutral_counter = 0

        # 3. Process ASL Alphabet Sign (A-Z)
        if len(raw_letter) == 1 and raw_letter.isalpha() and confidence >= self.min_confidence:
            letter = raw_letter.upper()

            # Guard against continuous duplicate committing:
            # If the user is still holding the same letter they just committed,
            # require a neutral pose or gesture transition before committing it again!
            if letter == self.last_committed_letter and not self.is_neutral_active:
                self.current_state = SystemState.WAITING
                return GestureEvent(
                    EventType.NONE,
                    value=letter,
                    confidence=confidence,
                    progress=0.0,
                    state=SystemState.WAITING,
                )

            # Accumulate hold frames for candidate letter
            if letter == self.candidate_letter:
                self.candidate_hold_count += 1
            else:
                self.candidate_letter = letter
                self.candidate_hold_count = 1

            progress = min(1.0, self.candidate_hold_count / self.hold_frames)
            self.current_state = SystemState.COLLECTING

            if self.candidate_hold_count >= self.hold_frames:
                # Fire LETTER event!
                committed = self.candidate_letter
                self.last_committed_letter = committed
                self.is_neutral_active = False
                self.candidate_hold_count = 0
                self.cooldown_counter = self.cooldown_frames
                self.current_state = SystemState.COOLDOWN
                return GestureEvent(
                    EventType.LETTER,
                    value=committed,
                    confidence=confidence,
                    progress=1.0,
                    state=SystemState.COLLECTING,
                )

            return GestureEvent(
                EventType.NONE,
                value=letter,
                confidence=confidence,
                progress=progress,
                state=SystemState.COLLECTING,
            )

        # Any other unclassified or low-confidence hand pose
        self.current_state = SystemState.WAITING
        self.candidate_letter = ""
        self.candidate_hold_count = 0
        return GestureEvent(
            EventType.NONE,
            state=SystemState.WAITING,
            progress=0.0,
        )

    def reset(self) -> None:
        """Resets all internal event state."""
        self.current_state = SystemState.WAITING
        self.candidate_letter = ""
        self.candidate_hold_count = 0
        self.last_committed_letter = ""
        self.cooldown_counter = 0
        self.neutral_counter = 0
        self.is_neutral_active = True
        self.control_candidate = ControlGestureType.NONE
        self.control_hold_count = 0
        self.has_confirmed_current_word = False

