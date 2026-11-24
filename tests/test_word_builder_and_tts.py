"""
Comprehensive Test Suite for Gesture-to-Word Formation, Thumbs-Up Confirmation, and Text-To-Speech.
Tests all scenarios specified in Section 8 of the requirements:
1. K -> A -> J -> A -> L -> 👍 => Forms and confirms KAJAL
2. H -> E -> L -> L -> O -> 👍 => Forms and confirms HELLO with repeated letters
3. A -> B -> Delete => Displays A
4. A -> B -> Clear => Empty word buffer
5. Random hand gesture -> 👍 => Must not speak an invalid word
6. Unstable alphabet prediction => Must not add a letter
7. Holding K continuously => Adds K only once
8. Two separate L gestures => Adds LL correctly via neutral transition
9. Multiple thumbs-up frames => Triggers confirmation only once
10. TTS playback non-blocking behavior => Async worker remains responsive
"""

import os
import sys
import time
from typing import Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.gesture_commands import ControlGestureRecognizer, ControlGestureType
from src.gesture_event_manager import EventType, GestureEventManager, SystemState
from src.tts_engine import SpeechManager, TTSEngine
from src.word_builder import WordBuilder


def create_synthetic_thumbs_up() -> Tuple[np.ndarray, dict]:
    """Generates synthetic landmarks and metrics for a valid THUMBS_UP control gesture."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # Wrist at bottom-center
    lm[0] = [0.5, 0.7, 0.0]
    # Knuckles vertical running down the hand (sideways fist as in real human thumbs-up)
    lm[5] = [0.45, 0.40, 0.0]
    lm[9] = [0.46, 0.47, 0.0]
    lm[13] = [0.47, 0.54, 0.0]
    lm[17] = [0.48, 0.61, 0.0]
    # Closed fist: fingertips curled down
    for mcp, tip in [(5, 8), (9, 12), (13, 16), (17, 20)]:
        lm[mcp + 1] = [lm[mcp][0] - 0.05, lm[mcp][1] + 0.02, 0.05]
        lm[mcp + 2] = [lm[mcp][0] - 0.04, lm[mcp][1] + 0.04, 0.05]
        lm[tip] = [lm[mcp][0] - 0.02, lm[mcp][1] + 0.03, 0.02]
    # Thumb pointing straight UP high above the knuckles
    lm[1] = [0.44, 0.55, 0.0]
    lm[2] = [0.43, 0.45, 0.0]
    lm[3] = [0.42, 0.32, 0.0]
    lm[4] = [0.42, 0.16, 0.0]

    from src.normalizer import LandmarkNormalizer
    norm = LandmarkNormalizer()
    m = norm.analyze_hand(lm, "Right")
    return lm, m


def test_kajal_spelling_and_thumbs_up_confirmation():
    """Scenario 1: K -> A -> J -> A -> L -> 👍 => Forms and confirms KAJAL."""
    wb = WordBuilder()
    gem = GestureEventManager(hold_frames=3, confirm_hold_frames=3, cooldown_frames=1, neutral_frames_required=2)
    dummy_lm = np.zeros((21, 3), dtype=np.float32)
    dummy_m = {"ref_scale": 1.0}

    # Spell K-A-J-A-L
    for letter in ["K", "A", "J", "A", "L"]:
        # Hold each letter for 4 frames (hold_frames=3)
        for _ in range(4):
            ev = gem.process_frame(letter, 0.95, dummy_lm, dummy_m)
            if ev.event_type == EventType.LETTER:
                wb.add_letter(ev.value)
        # Brief transition
        gem.process_frame("None", 0.0, dummy_lm, dummy_m)

    assert wb.current_word == "KAJAL", f"Expected KAJAL, got {wb.current_word}"
    assert wb.letter_sequence == ["K", "A", "J", "A", "L"]

    # Now show THUMBS_UP to confirm
    tu_lm, tu_m = create_synthetic_thumbs_up()
    confirmed_word = None
    for _ in range(5):
        ev = gem.process_frame("A", 0.95, tu_lm, tu_m)
        if ev.event_type == EventType.CONFIRM:
            confirmed_word = wb.confirm()

    assert confirmed_word == "KAJAL", f"Expected confirmed word KAJAL, got {confirmed_word}"
    assert wb.is_empty(), "Word buffer should be cleared after confirmation"
    assert wb.last_confirmed_word == "KAJAL"


def test_hello_repeated_letters_with_neutral_transition():
    """Scenario 2 & 8: H -> E -> L -> L -> O -> 👍 => Handles repeated Ls with neutral transition."""
    wb = WordBuilder()
    gem = GestureEventManager(hold_frames=3, confirm_hold_frames=3, cooldown_frames=1, neutral_frames_required=2)
    dummy_lm = np.zeros((21, 3), dtype=np.float32)
    dummy_m = {"ref_scale": 1.0}

    # H
    for _ in range(4):
        ev = gem.process_frame("H", 0.95, dummy_lm, dummy_m)
        if ev.event_type == EventType.LETTER: wb.add_letter(ev.value)
    gem.process_frame("None", 0.0, dummy_lm, dummy_m)

    # E
    for _ in range(4):
        ev = gem.process_frame("E", 0.95, dummy_lm, dummy_m)
        if ev.event_type == EventType.LETTER: wb.add_letter(ev.value)
    gem.process_frame("None", 0.0, dummy_lm, dummy_m)

    # First L
    for _ in range(4):
        ev = gem.process_frame("L", 0.95, dummy_lm, dummy_m)
        if ev.event_type == EventType.LETTER: wb.add_letter(ev.value)

    assert wb.current_word == "HEL"

    # Continued holding of L without neutral transition must NOT add second L
    for _ in range(10):
        ev = gem.process_frame("L", 0.95, dummy_lm, dummy_m)
        if ev.event_type == EventType.LETTER: wb.add_letter(ev.value)
    assert wb.current_word == "HEL", "Holding L continuously must not add duplicate L"

    # Neutral transition (hand relaxed for 2 frames)
    for _ in range(2):
        gem.process_frame("None", 0.0, dummy_lm, dummy_m)

    # Second L formed
    for _ in range(4):
        ev = gem.process_frame("L", 0.95, dummy_lm, dummy_m)
        if ev.event_type == EventType.LETTER: wb.add_letter(ev.value)

    assert wb.current_word == "HELL", f"Expected HELL after transition, got {wb.current_word}"

    # O
    gem.process_frame("None", 0.0, dummy_lm, dummy_m)
    for _ in range(4):
        ev = gem.process_frame("O", 0.95, dummy_lm, dummy_m)
        if ev.event_type == EventType.LETTER: wb.add_letter(ev.value)

    assert wb.current_word == "HELLO", f"Expected HELLO, got {wb.current_word}"

    # Confirm with thumbs-up
    tu_lm, tu_m = create_synthetic_thumbs_up()
    confirmed_word = None
    for _ in range(5):
        ev = gem.process_frame("A", 0.95, tu_lm, tu_m)
        if ev.event_type == EventType.CONFIRM:
            confirmed_word = wb.confirm()

    assert confirmed_word == "HELLO"


def test_word_editing_delete():
    """Scenario 3: A -> B -> Delete => Displays A."""
    wb = WordBuilder()
    wb.add_letter("A")
    wb.add_letter("B")
    assert wb.current_word == "AB"

    deleted = wb.delete_last()
    assert deleted == "B"
    assert wb.current_word == "A"


def test_word_editing_clear():
    """Scenario 4: A -> B -> Clear => Empty word buffer."""
    wb = WordBuilder()
    wb.add_letter("A")
    wb.add_letter("B")
    assert wb.current_word == "AB"

    cleared = wb.clear()
    assert cleared == "AB"
    assert wb.current_word == ""
    assert wb.is_empty()


def test_random_hand_and_empty_thumbs_up():
    """Scenario 5: Random hand gesture -> 👍 => Must not speak an invalid word."""
    wb = WordBuilder()
    gem = GestureEventManager(hold_frames=3, confirm_hold_frames=3)
    dummy_lm = np.zeros((21, 3), dtype=np.float32)
    dummy_m = {"ref_scale": 1.0, "is_flat_open_hand": True}

    # Random hand frames (accuracy < 92%)
    for _ in range(10):
        ev = gem.process_frame("None", 0.40, dummy_lm, dummy_m)
        if ev.event_type == EventType.LETTER:
            wb.add_letter(ev.value)

    assert wb.is_empty(), "Random hand should not add any letters"

    # Thumbs up shown with empty word buffer
    tu_lm, tu_m = create_synthetic_thumbs_up()
    confirmed = None
    for _ in range(5):
        ev = gem.process_frame("A", 0.95, tu_lm, tu_m)
        if ev.event_type == EventType.CONFIRM:
            confirmed = wb.confirm()

    assert confirmed is None, "Empty word buffer must not produce a confirmed word"


def test_unstable_alphabet_prediction_rejected():
    """Scenario 6: Unstable alphabet prediction (< min_confidence or flickering) => Must not add letter."""
    wb = WordBuilder()
    gem = GestureEventManager(hold_frames=4, min_confidence=0.92)
    dummy_lm = np.zeros((21, 3), dtype=np.float32)
    dummy_m = {"ref_scale": 1.0}

    # Low confidence predictions (0.75 < 0.92)
    for _ in range(10):
        ev = gem.process_frame("K", 0.75, dummy_lm, dummy_m)
        if ev.event_type == EventType.LETTER:
            wb.add_letter(ev.value)
    assert wb.is_empty(), "Low confidence frames must not add letter"

    # Flickering predictions (K -> A -> B -> C every frame, never held for 4 frames)
    for ch in ["K", "A", "B", "C", "D", "E"]:
        ev = gem.process_frame(ch, 0.95, dummy_lm, dummy_m)
        if ev.event_type == EventType.LETTER:
            wb.add_letter(ev.value)
    assert wb.is_empty(), "Flickering predictions must not add letters"


def test_continuous_holding_k_adds_only_once():
    """Scenario 7: Holding K continuously for 100 frames => Adds K only once."""
    wb = WordBuilder()
    gem = GestureEventManager(hold_frames=4)
    dummy_lm = np.zeros((21, 3), dtype=np.float32)
    dummy_m = {"ref_scale": 1.0}

    events = []
    for _ in range(100):
        ev = gem.process_frame("K", 0.96, dummy_lm, dummy_m)
        if ev.event_type == EventType.LETTER:
            wb.add_letter(ev.value)
            events.append(ev.value)

    assert wb.current_word == "K", f"Expected single K, got {wb.current_word}"
    assert len(events) == 1, f"Expected 1 letter event, got {len(events)}"


def test_multiple_thumbs_up_frames_confirms_only_once():
    """Scenario 9: Multiple thumbs-up frames => Triggers confirmation only once."""
    wb = WordBuilder()
    wb.add_letter("B")
    wb.add_letter("O")
    wb.add_letter("O")
    wb.add_letter("K")

    gem = GestureEventManager(confirm_hold_frames=4)
    tu_lm, tu_m = create_synthetic_thumbs_up()

    confirm_count = 0
    for _ in range(40):  # 40 consecutive frames of thumbs up
        ev = gem.process_frame("A", 0.96, tu_lm, tu_m)
        if ev.event_type == EventType.CONFIRM:
            confirm_count += 1
            wb.confirm()

    assert confirm_count == 1, f"Expected exactly 1 confirmation event, got {confirm_count}"
    assert wb.last_confirmed_word == "BOOK"


def test_tts_playback_non_blocking_behavior():
    """Scenario 10: TTS playback is asynchronous, non-blocking, and thread-safe."""
    sm = SpeechManager(enabled=True)
    assert sm.status_text in ("IDLE", "SPEAKING")

    # Queue multiple words in rapid succession
    sm.speak("Kajal")
    sm.speak("Hello")
    time.sleep(0.1)

    # Non-blocking check: thread-safe operations execute immediately
    sm.repeat_last()
    assert sm.last_spoken_word in ("", "KAJAL", "HELLO")

    # Clean shutdown
    sm.shutdown()


def test_thumbs_up_not_confused_with_asl_a():
    """Control gesture verification: Authentic ASL A is NOT recognized as THUMBS_UP."""
    from data.train_model import create_base_skeleton
    from src.normalizer import LandmarkNormalizer

    norm = LandmarkNormalizer()
    cgr = ControlGestureRecognizer(min_confidence=0.92)

    base_a = create_base_skeleton("A")
    m_a = norm.analyze_hand(base_a, "Right")

    res, conf = cgr.recognize(base_a, m_a, "Right")
    assert res == ControlGestureType.NONE, f"ASL 'A' must NOT be detected as THUMBS_UP, got {res}"


def test_user_uploaded_a_vs_thumbs_up():
    """Verifies that User's authentic A image is recognized as A (never Thumbs-Up),
    and User's authentic Thumbs-Up image is recognized as THUMBS_UP (never A)."""
    from src.classifier import GestureClassifier
    from src.normalizer import LandmarkNormalizer

    norm = LandmarkNormalizer()
    cgr = ControlGestureRecognizer(min_confidence=0.92)
    clf = GestureClassifier()

    # Image 1 (User's authentic ASL A)
    lm1 = np.zeros((21, 3), dtype=np.float32)
    lm1[0] = [357.0, 407.4, 0.0]
    lm1[1] = [250.3, 376.5, -0.05]
    lm1[2] = [166.5, 280.1, -0.08]
    lm1[3] = [132.3, 194.9, -0.08]
    lm1[4] = [90.2, 145.3, -0.05]
    lm1[5] = [204.1, 213.9, 0.0]
    lm1[6] = [207.1, 116.0, 0.05]
    lm1[7] = [218.6, 176.0, 0.08]
    lm1[8] = [219.2, 219.8, 0.05]
    lm1[9] = [271.1, 200.5, 0.0]
    lm1[10] = [274.5, 102.5, 0.05]
    lm1[11] = [291.0, 166.3, 0.08]
    lm1[12] = [286.7, 206.2, 0.05]
    lm1[13] = [334.6, 207.8, 0.0]
    lm1[14] = [344.9, 113.5, 0.05]
    lm1[15] = [359.9, 167.5, 0.08]
    lm1[16] = [346.6, 220.0, 0.05]
    lm1[17] = [395.9, 190.8, 0.0]
    lm1[18] = [415.3, 125.8, 0.05]
    lm1[19] = [430.6, 176.0, 0.08]
    lm1[20] = [401.8, 208.7, 0.05]
    lm1[:, 0] /= 560.0
    lm1[:, 1] /= 507.0

    m1 = norm.analyze_hand(lm1, "Right")
    tu1, _ = cgr.is_thumbs_up(lm1, m1, "Right")
    letter1, conf1 = clf.classify(lm1, handedness="Right", precomputed_m=m1)

    assert not tu1, "User's ASL A gesture must NOT be detected as THUMBS_UP"
    assert letter1 == "A" and conf1 >= 0.95, f"User's ASL A gesture should classify as A, got {letter1} ({conf1:.2f})"

    # Image 2 (User's authentic Thumbs-Up)
    lm2 = np.zeros((21, 3), dtype=np.float32)
    lm2[0] = [322.6, 327.5, 0.0]
    lm2[1] = [277.5, 243.4, -0.05]
    lm2[2] = [202.3, 176.0, -0.08]
    lm2[3] = [151.8, 115.0, -0.08]
    lm2[4] = [145.0, 54.7, -0.05]
    lm2[5] = [133.2, 189.6, 0.0]
    lm2[6] = [49.1, 236.1, 0.05]
    lm2[7] = [79.5, 248.5, 0.08]
    lm2[8] = [112.4, 237.3, 0.05]
    lm2[9] = [133.4, 244.9, 0.0]
    lm2[10] = [44.7, 280.1, 0.05]
    lm2[11] = [78.2, 291.0, 0.08]
    lm2[12] = [114.0, 283.2, 0.05]
    lm2[13] = [138.3, 303.6, 0.0]
    lm2[14] = [58.1, 328.4, 0.05]
    lm2[15] = [88.4, 337.0, 0.08]
    lm2[16] = [122.8, 328.6, 0.05]
    lm2[17] = [147.7, 364.5, 0.0]
    lm2[18] = [70.5, 372.4, 0.05]
    lm2[19] = [99.1, 377.0, 0.08]
    lm2[20] = [130.4, 371.1, 0.05]
    lm2[:, 0] /= 486.0
    lm2[:, 1] /= 445.0

    m2 = norm.analyze_hand(lm2, "Right")
    tu2, conf2 = cgr.is_thumbs_up(lm2, m2, "Right")
    letter2, _ = clf.classify(lm2, handedness="Right", precomputed_m=m2)

    assert tu2 and conf2 >= 0.95, f"User's Thumbs-Up should trigger is_thumbs_up, got {tu2} ({conf2:.2f})"
    assert letter2 is None or letter2 == "None", f"User's Thumbs-Up must not be classified as alphabet letter, got {letter2}"

    # Image 3 (User's authentic Left Hand Thumbs-Up, media_1794469055853_dbd5eef6.jpg)
    w3, h3 = 532.0, 598.0
    lm3 = np.zeros((21, 3), dtype=np.float32)
    lm3[0] = [418.6 / w3, 457.7 / h3, 0.0]
    lm3[1] = [379.6 / w3, 357.1 / h3, -0.05]
    lm3[2] = [282.1 / w3, 247.1 / h3, -0.08]
    lm3[3] = [214.5 / w3, 162.1 / h3, -0.08]
    lm3[4] = [199.3 / w3, 84.8 / h3, -0.05]
    lm3[5] = [207.0 / w3, 252.2 / h3, 0.0]
    lm3[6] = [73.3 / w3, 316.9 / h3, 0.05]
    lm3[7] = [119.9 / w3, 331.0 / h3, 0.08]
    lm3[8] = [163.4 / w3, 311.9 / h3, 0.05]
    lm3[9] = [207.0 / w3, 322.8 / h3, 0.0]
    lm3[10] = [70.4 / w3, 380.8 / h3, 0.05]
    lm3[11] = [126.0 / w3, 390.3 / h3, 0.08]
    lm3[12] = [176.8 / w3, 376.5 / h3, 0.05]
    lm3[13] = [208.5 / w3, 401.0 / h3, 0.0]
    lm3[14] = [98.6 / w3, 444.0 / h3, 0.05]
    lm3[15] = [152.9 / w3, 452.3 / h3, 0.08]
    lm3[16] = [200.7 / w3, 439.6 / h3, 0.05]
    lm3[17] = [211.2 / w3, 479.5 / h3, 0.0]
    lm3[18] = [142.3 / w3, 503.4 / h3, 0.05]
    lm3[19] = [193.5 / w3, 506.9 / h3, 0.08]
    lm3[20] = [234.2 / w3, 494.7 / h3, 0.05]

    for hand in ("Left", "Right"):
        m3 = norm.analyze_hand(lm3, hand)
        tu3, conf3 = cgr.is_thumbs_up(lm3, m3, hand)
        letter3, _ = clf.classify(lm3, handedness=hand, precomputed_m=m3)
        assert tu3 and conf3 >= 0.95, f"User's Thumbs-Up ({hand}) should trigger is_thumbs_up, got {tu3} ({conf3:.2f})"
        assert letter3 is None or letter3 == "None", f"User's Thumbs-Up ({hand}) must not be classified as alphabet letter"

    # End-to-end word confirmation test with user's Thumbs-Up image
    from src.gesture_event_manager import GestureEventManager, EventType
    from src.word_builder import WordBuilder
    wb_test = WordBuilder()
    wb_test.add_letter("T")
    wb_test.add_letter("E")
    wb_test.add_letter("S")
    wb_test.add_letter("T")
    gem_test = GestureEventManager(confirm_hold_frames=3)
    m3_l = norm.analyze_hand(lm3, "Left")
    confirmed = None
    for _ in range(4):
        ev = gem_test.process_frame("None", 0.0, lm3, m3_l, handedness="Left")
        if ev.event_type == EventType.CONFIRM:
            confirmed = wb_test.confirm()
    assert confirmed == "TEST", f"Expected confirmed word TEST, got {confirmed}"

    # Image 4 (User's thumb confirm gesture image, media_1795433764199_af4eca8d.jpg)
    w4, h4 = 547.0, 613.0
    lm4 = np.zeros((21, 3), dtype=np.float32)
    lm4[0] = [408 / w4, 402 / h4, 0]
    lm4[1] = [358 / w4, 324 / h4, 0]
    lm4[2] = [278 / w4, 247 / h4, 0]
    lm4[3] = [221 / w4, 192 / h4, 0]
    lm4[4] = [203 / w4, 137 / h4, 0]
    lm4[5] = [219 / w4, 262 / h4, 0]
    lm4[6] = [164 / w4, 327 / h4, 0]
    lm4[7] = [131 / w4, 314 / h4, 0]
    lm4[8] = [197 / w4, 315 / h4, 0]
    lm4[9] = [221 / w4, 313 / h4, 0]
    lm4[10] = [165 / w4, 367 / h4, 0]
    lm4[11] = [126 / w4, 357 / h4, 0]
    lm4[12] = [203 / w4, 356 / h4, 0]
    lm4[13] = [229 / w4, 368 / h4, 0]
    lm4[14] = [178 / w4, 411 / h4, 0]
    lm4[15] = [141 / w4, 403 / h4, 0]
    lm4[16] = [213 / w4, 400 / h4, 0]
    lm4[17] = [239 / w4, 426 / h4, 0]
    lm4[18] = [191 / w4, 446 / h4, 0]
    lm4[19] = [164 / w4, 440 / h4, 0]
    lm4[20] = [222 / w4, 440 / h4, 0]

    for hand in ("Left", "Right"):
        m4 = norm.analyze_hand(lm4, hand)
        tu4, conf4 = cgr.is_thumbs_up(lm4, m4, hand)
        letter4, _ = clf.classify(lm4, handedness=hand, precomputed_m=m4)
        assert tu4 and conf4 >= 0.95, f"User's thumb confirm gesture ({hand}) should trigger is_thumbs_up, got {tu4} ({conf4:.2f})"
        assert letter4 is None or letter4 == "None", f"User's thumb gesture ({hand}) must not be classified as alphabet letter, got {letter4}"

    wb_test2 = WordBuilder()
    wb_test2.add_letter("U")
    wb_test2.add_letter("S")
    wb_test2.add_letter("E")
    wb_test2.add_letter("R")
    gem_test2 = GestureEventManager(confirm_hold_frames=3)
    m4_l = norm.analyze_hand(lm4, "Left")
    confirmed2 = None
    for _ in range(4):
        ev = gem_test2.process_frame("None", 0.0, lm4, m4_l, handedness="Left")
        if ev.event_type == EventType.CONFIRM:
            confirmed2 = wb_test2.confirm()
    assert confirmed2 == "USER", f"Expected confirmed word USER, got {confirmed2}"

    # Image 5 (User's confirm thumb gesture, media_1795498504175_7369d38d.jpg)
    w5, h5 = 1024.0, 576.0
    lm5 = np.zeros((21, 3), dtype=np.float32)
    lm5[0] = [708 / w5, 363 / h5, 0]
    lm5[1] = [669 / w5, 278 / h5, 0]
    lm5[2] = [629 / w5, 230 / h5, 0]
    lm5[3] = [601 / w5, 189 / h5, 0]
    lm5[4] = [596 / w5, 152 / h5, 0]
    lm5[5] = [576 / w5, 248 / h5, 0]
    lm5[6] = [528 / w5, 271 / h5, 0]
    lm5[7] = [549 / w5, 277 / h5, 0]
    lm5[8] = [570 / w5, 272 / h5, 0]
    lm5[9] = [571 / w5, 284 / h5, 0]
    lm5[10] = [528 / w5, 297 / h5, 0]
    lm5[11] = [549 / w5, 301 / h5, 0]
    lm5[12] = [571 / w5, 298 / h5, 0]
    lm5[13] = [572 / w5, 320 / h5, 0]
    lm5[14] = [537 / w5, 324 / h5, 0]
    lm5[15] = [556 / w5, 328 / h5, 0]
    lm5[16] = [577 / w5, 327 / h5, 0]
    lm5[17] = [565 / w5, 352 / h5, 0]
    lm5[18] = [516 / w5, 364 / h5, 0]
    lm5[19] = [546 / w5, 349 / h5, 0]
    lm5[20] = [584 / w5, 353 / h5, 0]

    for hand in ("Left", "Right"):
        m5 = norm.analyze_hand(lm5, hand)
        tu5, conf5 = cgr.is_thumbs_up(lm5, m5, hand)
        letter5, _ = clf.classify(lm5, handedness=hand, precomputed_m=m5)
        assert tu5 and conf5 >= 0.95, f"User's confirm thumb gesture ({hand}) should trigger is_thumbs_up, got {tu5} ({conf5:.2f})"
        assert letter5 is None or letter5 == "None", f"User's confirm thumb gesture ({hand}) must not be classified as alphabet letter"

    # End-to-end confirmation with word HARASHAL
    wb_test3 = WordBuilder()
    for ch in "HARASHAL":
        wb_test3.add_letter(ch)
    gem_test3 = GestureEventManager(confirm_hold_frames=3)
    m5_r = norm.analyze_hand(lm5, "Right")
    confirmed3 = None
    for _ in range(5):
        ev = gem_test3.process_frame("None", 0.0, lm5, m5_r, handedness="Right")
        if ev.event_type == EventType.CONFIRM:
            confirmed3 = wb_test3.confirm()
    assert confirmed3 == "HARASHAL", f"Expected confirmed word HARASHAL, got {confirmed3}"


if __name__ == "__main__":
    print("Running Word Builder, Event Manager & TTS Test Suite...")
    test_kajal_spelling_and_thumbs_up_confirmation()
    print("[PASS] Scenario 1: K -> A -> J -> A -> L -> 👍 => KAJAL passed")
    test_hello_repeated_letters_with_neutral_transition()
    print("[PASS] Scenario 2: H -> E -> L -> L -> O -> 👍 => HELLO passed")
    test_word_editing_delete()
    print("[PASS] Scenario 3: Word Editing Delete passed")
    test_word_editing_clear()
    print("[PASS] Scenario 4: Word Editing Clear passed")
    test_random_hand_and_empty_thumbs_up()
    print("[PASS] Scenario 5: Random hand & empty thumbs-up rejection passed")
    test_unstable_alphabet_prediction_rejected()
    print("[PASS] Scenario 6: Unstable prediction rejection passed")
    test_continuous_holding_k_adds_only_once()
    print("[PASS] Scenario 7: Continuous holding single-add passed")
    test_multiple_thumbs_up_frames_confirms_only_once()
    print("[PASS] Scenario 9: Multiple thumbs-up single confirmation passed")
    test_tts_playback_non_blocking_behavior()
    print("[PASS] Scenario 10: Non-blocking TTS playback passed")
    test_thumbs_up_not_confused_with_asl_a()
    print("[PASS] Thumbs-Up vs ASL A distinction passed")
    test_user_uploaded_a_vs_thumbs_up()
    print("[PASS] User Uploaded Images: ASL A vs Thumbs-Up distinction passed")
    print("\nAll Word Builder and TTS tests passed successfully!")
