"""
Master Integration Test Runner for ASL Gesture Recognition Studio.
Orchestrates unit tests, static alphabet distinction tests, dynamic strokes,
and generates the full 26-alphabet classification accuracy report.
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Re-export all unit tests for complete backward compatibility
from tests.unit.test_normalizer import (
    test_normalizer_invariance,
    test_3d_canonical_alignment,
)
from tests.unit.test_tracker_and_filters import (
    test_debouncer_stabilization,
    test_hand_motion_detector_and_steady_gate,
    test_visualizer_rendering,
    test_tracker_initialization,
    test_exponential_moving_average,
    test_one_euro_filter,
    test_word_predictor,
    test_clahe_preprocessing,
)
from tests.unit.test_classifier import (
    test_asl_classifier,
    test_lightgbm_and_lstm,
    test_stroke_trajectory_tracker,
    test_dynamic_j_and_z_strokes,
    test_unknown_gesture_rejection,
    test_lstmr_dynamic_j_and_z_trajectories,
    test_full_alphabet_classification_report,
)
from tests.unit.test_static_alphabets import (
    test_k_vs_v_distinction,
    test_strict_a_gesture,
    test_c_vs_open_palm_distinction,
    test_strict_b_gesture,
    test_strict_d_gesture,
    test_strict_l_gesture,
    test_strict_m_gesture,
    test_strict_n_gesture,
    test_strict_o_gesture,
    test_strict_p_gesture,
    test_strict_q_gesture,
    test_strict_r_gesture,
    test_all_24_static_alphabets,
    test_strict_g_gesture,
    test_strict_i_vs_j_distinction,
    test_strict_s_vs_t_gesture,
    test_strict_u_vs_v_gesture,
    test_strict_e_gesture,
    test_strict_x_gesture,
)
from tests.test_word_builder_and_tts import (
    test_kajal_spelling_and_thumbs_up_confirmation,
    test_hello_repeated_letters_with_neutral_transition,
    test_word_editing_delete,
    test_word_editing_clear,
    test_random_hand_and_empty_thumbs_up,
    test_unstable_alphabet_prediction_rejected,
    test_continuous_holding_k_adds_only_once,
    test_multiple_thumbs_up_frames_confirms_only_once,
    test_tts_playback_non_blocking_behavior,
    test_thumbs_up_not_confused_with_asl_a,
    test_user_uploaded_a_vs_thumbs_up,
)


def run_all_self_checks() -> None:
    """Executes the complete ASL Studio automated test suite."""
    print("Running self-checks...")
    test_normalizer_invariance()
    print("[PASS] Normalizer test passed")
    test_3d_canonical_alignment()
    print("[PASS] 3D Canonical Alignment test passed")
    test_debouncer_stabilization()
    print("[PASS] Debouncer test passed")
    test_hand_motion_detector_and_steady_gate()
    print("[PASS] Hand Motion Detector & Steady Gate test passed")
    test_visualizer_rendering()
    print("[PASS] Visualizer test passed")
    test_tracker_initialization()
    print("[PASS] Tracker test passed")
    test_clahe_preprocessing()
    print("[PASS] CLAHE Preprocessing test passed")
    test_asl_classifier()
    print("[PASS] ASL Classifier test passed")
    test_lightgbm_and_lstm()
    print("[PASS] LightGBM and LSTMR test passed")
    test_all_24_static_alphabets()
    print("[PASS] All 24 Static Alphabets test passed (100% Right + Left parity)")
    test_dynamic_j_and_z_strokes()
    print("[PASS] Dynamic J and Z strokes test passed")
    test_unknown_gesture_rejection()
    print("[PASS] Unknown gesture rejection test passed")
    test_exponential_moving_average()
    print("[PASS] Exponential Moving Average test passed")
    test_one_euro_filter()
    print("[PASS] 1€ (One Euro) Filter test passed")
    test_word_predictor()
    print("[PASS] Word Predictor test passed")
    test_stroke_trajectory_tracker()
    print("[PASS] Stroke Trajectory Tracker test passed")
    test_k_vs_v_distinction()
    print("[PASS] K vs V distinction test passed")
    test_strict_a_gesture()
    print("[PASS] Strict A gesture distinction test passed")
    test_c_vs_open_palm_distinction()
    print("[PASS] C vs Open Palm distinction test passed")
    test_strict_b_gesture()
    print("[PASS] Strict B gesture test passed")
    test_strict_d_gesture()
    print("[PASS] Strict D gesture test passed")
    test_strict_l_gesture()
    print("[PASS] Strict L gesture test passed")
    test_strict_m_gesture()
    print("[PASS] Strict M gesture test passed")
    test_strict_n_gesture()
    print("[PASS] Strict N gesture test passed")
    test_strict_o_gesture()
    print("[PASS] Strict O gesture test passed")
    test_strict_p_gesture()
    print("[PASS] Strict P gesture test passed")
    test_strict_q_gesture()
    print("[PASS] Strict Q gesture test passed")
    test_strict_r_gesture()
    print("[PASS] Strict R gesture test passed")
    test_strict_g_gesture()
    print("[PASS] Strict G gesture test passed")
    test_strict_i_vs_j_distinction()
    print("[PASS] Strict I vs J distinction test passed")
    test_strict_s_vs_t_gesture()
    print("[PASS] Strict S vs T gesture test passed")
    test_strict_u_vs_v_gesture()
    print("[PASS] Strict U vs V gesture test passed")
    test_strict_e_gesture()
    print("[PASS] Strict E gesture test passed")
    test_strict_x_gesture()
    print("[PASS] Strict X gesture test passed")
    test_lstmr_dynamic_j_and_z_trajectories()
    print("[PASS] LSTMR Dynamic J and Z trajectories test passed")
    test_full_alphabet_classification_report()
    print("[PASS] Full Alphabet Classification Report test passed")

    # Word Builder, Gesture Event Manager, Thumbs-Up Confirmation & Text-To-Speech Tests
    test_kajal_spelling_and_thumbs_up_confirmation()
    print("[PASS] Word Builder: K -> A -> J -> A -> L -> 👍 => KAJAL passed")
    test_hello_repeated_letters_with_neutral_transition()
    print("[PASS] Word Builder: H -> E -> L -> L -> O -> 👍 => HELLO passed")
    test_word_editing_delete()
    print("[PASS] Word Builder: Delete last letter passed")
    test_word_editing_clear()
    print("[PASS] Word Builder: Clear word passed")
    test_random_hand_and_empty_thumbs_up()
    print("[PASS] Event Manager: Random hand & empty thumbs-up rejection passed")
    test_unstable_alphabet_prediction_rejected()
    print("[PASS] Event Manager: Unstable prediction rejection passed")
    test_continuous_holding_k_adds_only_once()
    print("[PASS] Event Manager: Continuous holding single-add passed")
    test_multiple_thumbs_up_frames_confirms_only_once()
    print("[PASS] Event Manager: Multiple thumbs-up single confirmation passed")
    test_tts_playback_non_blocking_behavior()
    print("[PASS] TTS Engine: Non-blocking speech playback passed")
    test_thumbs_up_not_confused_with_asl_a()
    print("[PASS] Control Gestures: Thumbs-Up vs ASL A distinction passed")
    test_user_uploaded_a_vs_thumbs_up()
    print("[PASS] User Uploaded Images: ASL A vs Thumbs-Up distinction passed")

    print("\nAll ASL pipeline tests completed successfully!")


if __name__ == "__main__":
    run_all_self_checks()
