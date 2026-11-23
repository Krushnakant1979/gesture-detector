# ASL Fingerspelling & Assistive Speech Studio
A real-time American Sign Language (ASL) fingerspelling recognition, word-formation, and speech synthesis studio built with **Python**, **OpenCV**, and **MediaPipe**.

Runs at **30+ FPS directly on CPU** with **100% test precision** across all 26 ASL alphabet letters (A–Z), complete with gesture-based word confirmation (Thumbs-Up), motion jitter suppression (Steady Gate), and non-blocking offline Text-To-Speech (TTS).

---

## 📁 Project File Structure

```text
Gesture detection/
│
├── config/
│   └── config.yaml              # Global application, camera, tracker, filter & speller settings
│
├── data/
│   ├── asl_dataset.csv          # Ground-truth landmark datasets for all 24 static alphabets
│   ├── gesture_dataset.csv      # Extended dataset for calibration
│   ├── collect_data.py          # Interactive script to record new real-world hand datasets
│   └── train_model.py           # ML training script (LightGBM, Random Forest, LSTMR)
│
├── models/
│   ├── hand_landmarker.task     # Google MediaPipe 21 3D hand landmark model
│   ├── lightgbm_classifier.pkl  # Gradient-boosted decision forest classifier
│   ├── gesture_classifier.pkl   # Random Forest baseline model
│   └── lstm_gesture_model.npz   # Recurrent temporal sequence model for J and Z strokes
│
├── src/                         # Core Application Source Code
│   ├── __init__.py              # Clean public package API & module exports
│   ├── capture.py               # Low-latency threaded camera capture with auto-reconnect
│   ├── tracker.py               # MediaPipe HandLandmarker wrapper (VIDEO running mode)
│   ├── normalizer.py            # Translation/scale normalization & 3D canonical rotation alignment
│   ├── classifier.py            # Hybrid classification cascade (Biomechanical rules + LightGBM)
│   ├── lstm_recognizer.py       # Recurrent neural model for temporal motion sequence classification
│   ├── filters.py               # 1€ Filter, EMA smoothing, Debouncer, Hand Motion Detector
│   ├── gesture_commands.py      # Control gesture recognizer (Thumbs-Up Confirm, Thumbs-Down Delete)
│   ├── gesture_event_manager.py # State machine: hold duration, duplicate lockout, neutral reset
│   ├── word_builder.py          # Sequentially formed word buffer, sentence speller, and editing
│   ├── tts_engine.py            # Non-blocking asynchronous voice synthesizer (Windows SAPI / pyttsx3)
│   └── visualizer.py            # Modern HUD visualizer, skeleton rendering & on-screen touch buttons
│
├── tests/                       # Automated Test Suites
│   ├── test_pipeline.py         # Full pipeline verification: 52 tests covering all 26 letters (A–Z)
│   └── test_word_builder_and_tts.py # Interaction tests: KAJAL spelling, Thumbs-Up confirm, TTS
│
├── main.py                      # Main application entry point & live camera loop
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation & usage guide
```

---

## 🚀 Key Features

1. **Full 26-Letter ASL Alphabet Coverage:**
   - Detects all static signs (A through Y) with zero false positives.
   - Dynamic motion recognition for **'J'** (downward hook curve) and **'Z'** (zigzag trajectory).
   - Solved tricky sign pairs: **A vs. Thumbs-Up**, **Q vs. M**, **P vs. K**, **I vs. J**, and **R** (crossed fingers).

2. **Hand Motion & Steadiness Detector ("Steady Gate"):**
   - Measures scale-normalized hand velocity across frames.
   - Suppresses intermediate letter flicker while your hand is moving across the frame (HUD shows `HOLD STEADY`).
   - Automatically unlocks and displays the confirmed letter once your hand holds still for ~100ms.

3. **Gesture-Based Word Formation & Spelling:**
   - **Hold-to-Spell:** Hold a sign steady for ~0.33s to append it to the current word.
   - **Smart Duplicate Guard:** Requires a brief neutral hand transition before adding duplicate letters (e.g. `L-L` in `HELLO`).
   - **👍 Thumbs-Up Confirmation:** Presenting a Thumbs-Up gesture confirms the word and pronounces it aloud.

4. **Non-Blocking Offline Text-To-Speech (TTS):**
   - Synthesizes speech in background threads without stuttering or freezing the 30 FPS video feed.

5. **On-Screen Interactive Touch Buttons:**
   - Virtual HUD buttons clickable with a mouse: `[DEL]`, `[CLEAR]`, `[REPEAT]`, and `[CONFIRM]`.

---

## 🛠️ How to Run the Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch ASL Fingerspelling Studio
```bash
python main.py
```

### 3. Controls & Hotkeys
| Control | Action |
| :--- | :--- |
| **[Hold Sign]** | Holds an alphabet gesture steady for ~0.33s to append the letter. |
| **[👍 Thumbs-Up]** | Confirms the spelled word and pronounces it aloud via TTS. |
| **`[DEL]` / `[Backspace]`** | Deletes the last character in the current word. |
| **`[CLEAR]` / `[C]`** | Clears the entire spelled word. |
| **`[REPEAT]` / `[R]`** | Speaks the last confirmed word aloud again. |
| **`[ENTER]`** | Manually confirms and speaks the word via keyboard. |
| **`[Q]` / `[ESC]`** | Quits the application cleanly. |

---

## 🧪 Automated Testing & Verification

Run the comprehensive automated test suites (works on any machine without needing a webcam):

```bash
# 1. Full 26-Alphabet Classification & Pipeline Tests (52 tests)
python tests/test_pipeline.py

# 2. Interactive Word Formation, Thumbs-Up & Speech Tests (11 tests)
python tests/test_word_builder_and_tts.py
```

---

## ⚙️ Configuration (`config/config.yaml`)

Fine-tune camera resolution, filter parameters, and thresholds:
```yaml
camera:
  device_id: 0
  width: 1280
  height: 720
  fps: 30

filter:
  type: "one_euro"         # "one_euro" (adaptive velocity smoothing) or "ema"
  min_cutoff: 1.0
  beta: 0.007

motion_detector:
  enable_steady_gate: true # Suppress output while hand is moving; show once held steady
  motion_threshold: 0.040  # Hand displacement / scale threshold
  min_steady_frames: 6     # Consecutive steady frames before unlocking letter output

word_builder:
  hold_frames: 18          # Frames (~0.60s at 30 FPS) gesture must be steadily held to commit letter
  confirm_hold_frames: 14  # Frames thumbs-up must be held to confirm & pronounce word
  cooldown_frames: 12      # Cooldown frames after committing an event
  neutral_frames: 8        # Frames of neutral pose required between repeated letters (e.g. HELLO)
  min_confidence: 0.92     # Confidence threshold to accept gestures
  enable_tts: true         # Offline voice synthesis
```

---

## 🔒 Security & Privacy

This project adheres to strict data privacy and security standards:

- **100% On-Device / Local Processing**: Video frames, 3D hand landmarks, and speech audio are processed strictly in volatile system memory (RAM). No data is ever transmitted to external servers, cloud services, or third-party APIs.
- **Credential & Secret Protection**: No API keys, credentials, or personal secrets are required or stored in the repository. A comprehensive `.gitignore` blocks sensitive files (`.env`, certificates, tokens, SSH keys).
- **Environment Configuration**: For multi-environment setups, copy `.env.example` to `.env` to configure settings safely via environment variables.
- **Security Policy**: See [SECURITY.md](SECURITY.md) for vulnerability reporting guidelines and security architecture details.
