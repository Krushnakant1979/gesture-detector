# Real-Time ASL Fingerspelling Alphabet Recognition Studio

A real-time American Sign Language (ASL) fingerspelling recognition system built with **Python 3**, **OpenCV**, and **Google MediaPipe Tasks**.

When a user forms hand gestures matching the ASL alphabet chart (**A through Z**), the application recognizes the hand shape in real time, displays the detected letter on screen, and allows spelling out words and sentences.

---

## ASL Alphabet Recognition Coverage

The system detects all 26 ASL alphabet signs depicted in the standard fingerspelling chart:

| Letter | Hand Gesture Description |
| :---: | :--- |
| **A** | Fist with thumb resting straight upright alongside the index knuckle |
| **B** | 4 fingers upright and held close together, thumb tucked across the palm |
| **C** | All 4 fingers and thumb curved into an open "C" cup |
| **D** | Index finger pointing straight up, thumb touching middle fingertip forming a loop |
| **E** | All 4 fingers curled down tight, thumb tucked across underneath the fingertips |
| **F** | Index fingertip and thumb tip touching (circle), middle, ring, and pinky straight up |
| **G** | Index finger pointing horizontally/forward with thumb parallel to it |
| **H** | Index and middle fingers extended together horizontally |
| **I** | Pinky finger extended straight up, other fingers curled in a fist |
| **J** | Pinky extended straight up tracing a "J" curve |
| **K** | Index and middle fingers upright in a V with thumb pointing up between them |
| **L** | Index pointing straight up, thumb extended horizontally at 90° forming an "L" |
| **M** | Fist with thumb tucked under the first 3 fingers (index, middle, ring) |
| **N** | Fist with thumb tucked under the first 2 fingers (index, middle) |
| **O** | All 5 fingertips curled and touching to form an "O" shape |
| **P** | Like "K", but tilted downward (index pointing forward, middle pointing down) |
| **Q** | Like "G", but tilted pointing downwards |
| **R** | Index and middle fingers extended upright and crossed over each other |
| **S** | Fist with thumb wrapped across the front of the curled fingers |
| **T** | Fist with thumb tucked between index and middle fingers |
| **U** | Index and middle fingers upright and held touching/parallel |
| **V** | Index and middle fingers upright and spread apart in a "V" |
| **W** | Index, middle, and ring fingers upright and spread in a "W", pinky curled |
| **X** | Fist with index finger crooked/bent into a hook shape |
| **Y** | Thumb and pinky extended out wide, middle three fingers curled |
| **Z** | Index finger extended pointing forward tracing a "Z" |

---

## How to Run It

```powershell
python main.py
```

### Interactive Word Speller Hotkeys
While the camera window is open:
- **`[SPACE]`**: Appends the currently recognized letter to the spelled word on the screen.
- **`[C]`**: Clears the spelled sentence.
- **`[BACKSPACE]`**: Deletes the last character.
- **`[Q]` or `[ESC]`**: Exits the application cleanly.

---

## Automated Verification

To run the automated test suite verifying normalization, debouncing, HUD rendering, and ASL classification:
```powershell
python tests/test_pipeline.py
```
Output:
```text
Running self-checks...
[PASS] Normalizer test passed
[PASS] Debouncer test passed
[PASS] Visualizer test passed
[PASS] Tracker test passed
[PASS] ASL Classifier test passed

All ASL pipeline tests completed successfully!
```
