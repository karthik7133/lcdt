"""
test.py
========
Continuous Fuzzy Logic Fatigue Tester.

Unlike live_predictor_fuzzy.py (which auto-exits after confirming awake/tired),
this runs FOREVER until you press Q. It shows a live dashboard with:

  - Real-time EAR, MAR, Pitch values
  - All fuzzy membership degrees (how much each rule fires)
  - Fatigue Index bar (0–100)
  - Rolling 20-frame average fatigue %
  - Live status: AWAKE / DROWSY / TIRED
  - Session stats: frames processed, blink count, avg fatigue over session

Press Q to quit.
"""

import cv2
import numpy as np
from collections import deque
import time
from math_utils import calculate_ear, calculate_mar, calculate_head_pitch
from fuzzy_engine import FuzzyFatigueEngine

# ── Load Engine ───────────────────────────────────────────────────────────────
print("Loading Fuzzy Logic Engine...")
try:
    import joblib
    engine = joblib.load("fuzzy_model.pkl")
    print("  Loaded fuzzy_model.pkl")
except Exception:
    engine = FuzzyFatigueEngine()
    print("  Using fresh FuzzyFatigueEngine")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [78, 13, 14, 308]

# ── State ─────────────────────────────────────────────────────────────────────
feature_buffer     = deque(maxlen=10)
prediction_history = deque(maxlen=20)
session_fatigue    = []          # for session-level average

blink_total    = 0
blink_cooldown = 0
frame_count    = 0
start_time     = time.time()

# ── Drawing helpers ───────────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX
BG_COLOR   = (18, 18, 18)
PANEL_W    = 380          # right-side info panel width

def draw_bar(img, x, y, w, h, value, max_val=100,
             low_color=(60,200,60), high_color=(0,60,220), label=""):
    """Draws a horizontal progress bar. Color shifts from green → red."""
    ratio = min(1.0, max(0.0, value / max_val))
    # Background
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    # Filled portion — interpolate color
    fill_color = (
        int(low_color[0] + (high_color[0] - low_color[0]) * ratio),
        int(low_color[1] + (high_color[1] - low_color[1]) * ratio),
        int(low_color[2] + (high_color[2] - low_color[2]) * ratio),
    )
    cv2.rectangle(img, (x, y), (x + int(w * ratio), y + h), fill_color, -1)
    # Border
    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)
    # Value text
    cv2.putText(img, f"{value:.1f}", (x + w + 6, y + h - 2),
                FONT, 0.45, (220, 220, 220), 1)
    if label:
        cv2.putText(img, label, (x, y - 5), FONT, 0.42, (180, 180, 180), 1)


def draw_panel(frame, result, ear, mar, pitch, ear_ma,
               blinks, frames, elapsed, session_avg):
    """
    Draws a dark side panel on the right with all fuzzy metrics.
    """
    h, w = frame.shape[:2]
    panel_x = w - PANEL_W

    # Dark panel background
    cv2.rectangle(frame, (panel_x, 0), (w, h), (22, 22, 28), -1)
    cv2.line(frame, (panel_x, 0), (panel_x, h), (60, 60, 80), 2)

    fi    = result["fatigue_index"]
    label = result["label"]
    sa    = result["strength_alert"]
    sm    = result["strength_moderate"]
    st    = result["strength_tired"]

    # ── Title ─────────────────────────────────────────────────────────────
    cv2.putText(frame, "FUZZY FATIGUE TESTER", (panel_x + 10, 30),
                FONT_BOLD, 0.55, (100, 200, 255), 1)
    cv2.line(frame, (panel_x + 10, 38), (w - 10, 38), (60, 60, 80), 1)

    # ── Status Badge ─────────────────────────────────────────────────────
    if fi < 35:
        status_text  = "AWAKE"
        status_color = (40, 220, 80)
    elif fi < 60:
        status_text  = "DROWSY"
        status_color = (0, 200, 255)
    else:
        status_text  = "TIRED"
        status_color = (50, 60, 255)

    cv2.rectangle(frame, (panel_x + 10, 48), (w - 10, 90), status_color, -1)
    text_sz = cv2.getTextSize(status_text, FONT_BOLD, 1.0, 2)[0]
    tx = panel_x + 10 + (PANEL_W - 20 - text_sz[0]) // 2
    cv2.putText(frame, status_text, (tx, 80), FONT_BOLD, 1.0, (255, 255, 255), 2)

    # ── Fatigue Index Bar ──────────────────────────────────────────────────
    cv2.putText(frame, "Fatigue Index", (panel_x + 10, 110),
                FONT, 0.48, (200, 200, 200), 1)
    draw_bar(frame, panel_x + 10, 118, PANEL_W - 60, 18, fi,
             low_color=(40, 200, 40), high_color=(30, 30, 230))

    # ── Rolling Average Bar ───────────────────────────────────────────────
    roll_avg = (sum(prediction_history) / max(1, len(prediction_history))) * 100
    cv2.putText(frame, "Smoothed (20-frame avg)", (panel_x + 10, 158),
                FONT, 0.45, (180, 180, 180), 1)
    draw_bar(frame, panel_x + 10, 165, PANEL_W - 60, 14, roll_avg,
             low_color=(40, 180, 40), high_color=(20, 20, 200))

    # ── Rule Activation Bars ──────────────────────────────────────────────
    cv2.putText(frame, "Rule Activations", (panel_x + 10, 200),
                FONT, 0.48, (200, 200, 200), 1)
    cv2.line(frame, (panel_x + 10, 205), (w - 10, 205), (50, 50, 60), 1)

    draw_bar(frame, panel_x + 10, 215, PANEL_W - 80, 13, sa * 100,
             low_color=(40, 200, 40), high_color=(40, 200, 40), label="Alert   ")
    draw_bar(frame, panel_x + 10, 238, PANEL_W - 80, 13, sm * 100,
             low_color=(0, 200, 200), high_color=(0, 200, 200), label="Moderate")
    draw_bar(frame, panel_x + 10, 261, PANEL_W - 80, 13, st * 100,
             low_color=(30, 30, 220), high_color=(30, 30, 220), label="Tired   ")

    # ── Raw Biometric Values ─────────────────────────────────────────────
    cv2.putText(frame, "Biometrics", (panel_x + 10, 300),
                FONT, 0.48, (200, 200, 200), 1)
    cv2.line(frame, (panel_x + 10, 305), (w - 10, 305), (50, 50, 60), 1)

    draw_bar(frame, panel_x + 10, 315, PANEL_W - 80, 13, ear * 400,
             low_color=(40, 200, 40), high_color=(30, 30, 220), label="EAR     ")
    draw_bar(frame, panel_x + 10, 338, PANEL_W - 80, 13, mar * 200,
             low_color=(40, 200, 40), high_color=(30, 30, 220), label="MAR     ")
    draw_bar(frame, panel_x + 10, 361, PANEL_W - 80, 13, pitch * 100,
             low_color=(30, 30, 220), high_color=(40, 200, 40), label="Pitch   ")
    draw_bar(frame, panel_x + 10, 384, PANEL_W - 80, 13, ear_ma * 400,
             low_color=(40, 200, 40), high_color=(30, 30, 220), label="EAR_MA  ")

    # ── Numeric readouts ──────────────────────────────────────────────────
    cv2.putText(frame, f"EAR={ear:.3f}  MAR={mar:.3f}",
                (panel_x + 10, 415), FONT, 0.44, (160, 200, 160), 1)
    cv2.putText(frame, f"Pitch={pitch:.3f}  EAR_MA={ear_ma:.3f}",
                (panel_x + 10, 432), FONT, 0.44, (160, 200, 160), 1)

    # ── Session Stats ─────────────────────────────────────────────────────
    cv2.putText(frame, "Session Stats", (panel_x + 10, 465),
                FONT, 0.48, (200, 200, 200), 1)
    cv2.line(frame, (panel_x + 10, 470), (w - 10, 470), (50, 50, 60), 1)

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    cv2.putText(frame, f"Time      : {mins:02d}:{secs:02d}",
                (panel_x + 10, 490), FONT, 0.44, (180, 180, 180), 1)
    cv2.putText(frame, f"Frames    : {frames}",
                (panel_x + 10, 507), FONT, 0.44, (180, 180, 180), 1)
    cv2.putText(frame, f"Blinks    : {blinks}",
                (panel_x + 10, 524), FONT, 0.44, (180, 180, 180), 1)
    cv2.putText(frame, f"Sess.Avg  : {session_avg:.1f}%",
                (panel_x + 10, 541), FONT, 0.44, (180, 180, 180), 1)

    # ── Quit hint ─────────────────────────────────────────────────────────
    cv2.putText(frame, "Press Q to quit", (panel_x + 10, h - 15),
                FONT, 0.42, (100, 100, 100), 1)


# ── Main Loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit(1)

# Make window wider to fit panel
ret, test_frame = cap.read()
if ret:
    fh, fw = test_frame.shape[:2]
    cv2.namedWindow("Fuzzy Fatigue Tester", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fuzzy Fatigue Tester", fw + PANEL_W, fh)

print("Fuzzy Fatigue Tester running. Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    elapsed = time.time() - start_time

    # Extend frame to the right for the info panel
    panel = np.zeros((frame.shape[0], PANEL_W, 3), dtype=np.uint8)
    canvas = np.hstack([frame, panel])

    # Default values if no face found
    ear, mar, pitch, ear_ma = 0.30, 0.05, 0.70, 0.30
    result = {"fatigue_index": 50.0, "label": 0,
              "strength_alert": 0.5, "strength_moderate": 0.0, "strength_tired": 0.0}

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        l_eye_pts = [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in LEFT_EYE]
        r_eye_pts = [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in RIGHT_EYE]
        mouth_pts = [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in MOUTH]

        ear   = (calculate_ear(l_eye_pts) + calculate_ear(r_eye_pts)) / 2.0
        mar   = calculate_mar(mouth_pts)
        pitch = calculate_head_pitch(lm)

        # Blink
        if ear < 0.18:
            if blink_cooldown == 0:
                blink_total += 1
                blink_cooldown = 5
        if blink_cooldown > 0:
            blink_cooldown -= 1

        # EAR moving average
        feature_buffer.append((ear, mar, pitch))
        ear_ma = float(np.mean([f[0] for f in feature_buffer]))

        # Fuzzy inference
        result = engine.predict(ear=ear, mar=mar, pitch=pitch, ear_ma=ear_ma)
        prediction_history.append(result["label"])
        session_fatigue.append(result["fatigue_index"])

        # Draw landmarks on camera side (minimal — just eye boxes)
        for idx in LEFT_EYE + RIGHT_EYE:
            px = int(lm.landmark[idx].x * w)
            py = int(lm.landmark[idx].y * h)
            cv2.circle(canvas, (px, py), 1, (0, 255, 150), -1)

    sess_avg = float(np.mean(session_fatigue)) if session_fatigue else 50.0

    # Draw info panel onto canvas right side
    draw_panel(canvas, result, ear, mar, pitch, ear_ma,
               blink_total, frame_count, elapsed, sess_avg)

    # FPS overlay on camera side
    fps = frame_count / max(1, elapsed)
    cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 25),
                FONT, 0.55, (150, 150, 150), 1)

    cv2.imshow("Fuzzy Fatigue Tester", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ── Session Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  SESSION SUMMARY")
print("=" * 50)
elapsed = time.time() - start_time
mins, secs = int(elapsed // 60), int(elapsed % 60)
print(f"  Duration    : {mins:02d}:{secs:02d}")
print(f"  Frames      : {frame_count}")
print(f"  Blinks      : {blink_total}")
if session_fatigue:
    print(f"  Avg Fatigue : {np.mean(session_fatigue):.1f}%")
    print(f"  Max Fatigue : {np.max(session_fatigue):.1f}%")
    print(f"  Min Fatigue : {np.min(session_fatigue):.1f}%")
    tired_pct = (sum(1 for x in session_fatigue if x >= 50) / len(session_fatigue)) * 100
    print(f"  Time Tired  : {tired_pct:.1f}% of session")
print("=" * 50)
