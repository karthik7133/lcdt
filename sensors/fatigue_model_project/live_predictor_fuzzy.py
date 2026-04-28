"""
live_predictor_fuzzy.py
========================
Drop-in replacement for live_predictor.py.

Replaces: SVM (fatigue_model_v2.pkl) + StandardScaler
With    : FuzzyFatigueEngine (fuzzy_engine.py) — no pkl needed for inference,
          but loads fuzzy_model.pkl if available for consistency.

Same exit codes as original:
  sys.exit(80)  → AI verified fatigue
  sys.exit(0)   → User is awake, camera shutdown
"""

import cv2
import numpy as np
from collections import deque
import sys
import os
from math_utils import calculate_ear, calculate_mar, calculate_head_pitch
from fuzzy_engine import FuzzyFatigueEngine

# ── Load Fuzzy Engine ─────────────────────────────────────────────────────
print("Loading Fuzzy Logic Fatigue Engine...")
try:
    import joblib
    engine = joblib.load("fuzzy_model.pkl")
    print("  Loaded pre-evaluated fuzzy_model.pkl")
except Exception:
    engine = FuzzyFatigueEngine()
    print("  Using fresh FuzzyFatigueEngine (fuzzy_model.pkl not found)")

# ── MediaPipe Setup ───────────────────────────────────────────────────────
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [78, 13, 14, 308]

# ── State Management ──────────────────────────────────────────────────────
feature_buffer     = deque(maxlen=10)
prediction_history = deque(maxlen=20)

blink_total   = 0
blink_cooldown = 0
awake_counter = 0
tired_counter = 0

# ── Open Webcam ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("--- FUZZY LOGIC LIVE TRACKING ACTIVE ---")
print("Tracking: EAR, MAR, Head Pitch, 10-frame EAR moving average")
print("Decision: 19 Expert Fuzzy Rules + Centroid Defuzzification")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results   = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape

            # Extract landmark coordinates
            l_eye_pts = [(face_landmarks.landmark[i].x * w,
                          face_landmarks.landmark[i].y * h) for i in LEFT_EYE]
            r_eye_pts = [(face_landmarks.landmark[i].x * w,
                          face_landmarks.landmark[i].y * h) for i in RIGHT_EYE]
            mouth_pts = [(face_landmarks.landmark[i].x * w,
                          face_landmarks.landmark[i].y * h) for i in MOUTH]

            # Compute biometric features
            avg_ear = (calculate_ear(l_eye_pts) + calculate_ear(r_eye_pts)) / 2.0
            mar     = calculate_mar(mouth_pts)
            pitch   = calculate_head_pitch(face_landmarks)

            # Blink detection
            if avg_ear < 0.18:
                if blink_cooldown == 0:
                    blink_total += 1
                    blink_cooldown = 5
            if blink_cooldown > 0:
                blink_cooldown -= 1

            # Build 10-frame EAR moving average
            feature_buffer.append((avg_ear, mar, pitch))

            if len(feature_buffer) == 10:
                ear_vals = [f[0] for f in feature_buffer]
                ear_ma   = float(np.mean(ear_vals))

                # ── FUZZY INFERENCE ──────────────────────────────────────
                result = engine.predict(
                    ear    = avg_ear,
                    mar    = mar,
                    pitch  = pitch,
                    ear_ma = ear_ma
                )

                raw_label    = result["label"]
                fatigue_idx  = result["fatigue_index"]
                str_alert    = result["strength_alert"]
                str_moderate = result["strength_moderate"]
                str_tired    = result["strength_tired"]

                prediction_history.append(raw_label)

                # Smoothed fatigue confidence across last 20 frames
                smoothed_fatigue = (sum(prediction_history) / len(prediction_history)) * 100

                # ── STATUS DECISION ──────────────────────────────────────
                if smoothed_fatigue > 50:
                    text  = "STATUS: TIRED"
                    color = (0, 0, 255)  # Red
                    awake_counter = 0
                    tired_counter += 1

                    if tired_counter > 30:
                        print(f"FUZZY AI VERIFIED FATIGUE (Index={fatigue_idx:.1f}). "
                              f"Returning risk code 80.")
                        cap.release()
                        cv2.destroyAllWindows()
                        sys.exit(80)
                else:
                    text  = "STATUS: AWAKE"
                    color = (0, 255, 0)  # Green
                    tired_counter = 0
                    awake_counter += 1

                    if awake_counter > 45:
                        print("Status Confirmed: AWAKE. Shutting down vision core.")
                        cap.release()
                        cv2.destroyAllWindows()
                        sys.exit(0)

                # ── HUD OVERLAY ──────────────────────────────────────────
                cv2.putText(image, text,
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                cv2.putText(image, f"Fatigue Index: {fatigue_idx:.1f}%",
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"EAR={avg_ear:.3f} MAR={mar:.3f} Pitch={pitch:.3f}",
                            (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(image, f"Rules: A={str_alert:.2f} M={str_moderate:.2f} T={str_tired:.2f}",
                            (30, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
                cv2.putText(image, f"Blinks: {blink_total} | EAR_MA: {ear_ma:.3f}",
                            (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("Fatigue Detection — Fuzzy Logic Engine", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
