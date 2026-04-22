import cv2
import mediapipe as mp
import joblib
import pandas as pd
import numpy as np
from collections import deque
from math_utils import calculate_ear, calculate_mar, calculate_head_pitch

print("Loading Multi-Modal V2 AI Brain (100% Precision Capable)...")
# 1. Load the model pipeline
model = joblib.load('fatigue_model_v2.pkl')

# 2. Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 13, 14, 308]

# 3. State Management
# feature_buffer stores (EAR, MAR, Pitch) for rolling context
feature_buffer = deque(maxlen=10)
# prediction_history for stability
prediction_history = deque(maxlen=20)

blink_total = 0
blink_cooldown = 0

# 4. Open Webcam (Laptop Cam)
cap = cv2.VideoCapture(0)
print("--- V2 LIVE TRACKING ACTIVE ---")
print("Tracking: Eyes, Mouth, Head Slump, and Blinks simultaneously.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            
            # Extract coordinates
            l_eye_pts = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in LEFT_EYE]
            r_eye_pts = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in RIGHT_EYE]
            mouth_pts = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in MOUTH]

            # 1. Basic Math
            avg_ear = (calculate_ear(l_eye_pts) + calculate_ear(r_eye_pts)) / 2.0
            mar = calculate_mar(mouth_pts)
            pitch = calculate_head_pitch(face_landmarks)

            # 2. Blink Detection
            if avg_ear < 0.18:
                if blink_cooldown == 0:
                    blink_total += 1
                    blink_cooldown = 5
            if blink_cooldown > 0:
                blink_cooldown -= 1

            # 3. Temporal Feature Engineering (Need 10 frames of context)
            feature_buffer.append((avg_ear, mar, pitch))
            
            if len(feature_buffer) == 10:
                ear_vals = [f[0] for f in feature_buffer]
                mar_vals = [f[1] for f in feature_buffer]
                pitch_vals = [f[2] for f in feature_buffer]
                
                ear_ma = np.mean(ear_vals)
                mar_ma = np.mean(mar_vals)
                pitch_ma = np.mean(pitch_vals)
                ear_std = np.std(ear_vals)
                
                # 4. Ask the Brain (Must match X order: EAR, MAR, Pitch, Blink_Count, EAR_MA, MAR_MA, Pitch_MA, EAR_STD)
                current_state = pd.DataFrame(
                    [[avg_ear, mar, pitch, blink_total, ear_ma, mar_ma, pitch_ma, ear_std]], 
                    columns=['EAR', 'MAR', 'Pitch', 'Blink_Count', 'EAR_MA', 'MAR_MA', 'Pitch_MA', 'EAR_STD']
                )
                
                raw_prediction = model.predict(current_state)[0]
                prediction_history.append(raw_prediction)
                
                fatigue_index = (sum(prediction_history) / len(prediction_history)) * 100
                
                # --- AUTO-KILL LOGIC ---
                if 'awake_counter' not in locals():
                    awake_counter = 0
                if 'tired_counter' not in locals():
                    tired_counter = 0

                if fatigue_index > 50: 
                    text = "STATUS: TIRED"
                    color = (0, 0, 255) # Red
                    awake_counter = 0 # Reset the counter if a tired state is detected
                    tired_counter += 1
                    
                    if tired_counter > 30: # AI is confident user is fatigued
                        print("AI VERIFIED FATIGUE. Returning risk code 80 to Watchdog.")
                        cap.release()
                        cv2.destroyAllWindows()
                        import sys
                        sys.exit(80) 
                else:
                    text = "STATUS: AWAKE"
                    color = (0, 255, 0) # Green
                    awake_counter += 1 # Count how many frames you are awake
                    tired_counter = 0
                    
                    # If awake for ~45 frames (approx 1.5 seconds), kill the camera
                    if awake_counter > 45:
                        print("Status Confirmed: AWAKE. Shutting down vision core to save power.")
                        cap.release()
                        cv2.destroyAllWindows()
                        import sys
                        sys.exit(0) 
                # -----------------------------

                cv2.putText(image, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                cv2.putText(image, f"Fatigue Index: {fatigue_index:.1f}%", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Blinks: {blink_total} | Pitch: {pitch:.2f}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Fatigue Detection V2 - Multi-Modal AI', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
