import cv2
import mediapipe as mp
import csv
import os
import time
from math_utils import calculate_ear, calculate_mar, calculate_head_pitch

# Set up the NEW CSV file (v2)
csv_file = 'dataset_v2.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['EAR', 'MAR', 'Pitch', 'Blink_Count', 'Label']) 

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 13, 14, 308]

# Blink counting state
blink_total = 0
blink_cooldown = 0

# Open Webcam (Laptop Cam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not find camera.")
    exit()

print("--- DATA COLLECTOR V2 (MULTI-MODAL) STARTED ---")
print("Press '0' to record an AWAKE frame.")
print("Press '1' to record a TIRED/YAWNING/SLUMPING frame.")
print("Press 'r' to Reset Blink Count.")
print("Press 'q' to Quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            
            # Extract basic coordinates
            l_eye_pts = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in LEFT_EYE]
            r_eye_pts = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in RIGHT_EYE]
            mouth_pts = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in MOUTH]

            # 1. EAR and MAR
            l_ear = calculate_ear(l_eye_pts)
            r_ear = calculate_ear(r_eye_pts)
            avg_ear = (l_ear + r_ear) / 2.0
            mar = calculate_mar(mouth_pts)

            # 2. Head Pitch
            pitch = calculate_head_pitch(face_landmarks)

            # 3. Blink Detection (Live)
            # Threshold 0.18 is standard for a blink
            if avg_ear < 0.18:
                if blink_cooldown == 0:
                    blink_total += 1
                    blink_cooldown = 5 # Skip next few frames
            if blink_cooldown > 0:
                blink_cooldown -= 1

            # HUD Display
            cv2.putText(image, f"EAR: {avg_ear:.2f} | MAR: {mar:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Pitch: {pitch:.2f} | Blinks: {blink_total}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Multi-Modal Data Collector', image)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('0') or key == ord('1'):
        if results.multi_face_landmarks:
            label = 0 if key == ord('0') else 1
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([avg_ear, mar, pitch, blink_total, label])
            
            status = "AWAKE" if label == 0 else "TIRED"
            print(f"Recorded -> {status} | EAR: {avg_ear:.2f}, Pitch: {pitch:.2f}, Blinks: {blink_total}")
        else:
            print("No face detected.")

    elif key == ord('r'):
        blink_total = 0
        print("Blink count reset.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
