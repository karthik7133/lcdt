import numpy as np

def calculate_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_ear(eye_landmarks):
    """Calculates the Eye Aspect Ratio (EAR)."""
    v1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])
    h = calculate_distance(eye_landmarks[0], eye_landmarks[3])
    return (v1 + v2) / (2.0 * h)

def calculate_mar(mouth_landmarks):
    """Calculates the Mouth Aspect Ratio (MAR)."""
    v = calculate_distance(mouth_landmarks[1], mouth_landmarks[2])
    h = calculate_distance(mouth_landmarks[0], mouth_landmarks[3])
    return v / h

def calculate_head_pitch(face_landmarks):
    """
    Approximates head pitch (up/down tilt).
    Uses the ratio of vertical distance (Nose to Chin) vs Horizontal distance.
    As the head tilts down, the vertical distance appears shorter.
    """
    # 1: Nose Tip, 152: Chin, 234: Left Cheek, 454: Right Cheek
    nose = face_landmarks.landmark[1]
    chin = face_landmarks.landmark[152]
    l_cheek = face_landmarks.landmark[234]
    r_cheek = face_landmarks.landmark[454]
    
    v_dist = abs(nose.y - chin.y)
    h_dist = abs(l_cheek.x - r_cheek.x)
    
    # Return a normalized ratio
    return v_dist / h_dist
