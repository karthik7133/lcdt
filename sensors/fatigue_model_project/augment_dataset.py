"""
augment_dataset.py
==================
Step 1: Extract EAR, MAR, Pitch, Blink_Count features from archive images
        using MediaPipe FaceMesh (same as live_predictor.py).
Step 2: Apply data augmentation (Gaussian noise + interpolation) to reach 10,000+ samples.
Step 3: Save the final dataset as dataset_v3.csv.

Usage:
    python augment_dataset.py

Output:
    dataset_v3.csv  (min 10,000 rows, columns: EAR, MAR, Pitch, Blink_Count, Label)
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from math_utils import calculate_ear, calculate_mar, calculate_head_pitch

# ── Config ────────────────────────────────────────────────────────────────────
FATIGUE_DIR    = os.path.join("archive", "Data", "Fatigue")
NON_FATIGUE_DIR = os.path.join("archive", "Data", "NonFatigue")
OUTPUT_CSV      = "dataset_v3.csv"
TARGET_SAMPLES  = 10000     # minimum rows in final dataset
AUG_NOISE_STD   = 0.008     # Gaussian noise std for augmentation
RANDOM_SEED     = 42

np.random.seed(RANDOM_SEED)

# ── MediaPipe Setup ───────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.3
)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [78, 13, 14, 308]


def extract_features_from_image(img_path):
    """
    Runs MediaPipe on one image and returns (EAR, MAR, Pitch) or None if face not found.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res  = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0]

    l_eye_pts = [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in LEFT_EYE]
    r_eye_pts = [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in RIGHT_EYE]
    mouth_pts = [(lm.landmark[i].x * w, lm.landmark[i].y * h) for i in MOUTH]

    ear   = (calculate_ear(l_eye_pts) + calculate_ear(r_eye_pts)) / 2.0
    mar   = calculate_mar(mouth_pts)
    pitch = calculate_head_pitch(lm)

    return float(ear), float(mar), float(pitch)


def process_directory(directory, label):
    """Extract features from all images in a folder."""
    records = []
    files   = sorted([f for f in os.listdir(directory)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total   = len(files)
    print(f"\n  Processing {total} images from: {directory}  (label={label})")

    for i, fname in enumerate(files):
        path = os.path.join(directory, fname)
        feat = extract_features_from_image(path)
        if feat is not None:
            ear, mar, pitch = feat
            records.append({
                "EAR":         ear,
                "MAR":         mar,
                "Pitch":       pitch,
                "Blink_Count": 0,        # static image — no blink count
                "Label":       label
            })
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{total}] extracted so far: {len(records)} valid faces")

    print(f"  Done. Valid faces extracted: {len(records)} / {total}")
    return records


def augment_records(records, target_total):
    """
    Augment by:
      1. Adding Gaussian noise to EAR, MAR, Pitch
      2. Linear interpolation between random pairs (mixup)
    Returns a DataFrame with target_total rows.
    """
    df_orig = pd.DataFrame(records)
    n_orig  = len(df_orig)
    needed  = target_total - n_orig

    if needed <= 0:
        print(f"\n  Original data ({n_orig} rows) already meets target. No augmentation needed.")
        return df_orig

    print(f"\n  Augmenting: need {needed} more rows (original: {n_orig}, target: {target_total})")

    aug_rows = []
    for _ in range(needed):
        # Pick a random base row
        base = df_orig.sample(1).iloc[0]

        # Method: 50% noise-only, 50% mixup with another sample
        if np.random.rand() < 0.5:
            # Gaussian noise augmentation
            new_ear   = base["EAR"]   + np.random.normal(0, AUG_NOISE_STD)
            new_mar   = base["MAR"]   + np.random.normal(0, AUG_NOISE_STD)
            new_pitch = base["Pitch"] + np.random.normal(0, AUG_NOISE_STD * 2)
            new_label = base["Label"]
        else:
            # Mixup: interpolate between two samples of the same class
            same_class = df_orig[df_orig["Label"] == base["Label"]]
            other = same_class.sample(1).iloc[0]
            alpha = np.random.uniform(0.2, 0.8)
            new_ear   = alpha * base["EAR"]   + (1 - alpha) * other["EAR"]
            new_mar   = alpha * base["MAR"]   + (1 - alpha) * other["MAR"]
            new_pitch = alpha * base["Pitch"] + (1 - alpha) * other["Pitch"]
            new_label = base["Label"]

        # Clip to valid ranges
        new_ear   = float(np.clip(new_ear,   0.05, 0.50))
        new_mar   = float(np.clip(new_mar,   0.00, 1.00))
        new_pitch = float(np.clip(new_pitch, 0.30, 1.20))

        aug_rows.append({
            "EAR":         new_ear,
            "MAR":         new_mar,
            "Pitch":       new_pitch,
            "Blink_Count": int(np.random.randint(0, 8)),
            "Label":       int(new_label)
        })

    df_aug = pd.DataFrame(aug_rows)
    df_final = pd.concat([df_orig, df_aug], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  Augmentation complete. Final dataset: {len(df_final)} rows")
    return df_final


# ── Main Pipeline ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  LCDT — Fatigue Dataset Augmentation Pipeline")
    print("=" * 60)

    # 1. Extract from archive images
    fatigue_records     = process_directory(FATIGUE_DIR,     label=1)
    non_fatigue_records = process_directory(NON_FATIGUE_DIR, label=0)

    all_records = fatigue_records + non_fatigue_records
    print(f"\nTotal raw extracted: {len(all_records)} "
          f"(Fatigue={len(fatigue_records)}, NonFatigue={len(non_fatigue_records)})")

    if len(all_records) < 10:
        print("ERROR: Too few faces detected. Check MediaPipe installation.")
        exit(1)

    # 2. Augment to 10k+
    # Split target proportionally between classes
    f_ratio  = len(fatigue_records) / max(1, len(all_records))
    nf_ratio = 1.0 - f_ratio
    target_f  = int(TARGET_SAMPLES * f_ratio)
    target_nf = TARGET_SAMPLES - target_f

    df_f  = augment_records(fatigue_records,     target_total=target_f)
    df_nf = augment_records(non_fatigue_records, target_total=target_nf)

    df_final = pd.concat([df_f, df_nf], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # 3. Save
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'='*60}")
    print(f"  Saved: {OUTPUT_CSV}")
    print(f"  Total rows: {len(df_final)}")
    print(f"  Fatigue (1):    {(df_final['Label']==1).sum()}")
    print(f"  NonFatigue (0): {(df_final['Label']==0).sum()}")
    print(f"  Columns: {list(df_final.columns)}")
    print(f"{'='*60}")

    face_mesh.close()
