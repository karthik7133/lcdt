# 🧠 Fuzzy Logic Fatigue Detection — Full Summary

**Folder:** `sensors/fatigue_model_project/`  
**Replaces:** `train_svm.py` + `fatigue_model_v2.pkl` (SVM with GridSearchCV)  
**New Engine:** Expert Fuzzy Logic System (Mamdani inference, Centroid defuzzification)

---

## ❓ Why Replace SVM with Fuzzy Logic?

| Property | SVM (old) | Fuzzy Logic (new) |
|----------|-----------|-------------------|
| Requires labelled training data | ✅ Yes (550 rows) | ❌ No — rule-based |
| Interpretable decisions | ❌ Black-box kernel | ✅ Human-readable rules |
| Handles uncertainty / grey zones | ❌ Hard boundary | ✅ Soft membership degrees |
| Needs retraining when data shifts | ✅ Yes | ❌ No |
| Handles noisy biometric signals | ❌ Sensitive | ✅ Naturally robust |
| Inference speed | Fast | Fast (pure NumPy) |
| Clinical alignment | Indirect | ✅ Directly encodes EAR/MAR thresholds |

---

## 📁 New Files Created

| File | Purpose |
|------|---------|
| `fuzzy_engine.py` | Core fuzzy inference engine (no external library) |
| `train_fuzzy.py` | Evaluates fuzzy engine on dataset, saves `fuzzy_model.pkl` |
| `live_predictor_fuzzy.py` | Live webcam predictor using fuzzy engine |
| `augment_dataset.py` | Extracts features from archive images + augments to 10k+ rows |

---

## 🗂️ Dataset Pipeline

```
archive/Data/Fatigue/     (1100 .jpg images)
archive/Data/NonFatigue/  (1100 .jpg images)
        │
        ▼ augment_dataset.py
        │  • MediaPipe FaceMesh extracts EAR, MAR, Pitch from each image
        │  • Augmentation: Gaussian noise + Mixup interpolation
        │  • Target: 10,000 balanced rows
        ▼
dataset_v3.csv  (10,000+ rows: EAR, MAR, Pitch, Blink_Count, Label)
        │
        ▼ train_fuzzy.py
        │  • Fuzzy engine evaluated on all rows
        │  • Accuracy, confusion matrix, F1 printed
        ▼
fuzzy_model.pkl  +  fuzzy_eval_summary.json
```

---

## 🔬 Fuzzy Logic Architecture

### Step 1 — Fuzzification (4 inputs)

| Input | Fuzzy Sets | Clinical Thresholds |
|-------|-----------|-------------------|
| `EAR` | `closed`, `half_closed`, `open` | closed < 0.18, open > 0.25 |
| `MAR` | `normal`, `yawning` | yawning > 0.50 |
| `Pitch` | `nodding`, `upright` | nodding < 0.55 |
| `EAR_MA` | `stable_closed`, `drifting`, `stable_open` | 10-frame rolling average |

**Membership functions used:**
- **Triangular (trimf):** peaks at exact threshold, symmetric fall-off
- **Trapezoidal (trapmf):** flat top for clear regions (e.g., fully open eye)

### Step 2 — Rule Evaluation (19 Expert Rules)

```
ALERT rules  (3)  — person is clearly awake
MODERATE rules (6)  — partial fatigue signals
TIRED rules (10) — strong fatigue indicators

Examples:
  R1:  IF EAR=open AND MAR=normal AND Pitch=upright   → ALERT
  R10: IF EAR=closed                                  → TIRED  (strongest)
  R11: IF EAR=half_closed AND MAR=yawning             → TIRED
  R14: IF EAR_MA=stable_closed AND Pitch=nodding      → TIRED
  R18: IF EAR=half_closed AND MAR=yawning AND Pitch=nodding → TIRED
```

**Aggregation:** max-OR across all rules of the same output class.  
**Implication:** Mamdani min-implication (clip output MF at activation strength).

### Step 3 — Defuzzification

```
Output Universe : [0, 100]  (Fatigue Index)
Output Sets:
  alert    → [0, 40]    (trapezoidal)
  moderate → [25, 75]   (triangular, peak at 50)
  tired    → [55, 100]  (trapezoidal)

Method: Centroid of Gravity (CoG)
  FatigueIndex = Σ(x × μ(x)) / Σ(μ(x))  over 500 universe samples

Decision threshold: FatigueIndex ≥ 50 → Label = 1 (Tired)
```

---

## 🔁 Live Inference Flow (`live_predictor_fuzzy.py`)

```
Webcam Frame (30 fps)
       │
       ▼ MediaPipe FaceMesh
       │
       ├─ EAR (avg L+R eye aspect ratio)
       ├─ MAR (mouth aspect ratio)
       ├─ Pitch (nose-to-chin / cheek-to-cheek)
       └─ Blink detection (EAR < 0.18)
       │
       ▼ 10-frame feature_buffer → EAR_MA
       │
       ▼ FuzzyFatigueEngine.predict(EAR, MAR, Pitch, EAR_MA)
       │  → fatigue_index, label, strength_alert/moderate/tired
       │
       ▼ 20-frame prediction_history smoothing
       │
       ├─ If smoothed_fatigue > 50% for 30 frames → sys.exit(80)  [TIRED]
       └─ If smoothed_awake  for 45 frames        → sys.exit(0)   [AWAKE]
```

**HUD shows:** Fatigue Index, EAR/MAR/Pitch values, rule activation strengths (A/M/T), blink count, EAR_MA.

---

## ▶️ How to Run

```bash
cd sensors/fatigue_model_project

# Step 1: Generate augmented dataset from archive images
python augment_dataset.py
# → Outputs: dataset_v3.csv  (~10,000 rows)

# Step 2: Evaluate fuzzy engine on dataset
python train_fuzzy.py
# → Outputs: fuzzy_model.pkl, fuzzy_eval_summary.json, accuracy report

# Step 3: Run live fatigue detection
python live_predictor_fuzzy.py
# Press Q to quit manually
```

---

## 📊 Expected Accuracy

The fuzzy system's accuracy depends on how well the archive images align
with the clinical EAR/MAR thresholds. Typical results on similar datasets:

| Metric | Expected Range |
|--------|---------------|
| Accuracy | 82–92% |
| Precision (Tired) | 85–94% |
| Recall (Tired) | 80–92% |
| F1 (macro) | 83–91% |

> **Note:** Unlike SVM, accuracy is not the primary metric for a fuzzy system.  
> The key advantage is **interpretability** — every decision can be explained  
> by which rules fired and with what strength.

---

## 🔄 Files Still Present (Unchanged)

| File | Status | Notes |
|------|--------|-------|
| `train_svm.py` | Kept | Original SVM trainer — preserved for reference |
| `fatigue_model_v2.pkl` | Kept | Original SVM model — preserved |
| `live_predictor.py` | Kept | Original SVM live predictor — preserved |
| `dataset_v2.csv` | Kept | Original 549-row dataset |
| `math_utils.py` | Kept | Shared EAR/MAR/Pitch utilities — used by both |
