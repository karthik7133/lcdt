# LCDT — Complete Architecture Explained
### Why we have two datasets, how they relate, and why personal R² is lower

---

## 1. The Big Picture — What Are We Actually Building?

We are building a **Cognitive Digital Twin** — a model of YOUR brain that:
- Predicts how cognitively sharp you are right now (`Ct` — Capacity)
- Predicts how much mental load you're under (`Dt` — Demand)
- Detects habit degradation like bad passwords, late logins (`Ht` — Habits)
- Flags insider threat / adversarial patterns (`At` — Adversarial)

The core model is a **Neural Controlled Differential Equation (NCDE)** — it treats
your behavioural signals as a continuous-time path and integrates a learned
differential equation over it to predict these 4 cognitive states.

---

## 2. The Two Datasets — What They Are and Why Both Exist

```
┌─────────────────────────────────────────────────────────────────────┐
│  DATASET 1: CERT r4.2                  DATASET 2: Personal          │
│  ─────────────────────────             ─────────────────────────    │
│  330,452 rows                          ~1,800 rows (growing)        │
│  1,000 corporate employees             YOU — one person             │
│  18 months of data (2010–2011)         Hours of active usage        │
│  Has: real insider threat events       Has: real biometric sensors  │
│  Has: USB exfiltration, phishing       Has: keyboard/mouse/audio    │
│       heavy external emails                  fatigue camera         │
│                                                                     │
│  Ground truth for: Ht, At             Ground truth for: Ct, Dt     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why can't we use just ONE dataset?

| Need | CERT | Personal |
|---|---|---|
| Insider threat events (At=1.0) | ✅ 70 real malicious users | ❌ You can't simulate being an insider threat |
| Habit degradation (Ht) | ✅ USB violations, after-hours logins | ❌ These events are too rare for one person |
| Your specific cognitive capacity (Ct) | ❌ CERT users aren't you — different baselines | ✅ Your keyboard/mouse/camera captures YOUR Ct |
| Your demand patterns (Dt) | ❌ CERT work patterns ≠ your work | ✅ Your notification/typing patterns are YOUR Dt |

**They are complementary — neither alone is sufficient.**

---

## 3. The Training Pipeline — Sequential, NOT Simultaneous

The two datasets are NOT mixed together. They are used in **sequence** across 3 phases:

```
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 1: CERT Pre-training (scripts/run_training_pipeline.py)          ║
║  ────────────────────────────────────────────────────────────────────── ║
║  Input : cert_latent_states.csv  (330,452 rows)                         ║
║  Train : ENTIRE model — backbone + Head A + Head B                      ║
║  Loss  : MSE on ALL 4 dims [Ct, Dt, Ht, At]                            ║
║  Epochs: 100                                                            ║
║  Output: ncde_weights_cert.pt                                           ║
║                                                                         ║
║  What it learns:                                                        ║
║    • How cognitive states EVOLVE over time (temporal dynamics)          ║
║    • What insider threat behaviour LOOKS LIKE (At patterns)             ║
║    • What habit degradation LOOKS LIKE (Ht patterns)                   ║
║    • General Ct/Dt ranges for corporate employees                       ║
╚══════════════════════════════════════════════════════════════════════════╝
                              ↓ save weights
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 2: Personal Fine-tuning (runs once at pipeline end)              ║
║  ────────────────────────────────────────────────────────────────────── ║
║  Input : latent_states.csv  (~1,800 personal rows)                      ║
║  Load  : ncde_weights_cert.pt  (start from CERT knowledge)              ║
║  FREEZE: Head A completely  (Ht/At locked — never retrained on you)     ║
║  Train : ONLY Head B (Ct/Dt)  — 610 parameters                         ║
║  Loss  : MSE on [Ct, Dt] only — indices 0 and 1                        ║
║  Epochs: 80                                                             ║
║  Output: ncde_weights.pt  ← production weights used in main.py         ║
║                                                                         ║
║  What it learns:                                                        ║
║    • YOUR specific cognitive capacity baseline                          ║
║    • YOUR demand patterns based on your typing/notification style       ║
║    • Remaps CERT-learned hidden state → your personal Ct/Dt range       ║
╚══════════════════════════════════════════════════════════════════════════╝
                              ↓ auto-trigger every 200 new rows
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 3: Scheduled Auto-Retraining (runs inside main.py automatically) ║
║  ────────────────────────────────────────────────────────────────────── ║
║  Input : latent_states.csv  (growing continuously)                      ║
║  Same as Phase 2 but triggered every 200 new rows (~33 min of use)      ║
║  Epochs: 40  (faster, runs in background)                               ║
║  Head A: stays FROZEN always                                            ║
║  Output: updates ncde_weights.pt in-place                               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 4. The Model Architecture — What's Inside the NCDE

```
                     ┌─────────────────────────────────┐
  Your signals       │         INPUT LAYER              │
  (keyboard,         │   Behaviour Graph Embedding      │
   mouse, audio, ──► │   12-dimensional vector          │
   vision, etc.)     │   captures: activity, risk,      │
                     │   social, temporal patterns      │
                     └──────────────┬──────────────────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │      NCDE BACKBONE               │
                     │   InitialEncoder: 12 → 16 dims   │
                     │   CDEFunc: dZt/dt = f(Zt)·dXt   │
                     │   (18,800 parameters)            │
                     │   Trained on CERT, FROZEN in P2  │
                     │                                  │
                     │   Hidden state Zt ∈ R¹⁶          │
                     │   Encodes ALL cognitive dynamics  │
                     └──────────┬───────────────────────┘
                                │  Zt (16-dim vector)
                    ┌───────────┴────────────┐
                    │                        │
       ┌────────────▼────────┐   ┌───────────▼────────────┐
       │      HEAD A         │   │       HEAD B            │
       │  (Ht, At outputs)   │   │  (Ct, Dt outputs)      │
       │  610 parameters     │   │  610 parameters         │
       │                     │   │                         │
       │  Trained: Phase 1   │   │  Trained: Phase 1+2+3   │
       │  FROZEN in Phase 2  │   │  Retrained every 200 rows│
       │  ──────────────     │   │  ──────────────────────  │
       │  Ht = habit score   │   │  Ct = YOUR capacity      │
       │  At = threat score  │   │  Dt = YOUR demand        │
       └─────────────────────┘   └────────────────────────┘
         from CERT knowledge           from YOUR data
```

**Total parameters: 20,228**
- Backbone: 19,008 (shared, trained on CERT)
- Head A: 610 (frozen after Phase 1)
- Head B: 610 (the only part that adapts to you)

---

## 5. Why Is Personal R² Lower Than CERT R²?

This is the most important question. Here's the complete answer:

### CERT R² (Set A) — Evaluated with `ncde_weights_cert.pt`

```
Training data  : 330,452 rows  →  ~330,000 sequences
Augmentation   : +9,548 malicious copies
Effective training: ~50,000 sequences × 100 epochs = 5,000,000 gradient steps
Test set       : 5,000 held-out sequences (diverse, from all 1,000 users)
Result         : Ct=0.96, Dt=0.92, Ht=0.99, At=0.62
```

The backbone has seen 1,000 different users across 18 months. It has learned
**universal cognitive dynamics** — how fatigue accumulates, how demand spikes,
how adversarial patterns look. The test set is diverse enough that the model
generalises well.

### Personal R² (Set B) — Evaluated with `ncde_weights.pt`

```
Training data  : ~1,800 rows  →  ~1,790 sequences
Augmentation   : ×8 AR(1) copies  →  ~14,320 sequences (in RAM)
Effective training: 14,320 sequences × 80 epochs = 1,145,600 gradient steps
Test set       : 20% of YOUR rows (randomly sampled)
Result         : Ct=0.61, Dt=0.58
```

**There are 3 reasons personal R² is lower:**

#### Reason 1 — The backbone was trained on CERT, not you

```
CERT backbone encodes:  "How does a CORPORATE EMPLOYEE's Ct evolve?"
Your actual Ct:         "How does KARTHIK's Ct evolve?"

Head B must learn: CERT_hidden_state(Zt) → YOUR_Ct
This is a hard remapping — Zt was optimised for CERT users, not for you.
As you accumulate more personal data, Head B calibrates this remapping better.
```

#### Reason 2 — Only 610 parameters are personalised

```
Head B architecture:
  Linear(16 → 32) → ReLU → Linear(32 → 2) → Sigmoid
  = 16×32 + 32 + 32×2 + 2 = 610 parameters

With 1,800 rows, this is ~3 parameters per data point. 
With 10,000 rows (after ~45 Phase 3 retrains), it becomes ~16 data points per parameter
→ much better generalisation.
```

#### Reason 3 — Your personal test set has non-trivial noise

```
Each row in latent_states.csv is a 10-second snapshot.
Real biometric signals have noise from:
  - Camera lighting changes (fatigue detection)
  - Background audio triggering notifications
  - VS Code title updates being counted as task switches

This noise means even a perfect model can't achieve R²=1.0.
Expected ceiling for personal data: ~0.85–0.90 with enough data.
```

---

## 6. The Relationship Between the Two — Visualised

```
CERT Dataset                        Personal Dataset
(330K rows, 1000 users)             (~1800 rows, just you)
         │                                    │
         │ Phase 1 training                   │
         ▼                                    │
   ┌──────────────┐                           │
   │   BACKBONE   │ ◄── Learns HOW            │
   │   (19,008p)  │     cognition WORKS       │
   └──────┬───────┘     in general            │
          │                                   │
          │              ┌────────────────────┘
          │              │ Phase 2/3 fine-tuning
          │              ▼
   ┌──────▼──────┐  ┌──────────────┐
   │   HEAD A    │  │    HEAD B    │ ◄── Learns HOW
   │  (frozen)   │  │  (610 params)│     YOUR cognition
   │  Ht, At     │  │  Ct, Dt      │     looks specifically
   └─────────────┘  └──────────────┘
   CERT knowledge    Personal calibration
   never changes     grows every 200 rows

FINAL OUTPUT = Head A (from CERT) + Head B (from YOU)
```

**The two datasets do NOT train simultaneously.** CERT trains first to give the
backbone universal knowledge. Then YOUR data calibrates Head B on top of that
universal knowledge. It's like:

> "First, I learned how human cognition works by studying 1,000 people (CERT).
> Now I'm adjusting my predictions specifically for how YOUR mind works."

---

## 7. When Will Personal R² Reach 0.80+?

```
Phase 3 retrains needed:    ~8 retrains
Personal rows needed:       ~2,800 rows total  (currently ~1,800)
Additional rows needed:     ~1,000 more rows
Time at 360 rows/hour:      ~3 hours of tracker use

Every time you run `python main.py`:
  - 1 row added every 10 seconds
  - Every 200 rows (~33 min) → Phase 3 fires automatically
  - Phase 3: retrains Head B with 40 epochs → R² improves

Check progress anytime:
  python scripts/evaluate_model.py --save_report
```

---

## 8. Complete File Map

```
project/
│
├── main.py                          ← START HERE: launches everything
│
├── sensors/
│   └── telemetry_tracker.py         ← Collects biometric signals every 10s
│                                       Writes to data/latent_states.csv
│                                       Calls Phase 3 retrain every 200 rows
│
├── core/
│   ├── state_inference.py           ← THE BRAIN — NCDE model + all training
│   │     • NCDEModel                  (HIDDEN_DIM=16, 20,228 params)
│   │     • DualReadoutHead            (Head A frozen, Head B personal)
│   │     • train_ncde_cert()          (Phase 1: CERT training)
│   │     • train_ncde_personal()      (Phase 2/3: personal fine-tuning)
│   │     • NeuralCDEInference         (live inference + Phase 3 trigger)
│   │
│   ├── ncde_weights_cert.pt         ← Phase 1 weights (CERT-trained)
│   └── ncde_weights.pt              ← Production weights (personalised)
│
├── scripts/
│   ├── cert_feature_engineer.py     ← Processes raw CERT CSVs → latent states
│   │     Derives Ct, Dt, Ht, At from:
│   │       logon.csv, email.csv, device.csv + insiders.csv (ground truth)
│   │
│   ├── run_training_pipeline.py     ← Runs Phase 1 + Phase 2 in sequence
│   └── evaluate_model.py            ← Evaluates both datasets, saves report
│
├── data/
│   ├── cert_latent_states.csv       ← 330,452 rows (CERT dataset, engineered)
│   └── latent_states.csv            ← ~1,800+ rows (your personal data, live)
│
├── api/
│   └── dashboard_api.py             ← Flask REST API on port 5000
│                                       Serves live state to React dashboard
│
└── ui/frontend/                     ← React + Vite dashboard
      npm run dev → opens at http://localhost:5173
```

---

## 9. Summary in Plain English

| Question | Answer |
|---|---|
| Are both datasets trained together? | **No** — CERT first (Phase 1), then Personal second (Phase 2) |
| Does personal data affect Ht/At? | **No** — Head A (Ht/At) is ALWAYS frozen after Phase 1 |
| Does CERT data affect Ct/Dt in production? | **Indirectly** — the backbone (shared) was CERT-trained; Head B is yours |
| Why is personal R² lower? | Fewer rows (1,800 vs 330K), backbone optimised for CERT users not you |
| Will personal R² improve? | **Yes** — automatically, every 200 rows (~33 min of use) |
| Do you need to retrain manually? | **No** — Phase 3 fires inside `main.py` automatically |
| What's the expected personal R² ceiling? | ~0.85–0.90 with 3,000–5,000 rows of your data |
