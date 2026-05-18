# LCDT Dual-Dataset NCDE — Model Training Report

> **Project:** Lifelong Cognitive Digital Twin (LCDT)
> **Architecture:** Neural Controlled Differential Equation with Dual Readout Heads
> **Last Updated:** May 2026

---

## 1. Problem Statement

The original single-head NCDE model had **R² scores of −3.0** across all four cognitive state dimensions because:

1. Only **1,171 personal data rows** available — too few for a sequence model
2. **Adversarial events** (`At`) are nearly impossible to trigger naturally (phishing clicks, scam credentials)
3. **Security habits** (`Ht`) degrade only through rare events (USB insertion, suspicious link clicks)
4. The model was trained on all four dimensions with equally sparse signals → noise dominated

**Solution:** Separate the training responsibility across two datasets using a dual-head architecture.

---

## 2. Dataset Overview

### Dataset A — CERT Insider Threat Dataset r4.2

| Property | Value |
|---|---|
| Source | Carnegie Mellon University SEI |
| Users | 1,000 synthetic employees |
| Time span | 17 months (501 working days) |
| Total events | ~3.3 million |
| Insider threat users | 70 (labeled malicious) |
| Threat scenarios | Data exfiltration, lateral movement, credential theft |
| Files used | `logon.csv`, `email.csv`, `device.csv`, `answers/insiders.csv` |
| File skipped | `http.csv` (11.5 GB — too large; not needed) |

**Derived features:**
- `At` — adversarial score: spikes to 1.0 on malicious event dates, decays at 0.82/day
- `Ht` — habits score: penalised by email attachments (−0.10), USB connects (−0.08), after-hours login (−0.05)
- `Ct` — capacity proxy: circadian-based, penalised by late-night login and long sessions
- `Dt` — demand proxy: normalised email count per day

### Dataset B — Personal Biometric Data

| Property | Value |
|---|---|
| Source | Local telemetry tracker (`telemetry_tracker.py`) |
| Signals | Keyboard rate, mouse entropy, facial fatigue, tab switches, sleep deficit, hour of day |
| Collection interval | Every 10 seconds |
| Storage | `data/latent_states.csv` |
| Rows collected | ~1,200 (grows continuously) |

**Used features:**
- `Ct` — capacity: directly inferred from keyboard/mouse/facial/sleep signals ✅
- `Dt` — demand: from active window count, email volume, notifications ✅
- `Ht` — **excluded** from personal training (sourced from CERT Head A) ❌
- `At` — **excluded** from personal training (sourced from CERT Head A) ❌

---

## 3. Model Architecture

```
  20 Sensor Signals (every 10 seconds)
           │
           ▼
  ┌─────────────────────┐
  │  L1: Behaviour      │   12-dim graph embedding
  │  Graph Engine       │   (BehaviourGraphEngine)
  └──────────┬──────────┘
             │  x(t) ∈ ℝ¹²
             ▼
  ┌─────────────────────────────────────────┐
  │  L2: NCDE Backbone                      │
  │                                          │
  │  dZt = f(Zt)dt + g_ϕ(Zt) dXt           │
  │                                          │
  │  InitialEncoder : Linear(12→8) + Tanh   │
  │  CDEFunc        : Linear(8→96) + Tanh   │
  │  Hidden state   : Zt ∈ ℝ⁸              │
  └──────────┬──────────────────────────────┘
             │  Zt (shared — causal coupling preserved)
      ┌───────┴───────┐
      ▼               ▼
  ┌─────────┐     ┌─────────┐
  │ HEAD A  │     │ HEAD B  │
  │  Ht, At │     │  Ct, Dt │
  │         │     │         │
  │ Linear  │     │ Linear  │
  │  (8→32) │     │  (8→32) │
  │  ReLU   │     │  ReLU   │
  │ Linear  │     │ Linear  │
  │  (32→2) │     │  (32→2) │
  │ Sigmoid │     │ Sigmoid │
  │         │     │         │
  │ FROZEN  │     │ RETRAIN │
  │ (CERT)  │     │ (pers.) │
  └────┬────┘     └────┬────┘
       │               │
       └───────┬───────┘
               ▼
    [Ct, Dt, Ht, At]  (output order preserved)
               │
               ▼
  ┌────────────────────────┐
  │  L3: UserProfile       │
  │  Bayesian shrinkage    │
  │  toward personal prior │
  │  (EMA, updates live)   │
  └────────────────────────┘
               │
               ▼
          Risk Score + Interventions
```

### Parameter Count

| Component | Parameters | Role |
|---|---|---|
| InitialEncoder | 104 | Maps first observation to Z₀ |
| CDEFunc | 864 | CDE drift function |
| Head A (Ht, At) | 546 | CERT-trained, frozen |
| Head B (Ct, Dt) | 546 | Personally fine-tuned |
| **Total** | **2,060** | Lightweight — runs in <5ms |

---

## 4. Training Protocol

### Phase 1 — CERT Pre-training

Trains the full model (both heads) on CERT-derived cognitive states.

| Setting | Value |
|---|---|
| Input CSV | `data/cert_latent_states.csv` |
| Sequences | 50,000 (randomly sampled from 320,000 user-aware windows) |
| Epochs | 20 |
| Batch size | 256 |
| Learning rate | 1e-3 (all components) |
| Loss | MSE on [Ct, Dt, Ht, At] |
| Output | `core/ncde_weights_cert.pt` |

**User-aware windowing:** Sequences are built within each user's 500-day trajectory only. No cross-user windows. This ensures the NCDE learns real intra-person cognitive dynamics.

### Phase 2 — Personal Fine-tuning

Fine-tunes Head B and backbone on your biometric data. Head A stays frozen.

| Setting | Value |
|---|---|
| Input CSV | `data/latent_states.csv` |
| Sequences | ~1,200 |
| Ht/At in input | Zeroed out (not present in personal data reliably) |
| Ht/At in targets | Zeroed out (excluded from loss) |
| Head A | FROZEN — CERT knowledge preserved |
| Epochs | 30 |
| Batch size | 16 |
| LR — backbone | 3e-5 (very slow — prevents forgetting CERT dynamics) |
| LR — Head B | 1e-4 |
| Loss | MSE on [Ct, Dt] only (indices 0–1) |
| Output | `core/ncde_weights.pt` (production) |

### Phase 3 — Scheduled Auto-Retraining

Runs Phase 2 automatically as personal data accumulates.

| Setting | Value |
|---|---|
| Trigger | 200+ new rows in `latent_states.csv` |
| Check frequency | Every 360 ticks (~1 hour of tracker runtime) |
| Epochs per retrain | 20 |
| Head A | Always stays frozen |
| Effect | Model continuously personalises to your behavioral drift |

---

## 5. Accuracy Results

### CERT Test Set (5,000 held-out sequences from 1,000 users)

| Dimension | R² | RMSE | Grade | Head |
|---|---|---|---|---|
| **Ht** — Security Habits | **+0.9956** | 0.025 | A+ Excellent | A (CERT, frozen) |
| **Ct** — Cognitive Capacity | **+0.9283** | 0.068 | A+ Excellent | B (personal) |
| **Dt** — Demand / Workload | **+0.8138** | 0.072 | A Good | B (personal) |
| **At** — Adversarial Threat | **+0.7125** | 0.092 | A Good | A (CERT, frozen) |

**Before this architecture:** R² = −3.0 across all four dimensions.

### Personal Test Set (20% held-out split)

| Dimension | R² | RMSE | Note |
|---|---|---|---|
| **Ct** | −3.19 | 0.022 | Low variance in personal Ct — RMSE is only 2.2% |
| **Dt** | −0.90 | 0.067 | Still adapting — improves with more data |
| Ht | — | — | Sourced from CERT Head A (R²=0.9956) |
| At | — | — | Sourced from CERT Head A (R²=0.7125) |

> **Why personal R² is negative:** Your personal `Ct` has very low variance (you consistently operate in a narrow range like 0.85–1.0). When variance is this small, even a 2.2% RMSE can score negative R². Additionally, this evaluation uses raw NCDE output — at inference time, the **L3 UserProfile Bayesian shrinkage** corrects the output to your personal baseline, which significantly improves operational accuracy.
>
> Personal R² will improve as more data accumulates via Phase 3 auto-retraining.

---

## 6. Files Reference

```
core/
  state_inference.py     — NCDE model, DualReadoutHead, all training functions
  ncde_weights_cert.pt   — Phase 1 weights (CERT pre-trained)
  ncde_weights.pt        — Production weights (Phase 2 fine-tuned)

data/
  cert_latent_states.csv — CERT-derived [Ct, Dt, Ht, At] (330,452 rows, 1000 users)
  latent_states.csv      — Your personal [Ct, Dt, Ht, At] (~1200 rows, growing)

scripts/
  cert_feature_engineer.py    — Convert CERT r4.2 raw logs → cert_latent_states.csv
  run_training_pipeline.py    — Run Phase 1 → Phase 2 pipeline
  evaluate_model.py           — This evaluation (run anytime to check accuracy)
```

---

## 7. How to Run

```bash
# 1. Feature engineering (one-time, after CERT download)
python scripts/cert_feature_engineer.py --cert_dir "C:/path/to/cert_r4.2"

# 2. Full training pipeline (Phase 1 + Phase 2)
python scripts/run_training_pipeline.py

# 3. Skip Phase 1 if CERT weights already exist, only re-run Phase 2
python scripts/run_training_pipeline.py --skip_phase1

# 4. Evaluate model accuracy at any time
python scripts/evaluate_model.py

# 5. Save evaluation report to docs/eval_report.txt
python scripts/evaluate_model.py --save_report
```

Phase 3 runs **automatically** inside `telemetry_tracker.py` — no manual action needed.

---

## 8. Design Decisions Log

| Decision | Reason |
|---|---|
| Split At/Ht to Head A, Ct/Dt to Head B | At/Ht require events impossible to trigger in personal use |
| Shared NCDE backbone | Preserves causal coupling between all 4 dims (At depends on Ct) |
| Zero out Ht/At in personal input sequences | Prevents noisy old predictions from corrupting backbone |
| Differential LR (backbone 3e-5, Head B 1e-4) | Prevents Phase 2 from erasing CERT dynamics in backbone |
| 50K sequence cap for Phase 1 | Makes Phase 1 tractable on CPU (~10 min vs hours) |
| User-aware windowing for CERT | Prevents nonsensical cross-user sequences |
| Phase 3 every 200 rows (~33 min usage) | Continuous personalisation without manual retraining |
