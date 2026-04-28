# 🛡️ Lifelong Cognitive Digital Twin (LCDT)

> A **6-layer scientific system** that predicts and prevents human cyber-security errors by modelling the employee's internal cognitive state as a continuous, living dynamical system — using Neural Controlled Differential Equations, Structural Causal Models, Weibull Survival Analysis, and Monte-Carlo counterfactual simulation.

---

## 📂 Project Structure

```text
/
├── main.py                            # Entry point — starts all sensors & inference loop
│
├── core/
│   ├── state_inference.py             # L1 + L2 + L3  —  Behaviour Graph + NCDE + Personalisation
│   ├── risk_forecaster.py             # L4 + L5  —  Causal SCM + Bayesian Hazard Model
│   ├── simulation_engine.py           # L6  —  Monte-Carlo Digital Twin Simulator
│   ├── baseline_engine.py             # Lifelong typing/mouse baseline + OS hygiene audit
│   ├── context_api.py                 # Google Calendar integration (L1 workload signal)
│   └── ncde_weights.pt                # Trained Neural CDE model weights (PyTorch)
│
├── sensors/
│   ├── telemetry_tracker.py           # Background watchdog: keys, mouse, audio, windows
│   └── fatigue_model_project/         # Vision Core AI: EAR / MAR / Head-Pose via camera
│       ├── fuzzy_engine.py            # Core Fuzzy Logic inference engine (Mamdani, CoG defuzz)
│       ├── train_fuzzy.py             # Evaluates fuzzy engine on dataset, exports fuzzy_model.pkl
│       ├── live_predictor_fuzzy.py    # Live webcam fatigue predictor using fuzzy engine
│       └── augment_dataset.py         # Extracts features from archive images, augments to 10k+ rows
│
├── api/
│   └── dashboard_api.py               # Flask REST API — bridges all 6 layers to the frontend
│
├── simulations/
│   └── simulate_phishing_click.py     # Adversarial signal injector for integration testing
│
├── integrations/
│   └── browser_extension/             # Chrome hook: HTTP exposure & Webmail detection
│
├── ui/
│   ├── alert_ui.py                    # Native OS fatigue intervention popup
│   └── frontend/                      # React + Vite + Recharts dashboard (Glassmorphism HUD)
│
└── data/                              # All telemetry, latent state logs, and CSV archives
    ├── latent_states.csv              # Historical [Ct, Dt, Ht, At, CRGt, risk_pct] — NCDE training data
    ├── telemetry_history.csv          # Raw keystroke / mouse history for baseline learning
    ├── digital_behaviour.csv          # Browser extension behaviour log
    └── adversarial_failures.csv       # Phishing / credential-theft event log
```

---

## 🏗️ Architecture: 6-Layer Pipeline

```
Raw Sensors  ──►  Behaviour Graph  ──►  NCDE (Zt)  ──►  Personalisation
   [L1]                [L1]              [L2]               [L3]
                                          │
                                          ▼
                            Causal Risk Engine (SCM)  ──►  Survival Analysis
                                    [L4]                        [L5]
                                          │
                                          ▼
                               Monte-Carlo Digital Twin
                                        [L6]
```

Each layer is a fully implemented Python class in the `core/` package.

---

## 🔬 Layer 1 — Multimodal Behaviour Graph

**File:** `core/state_inference.py` → `BehaviourGraphEngine`

Instead of a flat feature vector, all raw sensor signals are represented as a **Temporal Directed Multigraph** (`networkx.MultiDiGraph`). Every signal is an edge between named semantic nodes with a timestamp and weight.

### The 20 Signals and Their Graph Topology

| Signal | Source Node | Target Node | Event Type |
| :--- | :--- | :--- | :--- |
| `key_count` | User | Workstation | typing |
| `mouse_entropy` | User | Workstation | mouse_movement |
| `typing_error_rate` | User | Workstation | typing_friction |
| `task_switches` | Apps | User | context_switch |
| `notification_count` | System | User | interruption |
| `workload_modifier` | Schedule | User | calendar_pressure |
| `insecure_http_hits` | Browser | Risk | insecure_browse |
| `webmail_hits` | Browser | Risk | webmail_access |
| `link_clicks` | Browser | Risk | link_click |
| `email_frequency` | Email | User | email_activity |
| `unknown_senders` | Email | Risk | phishing_signal |
| `avg_response_time` | Email | User | email_response |
| `low_strength_passwords` | Security | Risk | weak_password |
| `good_password_paste` | Security | User | good_habit |
| `os_update_delayed` | System | Risk | update_delay |
| `sleep_deficit` | Bio | User | sleep_debt |
| `vision_fatigue` | Bio | User | vision_fatigue |
| `phishing_clicked` | Threat | Risk | phishing_click |
| `scam_credentials_given` | Threat | Risk | credential_theft |
| `hour_of_day` | Clock | User | circadian |

Graph edges have a **5-minute TTL** — stale edges are pruned automatically so that Demand (Dt) drops to idle levels when the user stops working, preventing the "stuck high risk" problem.

### The 12-Dimensional Graph Embedding

The graph is read by a `get_embedding()` method that extracts a 12-dimensional structural feature vector for the NCDE to consume:

```
[0]  Total active edges
[1]  Mean edge weight
[2]  Weighted in-degree of User node
[3]  Weighted in-degree of Risk node
[4]  Weighted out-degree of Threat node
[5]  Weighted out-degree of Browser node
[6]  Weighted out-degree of Email node
[7]  Weighted in-degree of Security node
[8]  Weighted out-degree of Bio node
[9]  Weighted in-degree of Workstation node
[10] Number of distinct event types currently active
[11] Recency score (edges in last 30 seconds / total edges)
```

### Google Calendar Integration

`core/context_api.py` → `GoogleContextEngine` authenticates via OAuth 2.0 and fetches today's calendar to seed the `workload_modifier` signal before the session starts:

| Meetings Today | Workload Modifier |
| :--- | :--- |
| 0 | 0.0 |
| 1–2 | 0.1 |
| 3–4 | 0.3 |
| 5+ | 0.5 |

---

## 🧠 Layer 2 — Neural Controlled Differential Equations (NCDE)

**File:** `core/state_inference.py` → `NCDEModel`, `CDEFunc`, `ReadoutHead`

The graph embedding sequence is consumed by a **Neural CDE** — a continuous-time deep learning model that replaces hand-crafted differential equations entirely.

### The NCDE Equation

```
dZt = f_θ(Zt) dt + g_ϕ(Zt) dXt
```

- `Xt` = the graph embedding path (interpolated as a **Hermite cubic spline**)
- `g_ϕ(Zt)` = the **CDE function** — a 2-layer MLP with `tanh` activation that outputs `(hidden_dim × input_dim)` — the control matrix
- Integration is performed by `torchcde.cdeint` using the **RK4** solver over the interval `[0, 1]`
- `f_θ` is implicitly contained inside the MLP (autonomous drift)

### Architecture

```
Input (batch, seq_len=10, 12)
    │
    ▼ Hermite cubic spline interpolation
    │
    ▼ Initial Encoder: Linear(12 → 8) + Tanh  →  z0
    │
    ▼ CDE Integration: torchcde.cdeint [RK4]
    │     g_ϕ(Zt): Linear(8→64)→Tanh→Linear(64→64)→Tanh→Linear(64→8×12)→Tanh
    │
    ▼ z_final  (batch, 8)
    │
    ▼ Readout Head: Linear(8→32)→ReLU→Linear(32→4)→Sigmoid
    │
Output: [Ct, Dt, Ht, At]  ∈ (0, 1)
```

**Dimensions:** Input=12, Hidden=8, Output=4  
**Training:** MSE loss on historical `latent_states.csv` sequences, Adam optimizer, 60 epochs  
**Weights:** Saved and loaded from `core/ncde_weights.pt`

### Stability Gate

When the graph embedding window is nearly constant (idle, std < 0.05), the NCDE is bypassed entirely to prevent RK4 numerical noise from causing oscillations. Instead, Ct/Ht/At decay gently toward the user's personal prior at a 5% recovery rate per tick.

### Signal-Driven Demand Override

Demand (Dt) is **always** computed directly from raw signals — not from the NCDE — to ensure it drops to the idle floor `~0.1` immediately when the user stops working:

```
Dt = 0.10 (idle floor)
   + 0.35 × typing_pressure    (key_count / personal_baseline)
   + 0.20 × task_switch_norm   (task_switches / 10)
   + 0.15 × notification_norm  (notification_count / 5)
   + 0.10 × calendar_pressure  (workload_modifier)
   + 0.10 × sleep_penalty      (if sleep_deficit = TRUE)
```

---

## 👤 Layer 3 — Personalized Latent Cognitive State Space

**File:** `core/state_inference.py` → `UserProfile`

Every employee has different cognitive physics. Layer 3 maintains a **per-user Bayesian prior**:

```
Zt(user) ~ N(μ_user, Σ_user)
```

### Online MAP Update (Exponential Moving Average)

```
μ_new   = (1 - α) × μ_old + α × z_observed
σ²_new  = (1 - α) × σ²_old + α × (z_observed - μ_new)²
```

where `α = 0.05` (slow adaptation to prevent reacting to single outlier sessions).

### Bayesian Shrinkage (Personalization Blend)

The NCDE output is blended with the user prior before being used downstream:

```
z_posterior = λ × z_model + (1 - λ) × μ_user
```

where `λ = min(0.85, 0.5 + n_obs / 200)` — the model is trusted more as more personal data accumulates.

### EMA Smoothing + Slew Rate Limiting

After personalisation, an additional EMA (`α = 0.10`) and per-tick max-change limiter (`MAX_SLEW = 0.05`) prevent oscillation from tick to tick.

---

## ⚖️ Layer 4 — Causal Risk Engine (Structural Causal Model)

**File:** `core/risk_forecaster.py` → `CausalRiskEngine`

Layer 4 models **causality**, not just correlation. The risk score is computed from a **Structural Causal Model (SCM)**:

```
Risk = σ( B_reserve × CRGt  +  B_habits × Ht  +  B_threat × At  +  bias )
```

Where `CRGt = Ct − Dt` (the Cognitive Reserve Gap — negative means Cognitive Debt).

### Data-Driven Causal Weight Estimation

When `latent_states.csv` contains ≥ 50 observations, causal weights are estimated via **OLS on the logit scale** from actual telemetry data:

```
logit(P) = B_reserve × CRGt + B_habits × Ht + B_threat × At + bias
B = (X'X)⁻¹ X'y
```

Default calibrated weights (used when data is insufficient):

| Weight | Value | Interpretation |
| :--- | :--- | :--- |
| B_reserve | −2.5 | Higher capacity reserve → lower risk |
| B_habits | −1.8 | Stronger security habits → strong protection |
| B_threat | +4.5 | Active adversarial event → sharp risk spike |
| bias | +0.5 | Prior risk at neutral state |

### do-Calculus Interventions

Layer 4 supports **surgical do-interventions** (`do(π)`) that modify a specific causal variable to compute counterfactual risk:

| Intervention `do(π)` | Variable Modified | Effect |
| :--- | :--- | :--- |
| `reduce_notifications` | Dt × 0.70 | 30% demand reduction |
| `reduce_meetings` | Dt × 0.80 | 20% demand reduction |
| `pause_work` | Dt × 0.30 | 70% demand reduction |
| `improve_sleep` | Ct + 0.15 | Capacity restoration |
| `security_training` | Ht + 0.20 | Habit improvement |
| `adversarial_drill` | At × 0.50 | Threat exposure awareness |

---

## 📈 Layer 5 — Bayesian Temporal Hazard Model (Survival Analysis)

**File:** `core/risk_forecaster.py` → `BayesianHazardModel`

Layer 5 replaces logistic regression with a proper **survival analysis** model that answers: *"What is the probability of a security mistake in the next τ hours?"*

### Weibull Proportional Hazard

```
P(mistake in [t, t+τ] | Zt) = 1 - exp(-(τ / λ(Zt))^k)
```

- **λ(Zt)** = scale parameter, inversely proportional to current risk: `λ = (100 - risk_pct) / 10`
- **k** = shape parameter with Gaussian prior `k ~ N(1.5, 0.15)` (k > 1 means increasing hazard over time)

### Posterior Uncertainty via Monte-Carlo

Instead of a point estimate, 500 samples are drawn from the posterior of `k` to produce a full **probability distribution** with CI bands:

```python
k_samples = N(k_mean, k_std, n=500)
probs     = 1 - exp(-(τ / λ)^k_samples)
→ { mean, median, std, lower_5th, upper_95th }
```

### Online Posterior Update

When a real security mistake is recorded, the Weibull shape `k` is updated via **Method of Moments**:

```
k_new ≈ (mean_time_to_mistake / std)^1.086
```

---

## 🔮 Layer 6 — True Digital Twin: Monte-Carlo Counterfactual Simulation

**File:** `core/simulation_engine.py` → `DigitalTwinSimulator`

Layer 6 simulates **500 independent future trajectories** using vectorised NumPy operations — completing in ~0.5 seconds — to compute:

```
Future trajectories ~ p(Zt:T | Zt, do(π))
```

### Stochastic Dynamics (Learned Noise Parameters)

Each trajectory evolves the state `[Ct, Dt, Ht, At]` with Gaussian noise calibrated from the historical standard deviation in `latent_states.csv`:

| Variable | Noise σ (default) | Dynamics |
| :--- | :--- | :--- |
| Ct | 0.015 | Slow — drains when Dt > threshold, recovers when idle |
| Dt | 0.025 | Fast — drifts per policy, decays at rest |
| Ht | 0.010 | Medium — shifts slowly with training reward |
| At | 0.020 | Event-driven — spikes on attack, decays at 0.92/step |

### Supported Intervention Policies

| Policy | Effect on Dynamics |
| :--- | :--- |
| `baseline` | No intervention |
| `increased_workload` | Dt drift +0.04, Ct drains faster |
| `reduced_workload` | Dt drift −0.03 |
| `reduce_meetings` | Dt drift −0.03 |
| `reduce_notifications` | Dt drift −0.02 |
| `security_training` | Ht gain +0.01 per step |
| `improve_sleep` | Ct recovery +0.02 per step |
| `pause_work` | Dt drift −0.05, Ct recovery +0.04 |
| `aging_progression` | Ct ceiling decays over time |
| `adversarial_drill` | Random At spikes at 5% probability |

### Aggregated Outputs (per time step)

Each simulation returns, for every step across 500 trajectories:

```
Mean_Risk         — expected risk across all futures
Median_Risk       — median trajectory
Risk_Upper_95     — 95th percentile (worst case)
Risk_Lower_5      — 5th percentile (best case)
Risk_Std          — trajectory uncertainty
Mean_Capacity     — expected cognitive capacity
Mean_Hazard_24h   — expected P(mistake in 24h) via Weibull
Burnout_Confidence — % of trajectories where risk > 70%
Burnout_Horizon   — first step where mean risk crosses 70%
```

### Counterfactual Delta

The API supports computing `E[Risk | do(π)] − E[Risk | baseline]` per step to quantify the exact risk reduction from any intervention.

---

## 🌐 REST API

**File:** `api/dashboard_api.py` — Flask server on `http://localhost:5000`

| Method | Endpoint | Layer | Description |
| :--- | :--- | :--- | :--- |
| GET | `/api/summary` | L2/L3 | Live state snapshot: Ct, Dt, Ht, At, CRGt, risk_pct |
| GET | `/api/latent_states` | L2/L3 | Historical state time series (last 30 records) |
| GET | `/api/stats` | L1 | Raw keystroke telemetry history |
| GET | `/api/forecast` | L5 | 7-day risk trajectory with CI bands |
| GET | `/api/hazard` | L5 | P(mistake in 1h / 24h / 7d) with posterior uncertainty |
| GET | `/api/counterfactuals` | L4 | All 6 interventions ranked by risk reduction |
| POST | `/api/simulation` | L6 | Monte-Carlo simulation for a given policy |
| POST | `/api/counterfactual_delta` | L6 | E[Risk\|do(π)] vs baseline delta |
| POST | `/api/behaviour` | L1 | Browser extension event log |
| POST | `/api/adversarial_signal` | L1 | Phishing / scam signal injection |

---

## 🚀 Getting Started

### 1. Start the Backend API
```bash
python api/dashboard_api.py
```
Initialises the NCDE model (loading `ncde_weights.pt`), the Causal SCM, and the Monte-Carlo simulator. Runs on `http://localhost:5000`.

### 2. Run the Watchdog Engine
```bash
python main.py
```
Starts all sensors (keyboard, mouse, audio, windows), runs the NCDE inference loop every 10 seconds, and logs latent states to `data/latent_states.csv`.

### 3. Launch the React Dashboard
```bash
cd ui/frontend
npm install
npm run dev
```
Opens the live dashboard with real-time Ct/Dt/Ht charts, risk score, 7-day survival forecast, and Monte-Carlo simulation panel.

### 4. Test an Adversarial Signal
```bash
python simulations/simulate_phishing_click.py
```
Posts a phishing-click event to `/api/adversarial_signal` to verify Layer 1 → Layer 4 threat propagation in real-time.

---

## 👁️ Vision Core — Fuzzy Logic Fatigue Detection

**Folder:** `sensors/fatigue_model_project/`  
**Engine:** Expert Fuzzy Logic System (Mamdani inference + Centroid of Gravity defuzzification)  
**Replaces:** Previous SVM classifier (`train_svm.py` + `fatigue_model_v2.pkl`)

The vision subsystem detects operator fatigue in real time via a webcam. MediaPipe FaceMesh extracts biometric features every frame; these are fed into a pure-NumPy fuzzy inference engine that produces a continuous **Fatigue Index ∈ [0, 100]**.

### Why Fuzzy Logic over SVM?

| Property | SVM (old) | Fuzzy Logic (new) |
| :--- | :--- | :--- |
| Requires labelled training data | ✅ Yes (550 rows) | ❌ No — rule-based |
| Interpretable decisions | ❌ Black-box kernel | ✅ Human-readable rules |
| Handles uncertainty / grey zones | ❌ Hard boundary | ✅ Soft membership degrees |
| Needs retraining on data shifts | ✅ Yes | ❌ No |
| Handles noisy biometric signals | ❌ Sensitive | ✅ Naturally robust |
| Clinical alignment | Indirect | ✅ Directly encodes EAR/MAR thresholds |

### Fuzzy Inference Pipeline

```
Webcam Frame (30 fps)
       │
       ▼  MediaPipe FaceMesh
       ├─ EAR  (avg Left + Right Eye Aspect Ratio)
       ├─ MAR  (Mouth Aspect Ratio)
       ├─ Pitch (nose-to-chin / cheek-to-cheek)
       └─ Blink detection (EAR < 0.18)
       │
       ▼  10-frame rolling buffer → EAR_MA
       │
       ▼  FuzzyFatigueEngine.predict(EAR, MAR, Pitch, EAR_MA)
          Step 1 — Fuzzification (triangular + trapezoidal MFs)
          Step 2 — 19 Expert Rules → activation strengths (alert / moderate / tired)
          Step 3 — Mamdani min-implication + max-OR aggregation
          Step 4 — Centroid of Gravity defuzzification → Fatigue Index
       │
       ▼  20-frame prediction_history smoothing
       ├─ smoothed_fatigue > 50% for 30 frames → sys.exit(80)  [TIRED]
       └─ smoothed_awake  for 45 frames        → sys.exit(0)   [AWAKE]
```

### Fuzzy Inputs & Membership Sets

| Input | Fuzzy Sets | Clinical Threshold |
| :--- | :--- | :--- |
| `EAR` | `closed`, `half_closed`, `open` | closed < 0.18, open > 0.25 |
| `MAR` | `normal`, `yawning` | yawning > 0.50 |
| `Pitch` | `nodding`, `upright` | nodding < 0.55 |
| `EAR_MA` | `stable_closed`, `drifting`, `stable_open` | 10-frame rolling average |

### Output & Decision Threshold

```
Fatigue Index universe: [0, 100]
  alert    → [0, 40]   (trapezoidal)
  moderate → [25, 75]  (triangular, peak at 50)
  tired    → [55, 100] (trapezoidal)

Decision: FatigueIndex ≥ 50  →  Label = 1 (Tired)
```

### Expected Performance

| Metric | Expected Range |
| :--- | :--- |
| Accuracy | 82–92% |
| Precision (Tired) | 85–94% |
| Recall (Tired) | 80–92% |
| F1 (macro) | 83–91% |

> Unlike SVM, accuracy is not the primary metric — the key advantage is **interpretability**: every decision can be traced back to which rules fired and with what strength.

### Vision Core Files

| File | Purpose |
| :--- | :--- |
| `fuzzy_engine.py` | Core fuzzy inference engine — membership functions, 19 rules, CoG defuzz |
| `train_fuzzy.py` | Evaluates engine on `dataset_v3.csv`, exports `fuzzy_model.pkl` + `fuzzy_eval_summary.json` |
| `live_predictor_fuzzy.py` | Live webcam predictor with HUD overlay |
| `augment_dataset.py` | Extracts EAR/MAR/Pitch from 2,200 archive images, augments to 10k+ rows |
| `math_utils.py` | Shared EAR / MAR / Pitch utility functions |

---

## 🛠️ Technical Stack

| Component | Technology |
| :--- | :--- |
| **Core Engine** | Python 3.11 — NumPy, Pandas, Flask |
| **Neural CDE Model** | PyTorch + `torchcde` (Hermite cubic spline, RK4 integration) |
| **Causal Graph** | NetworkX `MultiDiGraph` (Temporal Behaviour Graph) |
| **Survival Analysis** | NumPy — Weibull Monte-Carlo posterior sampling |
| **Vision / Fatigue Model** | Fuzzy Logic (Mamdani) — MediaPipe FaceMesh + pure-NumPy inference engine |
| **Sensors** | Pynput (keys/mouse), Pycaw (audio), PyGetWindow (windows) |
| **Calendar** | Google Calendar API v3 (OAuth 2.0) |
| **API** | Flask + Flask-CORS |
| **Frontend** | React 18 + Vite + Recharts + Lucide-React |
| **UI Aesthetic** | Dark-mode Glassmorphism, Neon HUD, Orbitron typography |

---

## 🔒 Privacy Guarantee

- **No Keylogging**: Only keystroke *counts and rates* are recorded — never content.
- **Privacy-First Vision**: Camera activates only on inactivity detection and shuts down immediately after EAR/MAR/Pitch verification completes. The fuzzy engine processes only numerical biometric ratios — no raw images are stored.
- **Fully Local**: All inference, telemetry, and model training runs on-device. No data leaves the machine.
- **Read-Only Calendar**: Google Calendar access is scoped to `calendar.readonly`. Credentials stored locally in `core/credentials.json`.
