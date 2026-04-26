# 🛡️ Cyber Watchdog: Lifelong Cognitive-Cyber Digital Twin (V5)

> A 5-layer scientific system that predicts and prevents cyber-security errors by modeling the human **internal cognitive state** as a living dynamical system — not a static checklist.

---

## 📂 Project Structure

```text
/
├── main.py                          # Entry point: starts all sensors & watchdog
├── api/
│   └── dashboard_api.py             # Flask server: telemetry ingestion, simulation & state API
├── core/
│   ├── state_inference.py           # L2/L3: NCDE + EMA hybrid cognitive state engine
│   ├── simulation_engine.py         # L5: Causal (SCM) Monte-Carlo simulation engine
│   ├── risk_forecaster.py           # L4: Logistic risk scorer & 7-day trajectory forecaster
│   ├── baseline_engine.py           # Lifelong learning for typing/mouse baselines & OS hygiene
│   ├── context_api.py               # Google Calendar integration, sleep deficit & interruption detection
│   └── ncde_weights.pt              # Trained Neural CDE model weights
├── sensors/
│   ├── telemetry_tracker.py         # Background watchdog (keystrokes, mouse, audio, windows)
│   └── fatigue_model_project/       # Vision Core AI (EAR / MAR / Head-Pose analysis)
├── simulations/
│   └── simulate_phishing_click.py   # Phishing/adversarial signal injector for testing
├── integrations/
│   └── browser_extension/           # Browser hook (HTTP exposure & Webmail detection)
├── ui/
│   ├── alert_ui.py                  # Native fatigue intervention popup
│   └── frontend/                    # React + Vite dashboard (Recharts, Lucide, Glassmorphism HUD)
├── data/                            # All telemetry logs and CSV archives
├── ARCHITECTURE_V5.md               # Full technical architecture reference
└── README.md
```

---

## 🏗️ Architecture Overview

The system is organized as a **5-layer pipeline**, each layer building on the output of the previous one.

```
Layer 1 → Raw Sensors      (Physical + Digital Biomarkers)
Layer 2 → State Inference  (Latent Vector Zt from NCDE + EMA)
Layer 3 → Dynamics Engine  (SDEs: Slow / Fast / Medium timescales)
Layer 4 → Risk Forecaster  (Mistake Probability + 7-Day Trajectory)
Layer 5 → Digital Twin Sim (Causal SCM Monte-Carlo Intervention Engine)
```

---

## 🔬 Layer 1: Human & Digital Sensing (The Input Layer)

*Multi-modal capture of physical and digital biomarkers.*

### 1.1 Human Sensing (Cognitive Biomarkers)

| Feature | Logic | File |
| :--- | :--- | :--- |
| **Typing Friction** | Ratio of corrections (Backspaces/Deletes) to total keystrokes. High ratio = fatigue. | `sensors/telemetry_tracker.py` |
| **Task Switching** | Rapid window-title changes per minute (20+ = context-overload). | `sensors/telemetry_tracker.py` |
| **Sleep Deficit** | Flagged when fatigue is detected in the first hour of a session. | `core/context_api.py` |
| **Interruption Count** | Audio-peak detection of system notification sounds via `pycaw`. | `sensors/telemetry_tracker.py` |
| **Calendar Context** | Google Calendar integration to detect back-to-back meeting load. | `core/context_api.py` |
| **Vision Verification** | Deep-learning check (EAR / MAR / Head Slump) via camera. | `sensors/fatigue_model_project/` |

### 1.2 Digital Behaviour (Technical Intelligence)

| Feature | Logic | File |
| :--- | :--- | :--- |
| **Browser Exposure** | Extension hooks for Insecure HTTP and high-risk Webmail access. | `integrations/browser_extension/` |
| **OS Hygiene Audit** | Queries last security patch date (>30 days = Risk). | `core/baseline_engine.py` |
| **Password Habits** | Rewards Ctrl+V Paste actions following Password Manager activity. | `sensors/telemetry_tracker.py` |
| **Adversarial Signals** | Ingests simulated phishing clicks and credential-scam failures. | `api/dashboard_api.py` |

---

## 🧠 Layer 2: State Inference — Latent Vector Mapping

*Maps raw Layer 1 signals into the Hidden Human State vector **Z_t**.*

The inference engine in `core/state_inference.py` uses a **Neural Controlled Differential Equation (NCDE)** model (weights in `core/ncde_weights.pt`) combined with an **EMA-smoothed hybrid** to produce stable, continuous-time cognitive state estimates — even during sensor noise or idle gaps.

| State Variable | Definition | Timescale |
| :--- | :--- | :--- |
| **C_t (Capacity)** | Total mental energy available. | Slow (hours) |
| **D_t (Demand)** | Current cognitive pressure / overwhelm. | Fast (minutes) |
| **H_t (Habits)** | Cybersecurity hygiene score. | Medium (days) |
| **A_t (Adversarial)** | Immediate threat-exposure level. | Event-driven |
| **CRG_t (Reserve Gap)** | D_t − C_t: positive = Cognitive Debt. | Derived |

---

## ⚙️ Layer 3: Multi-Timescale Dynamics (The "Brain")

*Stochastic Differential Equations give the hidden state biological momentum.*

### Slow Dynamics — Cognitive Capacity (C_t)
```
dC_t = α(μ - C_t)dt + σ·dW_t
```
Capacity slowly drifts toward a personal baseline (μ). A sleep deficit lowers μ, making recovery take hours.

### Fast Dynamics — Cognitive Demand (D_t)
```
dD_t = [f(work) - β(D_t - 0.1)]dt + σ·dW_t
```
Demand spikes immediately with workload and cools off slowly — it cannot drop to zero the instant work stops.

### Medium Dynamics — Habit Evolution (H_t)
```
H_{t+1} = H_t + η·R_t
```
Security habits evolve via a Learning Rate (η) multiplied by behavioral rewards/penalties (R_t).

**Implementation**: `core/state_inference.py`

---

## 📈 Layer 4: Predictive Analytics (The Risk Forecaster)

*Converts abstract psychology into a concrete, actionable business metric.*

### Mistake Probability — P(M_t)
```
P(M_t) = σ(β₁·CRG_t + β₂·H_t + β₃·A_t + bias)
```
- **Reserve Gap (CRG_t)**: Positive cognitive debt exponentially raises risk.
- **Habit Shield (H_t)**: Acts as a protective negative weight.
- **Adversarial (A_t)**: Real-time threat amplifier.

### 7-Day Risk Trajectory
Recursively simulates Capacity drain under sustained Demand to predict the exact day a user enters a **Burnout Window** — before it happens.

**Implementation**: `core/risk_forecaster.py`

---

## 🔮 Layer 5: Digital Twin Simulation (Causal Inference Engine)

*Answers "what if?" questions using Structural Causal Models and Monte-Carlo sampling.*

The simulation engine (`core/simulation_engine.py`) models the **Expected Risk** under specific policy interventions — `E[M_{t:T} | do(π)]`:

| Intervention Policy | Effect Modelled |
| :--- | :--- |
| `do(security_training)` | Injects a reward spike into the Habit learning rate. |
| `do(ui_policy_change)` | Zeroes task-switching penalties to quantify UX impact on risk. |
| `do(increased_workload)` | Simulates long-term failure rate under extreme stress scenarios. |
| `do(aging_progression)` | Simulates long-term cognitive drift and capacity reduction. |

Results are streamed to the React dashboard as stochastic probability distributions.

---

## 🚀 Getting Started

### 1. Start the Backend API
```bash
python api/dashboard_api.py
```
Runs on `http://localhost:5050`. Exposes telemetry ingestion, cognitive state, and simulation endpoints.

### 2. Run the Watchdog Engine
```bash
python main.py
```
Starts all sensors (keyboard, mouse, audio, vision) and the state inference pipeline.

### 3. Launch the Frontend Dashboard
```bash
cd ui/frontend
npm install
npm run dev
```
Opens the live Glassmorphism HUD showing real-time cognitive metrics, risk score, and simulation results.

### 4. Inject an Adversarial Signal (Testing)
```bash
python simulations/simulate_phishing_click.py
```
Sends a phishing-click event to the API to verify the system reacts in real-time.

---

## 🛠️ Technical Stack

| Layer | Technology |
| :--- | :--- |
| **Core Logic** | Python 3.11 — NumPy, Pandas, Flask |
| **NCDE Model** | PyTorch (`torchcde`) |
| **Sensors** | Pynput (Keys/Mouse), Pycaw (Audio), PyGetWindow (Context) |
| **Calendar** | Google Calendar API (`google-auth`, `google-api-python-client`) |
| **Frontend** | React + Vite + Recharts + Lucide-React |
| **UI Aesthetic** | Dark-mode Glassmorphism, Neon HUD, Orbitron Typography |

---

## 🔒 Privacy Guarantee

- **No Keylogging**: The system counts keystroke *frequencies* — never content.
- **Privacy-First Vision**: The camera activates only on inactivity detection and shuts down immediately after verification.
- **Local Processing**: All telemetry is stored and processed entirely on-device. Nothing is sent to external servers.
- **Credential Safety**: Google Calendar credentials (`core/credentials.json`) are read-only and stored locally.
