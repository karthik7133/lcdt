# 🛡️ Cyber Watchdog: The 4-Layer Digital Twin (Predictive Human Analytics)

Layer 1 of the Cyber Watchdog is a multi-modal monitoring system that identifies moments of high vulnerability by analyzing both your **Physical State** (Fatigue/Cognitive Friction) and your **Digital Attack Surface**.

---

## 📂 Project Structure

```text
/
├── main.py                 # Main entry point to start sensors
├── api/
│   └── dashboard_api.py    # Flask server for telemetry & behaviour logging
├── core/
│   ├── baseline_engine.py  # Lifelong learning for typing/mouse baselines
│   └── context_api.py      # Sleep deficit & interruption detection
├── sensors/
│   ├── telemetry_tracker.py # Background watchdog (Physical + Habits)
│   └── fatigue_model_project/ # Vision Core AI (EAR/MAR analysis)
├── ui/
│   ├── alert_ui.py         # Fatigue intervention popup
│   └── frontend/           # Dashboard UI (React)
├── integrations/
│   └── browser_extension/  # Browser exposure hook (HTTP/Webmail)
├── simulations/
│   └── simulate_phishing_click.py # Phishing simulation tester
├── data/                   # All telemetry logs and CSVs
└── README.md
```

---

## 🧠 Layer 1.1: Human Sensing (Cognitive Biomarkers)
*Silent tracking of physical and mental state to identify fatigue.*

| Feature | Logic | File |
| :--- | :--- | :--- |
| **Typing Friction** | Detects high error rates (Backspaces/Deletes vs total keys). | `sensors/telemetry_tracker.py` |
| **Task Switching** | Monitors rapid window title changes per minute. | `sensors/telemetry_tracker.py` |
| **Sleep Deficit** | Flagged if Fatigue is detected in the 1st hour of work. | `core/context_api.py` |
| **Interruption Count**| Detects system notification sounds (Audio Peak analysis). | `sensors/telemetry_tracker.py` |
| **Vision Verification**| Deep learning check (EAR/MAR/Head Slump) via camera. | `sensors/vision/` |

---

## 💻 Layer 1.2: Digital Behaviour (Technical Intelligence)
*Monitoring the technical attack surface and security habits.*

| Feature | Logic | File |
| :--- | :--- | :--- |
| **Browser Exposure** | Extension hooks for Insecure HTTP and Webmail access. | `integrations/browser_extension` |
| **OS Hygiene Audit** | Queries `wmic` for the last security patch date (>30 days = Risk). | `core/baseline_engine.py` |
| **Password Habits** | Rewards "Paste" actions (Ctrl+V) after Password Manager use. | `sensors/telemetry_tracker.py` |
| **Adversarial Signals**| Ingests failures from external phishing/scam simulators. | `api/dashboard_api.py` |

---

## 🔮 Layer 2 & 3: Multi-Timescale State Inference (Digital Twin)
*The system's "Brain" that maps raw biomarkers into the Hidden Human State vector $Z_t$ using differential equations.*

| State | Definition | Dynamical Behavior |
| :--- | :--- | :--- |
| **Capacity ($C_t$)** | Total mental battery. | **Slow Dynamics**: Drifts towards a baseline $\mu$. |
| **Demand ($D_t$)** | Current cognitive overwhelm. | **Fast Dynamics**: Spikes with work, "cools off" during rest. |
| **Habits ($H_t$)** | Cybersecurity hygiene score. | **Medium Dynamics**: Evolves via a learning rate $\eta$. |
| **Adversarial ($A_t$)** | Current threat exposure. | **Static Decay**: Maxed by attacks, decays over time. |
| **Reserve Gap ($CRG_t$)** | The gap between supply and demand. | Calculated as $D_t - C_t$ to identify "Cognitive Debt". |

---

## 📈 Layer 4: Predictive Analytics (Cyber Risk Forecaster)
*Converting psychology into business-ready predictive metrics.*

| Metric | Definition | Mathematical Core |
| :--- | :--- | :--- |
| **Mistake Probability** | The 0-100% chance of a cyber error. | Logistic Sigmoid Function ($\sigma$). |
| **Risk Trajectory** | 7-day forecast of vulnerability. | Recursive Capacity degradation simulation. |

**File**: `core/risk_forecaster.py`

**File**: `core/state_inference.py`

---

## 🚀 Getting Started

### 1. Start the Backend (Dashboard API)
The dashboard must be running to receive behaviour and simulation signals.
```bash
python api/dashboard_api.py
```

### 2. Run the Watchdog
The main watchdog monitors your physical inputs and digital habits.
```bash
python main.py
```

### 3. Test a Simulation
Simulate a phishing link click to see the system react in real-time:
```bash
python simulations/simulate_phishing_click.py
```

---

## 🛡️ Privacy Guarantee
*   **No Keylogging**: We count the *number* of keys, never the content.
*   **Privacy-First Vision**: The camera only boots when inactivity is detected and shuts down instantly once verification is complete.
*   **Local Processing**: All telemetry is stored and processed on your machine.
