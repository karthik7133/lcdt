# 🛡️ Lifelong Cognitive-Cyber Digital Twin: V5 Architecture

This document provides a comprehensive breakdown of the **5-Layer Architecture** implemented in the Cyber Watchdog project. The system is designed to predict and prevent cyber-security errors by modeling the human "Internal State" as a dynamical system.

---

## 📂 Layer 1: Human & Digital Sensing (The Input Layer)
*Capturing raw multi-modal biomarkers from physical and digital activity.*

### 1.1 Human Sensing (Cognitive Biomarkers)
- **Typing Friction**: Ratio of corrections (Backspaces/Deletes) to total keystrokes. High friction indicates cognitive fatigue.
- **Task Switching**: Frequency of active window title changes. 20+ switches/min indicates context-switching overwhelm.
- **Sleep Deficit**: Detected if fatigue occurs within the first hour of a session (via `core/context_api.py`).
- **Interruption Count**: Audio peak detection of system notifications using `pycaw`.
- **Vision Verification**: Deep learning verification (EAR/MAR/Head Pose) via camera using the Vision Core AI.

### 1.2 Digital Behaviour (Technical Intelligence)
- **Browser Exposure**: Real-time hooks for Insecure HTTP hits and high-risk Webmail access.
- **OS Hygiene**: Automated audit of security patch dates (>30 days = Risk).
- **Password Habits**: Tracking "Paste" events (Ctrl+V) following Password Manager activity.
- **Adversarial Signals**: Ingestion of simulated phishing clicks and scam credential failures.

---

## 🧠 Layer 2: State Inference (Latent Vector Mapping)
*Mapping raw Layer 1 signals into the Hidden Human State vector $Z_t$.*

The system aggregates all signals into a normalized vector:
- **$C_t$ (Capacity)**: Total mental energy available.
- **$D_t$ (Demand)**: Current cognitive pressure.
- **$H_t$ (Habits)**: Cybersecurity hygiene score.
- **$A_t$ (Adversarial)**: Immediate threat presence.

---

## 🔬 Layer 3: Multi-Timescale Dynamics (The "Brain")
*Implementing biological momentum and momentum using differential equations (Euler Method).*

Instead of static resets, Layer 3 uses **Stochastic Differential Equations** to ensure states have "memory."

### 1. Slow Dynamics (Cognitive Capacity - $C_t$)
$$dC_t = \alpha(\mu - C_t)dt + \sigma dW_t$$
- **Logic**: Capacity slowly drifts toward a baseline limit ($\mu$). If you have a sleep deficit, the baseline ($\mu$) drops, and it takes hours of "rest" (idle time) to recharge.

### 2. Fast Dynamics (Cognitive Demand - $D_t$)
$$dD_t = [f(work) - \beta(D_t - 0.1)]dt + \sigma dW_t$$
- **Logic**: Stress builds rapidly based on workload $f(work)$. The "cooling off" factor ($\beta$) ensures stress doesn't drop instantly to zero when work stops, but dissipates slowly.

### 3. Medium Dynamics (Habit Evolution - $H_t$)
$$H_{t+1} = H_t + \eta R_t$$
- **Logic**: Habits change slowly via a Learning Rate ($\eta$) multiplied by Rewards/Penalties ($R_t$) from digital behavior.

---

## 📈 Layer 4: Predictive Analytics (The Risk Forecaster)
*Converting abstract psychology into a concrete business metric: Cyber Mistake Probability.*

### 4.1 The Mistake Function ($P(M_t)$)
The system squashes the latent state through a **Logistic Sigmoid Function**:
$$P(M_t) = \sigma(\beta_1 \cdot CRG_t + \beta_2 \cdot H_t + \beta_3 \cdot A_t + \text{bias})$$
- **Reserve Gap ($CRG_t$)**: $D_t - C_t$. Positive gap (Debt) exponentially increases risk.
- **Habit Shield ($H_t$)**: Acts as a negative weight (protective factor).

### 4.2 Risk Trajectory Forecasting
The system recursively simulates the Capacity drain under current Demand to output a **7-Day Risk Forecast**, predicting exactly when a user will enter a "Burnout Window."

---

## 🔮 Layer 5: Digital Twin Simulation (Causal Inference)
*Modeling the impact of specific interventions: $E[M_{t:T} | do(\pi)]$.*

The simulation engine allows the system to calculate the "Expected Risk" given a specific policy change:
- **$do(\text{security\_training})$**: Injecting a reward spike into the Habit learning rate.
- **$do(\text{ui\_policy\_change})$**: Zeroing out task-switching penalties to show the impact of better software design.
- **$do(\text{increased\_workload})$**: Simulating the long-term failure rate under extreme stress.
- **$do(\text{aging\_progression})$**: Simulating long-term cognitive drift.

---

## 🛠️ Technical Stack
- **Core Logic**: Python 3.11 (Numpy, Pandas, Flask)
- **Sensors**: Pynput (Keys/Mouse), Pycaw (Audio), PyGetWindow (Context)
- **Frontend**: React + Vite + Recharts + Lucide-React
- **Aesthetics**: Dark-mode Glassmorphism, Neon HUD, Orbitron Typography.

**Implementation Status**: All 5 Layers are fully implemented and integrated across `core/`, `sensors/`, and `ui/`.
