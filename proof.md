# 📜 Proof of Mathematical Integrity: LCDT Framework

This document provides concrete proof that the mathematical models documented in the **Algorithm Walkthrough** and **PDF Reports** are identical to the actual production code constants in the Cyber Watchdog system.

---

## 1. Layer 4: Risk Prediction Weights (Pₘₜ)
The mistake probability uses a Logistic Sigmoid function squashing a linear combination of states.

| Parameter | Documentation | Source Code (`core/risk_forecaster.py`) | Status |
| :--- | :--- | :--- | :--- |
| **Beta 1 (CRG)** | -3.0 | `self.beta1 = -3.0` | ✅ Identical |
| **Beta 2 (Habits)** | -3.0 | `self.beta2 = -3.0` | ✅ Identical |
| **Beta 3 (Adversarial)** | +6.0 | `self.beta3 = 6.0` | ✅ Identical |
| **Bias Offset** | 0.5 | `self.bias = 0.5` | ✅ Identical |
| **Formula** | $1 / (1 + e^{-x})$ | `return 1 / (1 + np.exp(-x))` | ✅ Identical |

---

## 2. Layer 3: System Dynamics (Timescales)
These constants control how fast stress builds, energy recovers, and habits are learned.

| Parameter | Documentation | Source Code (`core/state_inference.py`) | Status |
| :--- | :--- | :--- | :--- |
| **Recovery Rate (α)** | 0.01 | `self.alpha_up = 0.01` | ✅ Identical |
| **Cooling Rate (λ/β)** | 0.2 | `self.beta = 0.2` | ✅ Identical |
| **Learning Rate (η)** | 0.1 | `self.eta = 0.1` | ✅ Identical |
| **Time Step (dt)** | 1.0 (10s) | `self.dt = 1.0` | ✅ Identical |
| **Reserve Gap Formula** | $C_t - D_t$ | `self.state["CRGt"] = self.state["Ct"] - self.state["Dt"]` | ✅ Identical |

---

## 3. Layer 1: Sensor Thresholds
The raw data capture logic that feeds the inference engine.

| Parameter | Documentation | Verified Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Audio Peak Floor** | 0.05 | `if peak > 0.05:` (Sensor Logic) | ✅ Identical |
| **Switch Normalization** | 40 switches | `task_switches / 40.0` | ✅ Identical |
| **Notification Norm.** | 10 alerts | `notification_count / 10.0` | ✅ Identical |

---

## 4. Advanced "Biological" Logic (In-Code Extensions)
The production code includes additional complexity for high-fidelity simulation:

*   **Recovery Inhibition**: Recovery speed is reduced by **90%** (`alpha_up * 0.1`) if keyboard or mouse activity is detected.
*   **Circadian Rhythm**: The system automatically penalizes the capacity baseline ($\mu$) by **0.2** between 11 PM and 5 AM.
*   **Burnout Drain**: If Stress ($D_t$) stays above **0.6** for prolonged periods, the daily forecast (`forecast_trajectory`) begins permanently draining Capacity ($C_t$) by **2% per day**.

---

### Verification Summary
The documentation provides a 1:1 human-readable map of the underlying Python implementation. Any simulation run in **Layer 5 (Digital Twin)** uses these exact verified weights to forecast long-term security risk.
