# 🗺️ Layer 2, 3 & 4 Mapping: The Digital Twin Inference Logic

This document details how the **12 Biomarkers** from Layer 1 are mapped into psychological states (Layer 2/3) and then squashed into predictive risk metrics (Layer 4).

---

## 🧠 State Definitions (Layers 2 & 3)
The Inference Engine calculates four primary latent states and one critical derivative:

1.  **Cognitive Capacity ($C_t$)**: Your total mental energy/battery. (Slow Dynamics)
2.  **Cognitive Demand ($D_t$)**: How much pressure is currently on your brain. (Fast Dynamics)
3.  **Habit State ($H_t$)**: Your overall security hygiene score. (Medium Dynamics)
4.  **Adversarial Exposure ($A_t$)**: The immediate presence of an active threat.
5.  **Reserve Gap ($CRG_t$)**: The difference between Demand and Capacity ($D_t - C_t$).

---

## 📈 Layer 4: Predictive Analytics (Cyber Risk)

Layer 4 converts the abstract psychological states into a concrete **Cyber Mistake Probability ($P(M_t)$)**.

### 1. The Predictive Formula
The system uses a **Logistic Sigmoid Function** to calculate risk:

$$P(M_t) = \sigma(\beta_1 \cdot CRG_t + \beta_2 \cdot H_t + \beta_3 \cdot A_t + \text{bias})$$

Where:
- **$\sigma$ (Sigmoid)**: Squashes the value between 0% and 100%.
- **$\beta_1 = 4.0$ (Reserve Gap)**: Positive weight. High cognitive debt increases risk.
- **$\beta_2 = -3.0$ (Habits)**: Negative weight. Good hygiene acts as a protective shield.
- **$\beta_3 = 6.0$ (Adversarial)**: Massive positive weight. Active threats bypass psychological defenses.
- **$\text{bias} = -2.0$**: Sets the "healthy baseline" risk to approx. 5%.

### 2. Risk Trajectory Forecasting
By simulating the slow degradation of Capacity ($C_t$) under high Demand ($D_t$), the system forecasts the risk trajectory over 7+ days, identifying potential "Burnout Windows" where a mistake becomes inevitable.

---

## 🧠 Layer 3: Multi-Timescale Dynamical System
(See previous sections for $dC_t$, $dD_t$, and $H_{t+1}$ formulas using Euler Method).

---

## 🔬 The Inference Formula: Cognitive Reserve Gap ($CRG_t$)
$$CRG_t = D_t - C_t$$

- **$CRG_t < 0$ (Safe)**: You have a "buffer."
- **$CRG_t > 0$ (Vulnerable)**: You are in **Cognitive Debt**.

---

## 🛠️ Implementation Proof
- **States**: `core/state_inference.py`
- **Predictive Risk**: `core/risk_forecaster.py`
- **Flow**: Layer 1 Signals $\rightarrow$ Layer 3 Differential Equations $\rightarrow$ Layer 4 Sigmoid Projection.
