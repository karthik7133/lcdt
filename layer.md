# 🛡️ LCDT — All 6 Layers: Algorithms + Sample Data

---

## 🔬 Layer 1 — Multimodal Behaviour Graph

**File:** `core/state_inference.py` → `BehaviourGraphEngine`  
**Algorithm:** Temporal Directed Multigraph (NetworkX `MultiDiGraph`)  
**No SVM, no classifier of any kind — pure graph theory.**

### What it does
- Receives 20 raw sensor signals every 10 seconds
- Encodes each non-zero signal as a directed edge `(Source → Target)` with a weight and Unix timestamp
- Prunes edges older than **300 seconds (5-min TTL)** so idle periods drop naturally
- Extracts a **12-dimensional structural embedding** for the NCDE

### Signal → Graph Edge Mapping

| Signal | Source → Target | Event Type |
|--------|----------------|-----------|
| `key_count` | User → Workstation | typing |
| `mouse_entropy` | User → Workstation | mouse_movement |
| `typing_error_rate` | User → Workstation | typing_friction |
| `task_switches` | Apps → User | context_switch |
| `notification_count` | System → User | interruption |
| `workload_modifier` | Schedule → User | calendar_pressure |
| `insecure_http_hits` | Browser → Risk | insecure_browse |
| `webmail_hits` | Browser → Risk | webmail_access |
| `link_clicks` | Browser → Risk | link_click |
| `email_frequency` | Email → User | email_activity |
| `unknown_senders` | Email → Risk | phishing_signal |
| `avg_response_time` | Email → User | email_response |
| `low_strength_passwords` | Security → Risk | weak_password |
| `good_password_paste` | Security → User | good_habit |
| `os_update_delayed` | System → Risk | update_delay |
| `sleep_deficit` | Bio → User | sleep_debt |
| `vision_fatigue` | Bio → User | vision_fatigue |
| `phishing_clicked` | Threat → Risk | phishing_click |
| `scam_credentials_given` | Threat → Risk | credential_theft |
| `hour_of_day` | Clock → User | circadian |

### 12-Dimensional Embedding Output

| Index | Feature |
|-------|---------|
| [0] | Total active edges |
| [1] | Mean edge weight |
| [2] | Weighted in-degree of User node |
| [3] | Weighted in-degree of Risk node |
| [4] | Weighted out-degree of Threat node |
| [5] | Weighted out-degree of Browser node |
| [6] | Weighted out-degree of Email node |
| [7] | Weighted in-degree of Security node |
| [8] | Weighted out-degree of Bio node |
| [9] | Weighted in-degree of Workstation node |
| [10] | Distinct active event types |
| [11] | Recency score (edges in last 30s / total) |

### 📊 Sample Data — Layer 1 Raw Sensor Tick

**Input signals (one 10-second tick from `telemetry_history.csv`):**

```
timestamp          : 2026-04-22 13:09:06
keys_pressed       : 60
mouse_distance     : 1552.93
task_switches      : 3
notification_count : 2
hour_of_day        : 13
sleep_deficit      : FALSE
vision_fatigue     : 0.12
workload_modifier  : 0.1   (1 meeting today)
```

**Resulting graph edges added:**

```
User   → Workstation  [typing]         weight=60
User   → Workstation  [mouse_movement] weight=1552.93
Apps   → User         [context_switch] weight=3
System → User         [interruption]   weight=2
Clock  → User         [circadian]      weight=13
Bio    → User         [vision_fatigue] weight=0.12
Schedule→User         [calendar_pressure] weight=0.1
```

**12-dim embedding produced:**

```
[0]  n_edges          = 7
[1]  mean_weight      = 233.02
[2]  w_in(User)       = 75.22
[3]  w_in(Risk)       = 0.0
[4]  w_out(Threat)    = 0.0
[5]  w_out(Browser)   = 0.0
[6]  w_out(Email)     = 0.0
[7]  w_in(Security)   = 0.0
[8]  w_out(Bio)       = 0.12
[9]  w_in(Workstation)= 1612.93
[10] distinct_events  = 7
[11] recency_score    = 1.0
```

**tensor([7.0, 233.02, 75.22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 1612.93, 7.0, 1.0])**

---

## 🧠 Layer 2 — Neural Controlled Differential Equations (NCDE)

**File:** `core/state_inference.py` → `NCDEModel`, `CDEFunc`, `ReadoutHead`  
**Algorithm:** Neural CDE — `dZt = g_ϕ(Zt) dXt` integrated via **RK4** over a **Hermite cubic spline** path

### Architecture

```
Input: (1, seq_len=10, 12)  — rolling window of L1 embeddings
  │
  ▼  Hermite cubic spline interpolation → path X(t)
  │
  ▼  Initial Encoder: Linear(12→8) + Tanh  →  z0  (shape: 1×8)
  │
  ▼  CDE Integration via torchcde.cdeint [RK4, t=[0,1]]
  │     g_ϕ(Zt): Linear(8→64)→Tanh→Linear(64→64)→Tanh→Linear(64→96)→Tanh
  │     output shape: (1, 8×12) reshaped to (1, 8, 12) — the control matrix
  │
  ▼  z_final  (1, 8)
  │
  ▼  Readout Head: Linear(8→32)→ReLU→Linear(32→4)→Sigmoid
  │
Output: [Ct, Dt, Ht, At]  ∈ (0, 1)
```

- **Training:** MSE loss on `data/latent_states.csv`, Adam optimizer, 60 epochs
- **Stability Gate:** If window std < 0.05 (idle), NCDE is bypassed; state decays at 5%/tick toward user prior
- **Dt Override:** Demand is always computed from raw signals, not NCDE

### 📊 Sample Data — Layer 2 Input & Output

**Rolling window (10 ticks of 12-dim embeddings):**

```
Tick  [0]  [1]    [2]   [3]  ... [11]
  1    5   180.1  52.1  0.0  ...  0.85
  2    6   195.3  55.0  0.0  ...  0.90
  3    7   233.0  75.2  0.0  ...  1.00
  4    7   228.5  73.8  0.0  ...  0.95
  5    8   241.2  80.1  0.0  ...  0.92
  6    6   210.0  68.3  0.0  ...  0.88
  7    5   190.4  60.2  0.0  ...  0.80
  8    4   170.1  50.4  0.0  ...  0.75
  9    3   140.7  40.0  0.0  ...  0.60
 10    2    90.2  20.1  0.0  ...  0.40
```

**NCDE output (before personalisation):**

```python
z_out = model(x_seq)
# → tensor([[0.821, 0.312, 0.764, 0.031]])
# Ct=0.821  Dt=0.312  Ht=0.764  At=0.031
```

---

## 👤 Layer 3 — Personalised Latent Cognitive State Space

**File:** `core/state_inference.py` → `UserProfile`  
**Algorithm:** Bayesian MAP via Exponential Moving Average (EMA) + shrinkage blending

### Formulas

```
# Online MAP update
mu_new   = (1 - 0.05) * mu_old + 0.05 * z_observed
sigma_new= (1 - 0.05) * sigma_old + 0.05 * (z_obs - mu_new)^2

# Bayesian shrinkage blend
lambda   = min(0.85, 0.5 + n_obs / 200)
z_post   = lambda * z_model + (1 - lambda) * mu_user

# EMA smoothing (alpha=0.10) + slew-rate limit (MAX_SLEW=0.05/tick)
ema_new  = 0.90 * ema_old + 0.10 * z_post
```

### 📊 Sample Data — Layer 3 Personalisation

**User prior (calibrated from 120 historical observations):**

```
mu    = [Ct=0.84, Dt=0.28, Ht=0.76, At=0.02]
sigma = [Ct=0.06, Dt=0.09, Ht=0.04, At=0.01]
n_obs = 120
```

**Shrinkage computation:**

```
lambda = min(0.85, 0.5 + 120/200) = min(0.85, 1.1) = 0.85

z_model   = [0.821, 0.312, 0.764, 0.031]
mu_user   = [0.840, 0.280, 0.760, 0.020]

z_post    = 0.85*z_model + 0.15*mu_user
          = [0.824, 0.307, 0.763, 0.029]
```

**After EMA smoothing + slew-rate limiting:**

```
Capacity_Ct    = 0.819
Demand_Dt      = 0.298   ← always signal-driven, not from NCDE
Habits_Ht      = 0.761
Adversarial_At = 0.028
Reserve_CRGt   = 0.521   (= Ct - Dt)
```

---

## ⚖️ Layer 4 — Causal Risk Engine (Structural Causal Model)

**File:** `core/risk_forecaster.py` → `CausalRiskEngine`  
**Algorithm:** SCM — logistic regression on the logit scale, with **do-calculus** interventions

### Risk Formula

```
Risk = sigmoid( B_reserve * CRGt  +  B_habits * Ht  +  B_threat * At  +  bias )

Where CRGt = Ct - Dt  (Cognitive Reserve Gap)
```

### Causal Weights

| Weight | Default | Meaning |
|--------|---------|---------|
| B_reserve | −2.5 | Higher reserve → lower risk |
| B_habits | −1.8 | Better habits → lower risk |
| B_threat | +4.5 | Active attack → sharp spike |
| bias | +0.5 | Prior at neutral state |

> When `latent_states.csv` has ≥ 50 rows, weights are re-estimated via **OLS on logit(risk)**:  
> `B = (X'X)^-1 X'y`

### do-Calculus Interventions

| Intervention | Variable Modified | Effect |
|-------------|-----------------|--------|
| `reduce_notifications` | Dt × 0.70 | −30% demand |
| `reduce_meetings` | Dt × 0.80 | −20% demand |
| `pause_work` | Dt × 0.30 | −70% demand |
| `improve_sleep` | Ct + 0.15 | capacity boost |
| `security_training` | Ht + 0.20 | habit improvement |
| `adversarial_drill` | At × 0.50 | threat awareness |

### 📊 Sample Data — Layer 4 Risk Computation

**Input (from L3):**

```
Ct = 0.819,  Dt = 0.298,  Ht = 0.761,  At = 0.028
CRGt = Ct - Dt = 0.521
```

**Logit calculation (default weights):**

```
logit = (-2.5 * 0.521) + (-1.8 * 0.761) + (4.5 * 0.028) + 0.5
      = -1.303 + (-1.370) + 0.126 + 0.5
      = -2.047

Risk = sigmoid(-2.047) = 1 / (1 + e^2.047)
     = 1 / (1 + 7.745) = 11.4%
```

**Counterfactual — do(reduce_notifications):**

```
Dt_new = 0.298 * 0.70 = 0.209
CRGt_new = 0.819 - 0.209 = 0.610

logit = (-2.5 * 0.610) + (-1.8 * 0.761) + (4.5 * 0.028) + 0.5 + (-0.3)
      = -1.525 + (-1.370) + 0.126 + 0.5 - 0.3 = -2.569

Risk_new = sigmoid(-2.569) = 7.1%   →  Risk reduction: 4.3 pp
```

**All counterfactuals ranked:**

```
Intervention           Original  Mitigated  Reduction
pause_work              11.4%      5.2%       6.2 pp
reduce_notifications    11.4%      7.1%       4.3 pp
improve_sleep           11.4%      7.8%       3.6 pp
reduce_meetings         11.4%      8.5%       2.9 pp
security_training       11.4%      9.1%       2.3 pp
adversarial_drill       11.4%     11.4%       0.0 pp
```

---

## 📈 Layer 5 — Bayesian Temporal Hazard Model (Survival Analysis)

**File:** `core/risk_forecaster.py` → `BayesianHazardModel`  
**Algorithm:** Weibull Proportional Hazard + 500-sample Monte-Carlo posterior

### Weibull Formula

```
P(mistake in [t, t+τ] | risk_pct) = 1 - exp( -(τ / λ)^k )

λ = (100 - risk_pct) / 10      ← scale: higher risk = shorter time-to-mistake
k ~ N(1.5, 0.15)               ← shape prior (k>1: increasing hazard over time)

Posterior: draw 500 samples of k → distribution of P(mistake)
```

**Online update** (when real mistake observed):

```
k_new ≈ (mean_time_to_mistake / std_time_to_mistake) ^ 1.086
```

### 📊 Sample Data — Layer 5 Hazard Predictions

**Input:**

```
risk_pct = 11.4%
λ = (100 - 11.4) / 10 = 8.86
k_mean = 1.5,  k_std = 0.15
```

**500-sample Monte-Carlo output:**

```
Horizon     mean    median   std   lower_5%  upper_95%
  1 hour    2.1%    2.0%    0.4%    1.5%      2.9%
  24 hours 35.8%   35.1%   4.2%   28.9%     43.2%
   7 days  89.3%   89.7%   3.1%   83.6%     94.1%
```

**7-day trajectory forecast (from CyberRiskForecaster.forecast_trajectory):**

```
Day  Mean_Risk  P(mistake_24h)  Upper_Bound
  1   11.4%        35.8%          43.2%
  2   12.1%        37.5%          45.0%
  3   12.8%        39.2%          46.8%
  4   13.6%        41.0%          48.6%
  5   14.4%        42.8%          50.5%
  6   15.3%        44.7%          52.4%
  7   16.2%        46.6%          54.4%
```

*(Ct erodes by 0.015/day when Dt > 0.6 — sustained demand burnout trajectory)*

---

## 🔮 Layer 6 — True Digital Twin: Monte-Carlo Counterfactual Simulation

**File:** `core/simulation_engine.py` → `DigitalTwinSimulator`  
**Algorithm:** Vectorised Monte-Carlo — 500 parallel stochastic trajectories (NumPy arrays)

### Stochastic Dynamics per Step

```python
# For each of 500 trajectories simultaneously:
dt = clip(dt + drift_d + N(0, 0.025), 0.05, 1.0)
ct = clip(ct ± 0.015 + N(0, 0.015) + recovery, 0.05, ceiling)
ht = clip(ht + ht_gain + N(0, 0.010), 0.0, 1.0)
at = where(rand < spike_prob, 1.0, clip(at * 0.92 + N(0, 0.020), 0.0, 1.0))

risk = sigmoid(B_reserve*(ct-dt) + B_habits*ht + B_threat*at + bias) * 100
```

### Noise Parameters (calibrated from `latent_states.csv`)

| Variable | σ (default) | Dynamics |
|----------|------------|---------|
| Ct | 0.015 | Slow — drains when Dt > 0.65, recovers when idle |
| Dt | 0.025 | Fast — drifts per policy |
| Ht | 0.010 | Medium — very stable |
| At | 0.020 | Event-driven — decays at 0.92/step |

### Supported Policies

| Policy | Effect |
|--------|--------|
| `baseline` | Default slow cooling (Dt drift −0.015) |
| `increased_workload` | Dt +0.04, Ct drains faster |
| `reduced_workload` | Dt −0.03 |
| `reduce_meetings` | Dt −0.03 |
| `security_training` | Ht +0.01/step |
| `improve_sleep` | Ct recovery +0.02/step |
| `reduce_notifications` | Dt −0.02 |
| `pause_work` | Dt −0.05, Ct recovery +0.04 |
| `aging_progression` | Ct ceiling decays over time |
| `adversarial_drill` | At spike at 5% probability/step |

### 📊 Sample Data — Layer 6 Monte-Carlo Output

**Initial state:**

```
Ct=0.819, Dt=0.298, Ht=0.761, At=0.028
Risk_t0 = 11.4%
Policy  = "baseline"
```

**Aggregated output (500 trajectories × 30 steps):**

```
Step  Mean_Risk  Median  Std   Upper_95  Lower_5  Mean_Ct  Hazard_24h  Burnout%
   1   11.9%    11.8%   1.2%   14.0%     9.8%    0.8174     36.4%      0.0%
   3   12.6%    12.4%   1.5%   15.2%    10.1%    0.8148     38.1%      0.0%
   5   13.2%    13.0%   1.8%   16.4%    10.4%    0.8122     39.7%      0.0%
  10   15.1%    14.9%   2.4%   19.4%    11.5%    0.8048     43.5%      0.2%
  15   17.2%    17.0%   3.0%   22.7%    12.5%    0.7964     47.4%      0.6%
  20   19.6%    19.3%   3.6%   26.2%    13.6%    0.7872     51.5%      1.2%
  25   22.3%    22.0%   4.2%   30.0%    15.1%    0.7772     55.7%      2.0%
  30   25.3%    24.9%   4.9%   34.3%    16.8%    0.7663     60.2%      3.4%

Burnout_Horizon_Step = None  (mean risk never crosses 70%)
```

**Counterfactual delta — do(pause_work) vs baseline:**

```
Step  Baseline  Intervened  Delta (risk reduction)
   1   11.9%      8.1%         3.8 pp
   5   13.2%      7.6%         5.6 pp
  10   15.1%      7.0%         8.1 pp
  15   17.2%      6.5%        10.7 pp
  20   19.6%      6.1%        13.5 pp
  30   25.3%      5.4%        19.9 pp

Mean reduction over 30 steps = 9.4 pp
```

---

## 🔄 Full Pipeline Summary

```
Raw Sensors (10s tick)
      │
      ▼
[L1] BehaviourGraphEngine
     MultiDiGraph, 20 signals, 5-min TTL
     → 12-dim embedding tensor
      │
      ▼
[L2] NCDEModel
     dZt = g_ϕ(Zt)dXt, RK4, Hermite spline
     → [Ct, Dt, Ht, At] raw
      │
      ▼
[L3] UserProfile
     Bayesian MAP + EMA shrinkage + slew limit
     → [Ct, Dt, Ht, At] personalised (0–1)
      │
      ▼
[L4] CausalRiskEngine
     SCM: logit = B*CRG + B*Ht + B*At + bias
     + do-calculus interventions
     → risk_pct (0–100%)
      │
      ▼
[L5] BayesianHazardModel
     Weibull: P(mistake in τh) via 500 k-samples
     → {mean, median, std, lower_5, upper_95}
      │
      ▼
[L6] DigitalTwinSimulator
     500 vectorised MC trajectories × 30 steps
     → risk bands, burnout horizon, counterfactual delta
```

---

## ❌ Algorithms NOT Used Anywhere in LCDT

| Technique | Used? |
|-----------|-------|
| SVM | ❌ No |
| Random Forest / XGBoost | ❌ No |
| K-Nearest Neighbours | ❌ No |
| LSTM / GRU / Transformer | ❌ No |
| Reinforcement Learning | ❌ No |
| PCA / Dimensionality Reduction | ❌ No |
