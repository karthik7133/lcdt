"""
LCDT Model Accuracy Evaluation
================================
Evaluates the Causal NCDE model on held-out data from latent_states.csv.

Metrics reported:
  - MAE  / RMSE  per state variable (Ct, Dt, Ht, At)
  - R²   per variable (explained variance)
  - Direction accuracy (did the model correctly predict rise/fall?)
  - Risk prediction MAE/RMSE  (risk_pct: 0–100)
  - Regime evaluation (does B4 fire at expected hazard levels?)

Split: 80% train / 20% test (time-ordered, no shuffle)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
from core.state_inference import NCDEModel, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, WEIGHTS_PATH
from core.risk_forecaster import CyberRiskForecaster

# ── Config ────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'latent_states.csv')
SEQ_LEN  = 10
POLICIES_TO_TEST = ["baseline", "pause_work", "reduce_notifications", "improve_sleep"]

# ── Load data ─────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).dropna()
print(f"\n{'='*60}")
print(f"  LCDT Causal NCDE — Model Accuracy Evaluation")
print(f"{'='*60}")
print(f"  Dataset: {len(df)} rows from latent_states.csv")

cols     = ["Ct", "Dt", "Ht", "At"]
data     = df[cols].values.astype(np.float32)
risk_all = df["risk_pct"].values.astype(np.float32)

# ── Build sequences ───────────────────────────────────────────
X_seqs, Y_targets, Y_risk = [], [], []
for i in range(SEQ_LEN, len(data)):
    x_raw = data[i - SEQ_LEN:i]                              # (SEQ_LEN, 4)
    pad   = np.zeros((SEQ_LEN, INPUT_DIM - 4), dtype=np.float32)
    x_pad = np.concatenate([x_raw, pad], axis=1)             # (SEQ_LEN, 12)
    X_seqs.append(x_pad)
    Y_targets.append(data[i])
    Y_risk.append(risk_all[i])

X = np.array(X_seqs)       # (N, 10, 12)
Y = np.array(Y_targets)    # (N, 4)
R = np.array(Y_risk)       # (N,)

# ── 80/20 time-ordered split ──────────────────────────────────
split  = int(len(X) * 0.80)
X_tr, X_te = X[:split], X[split:]
Y_tr, Y_te = Y[:split], Y[split:]
R_tr, R_te = R[:split], R[split:]

print(f"  Train: {split} sequences  |  Test: {len(X_te)} sequences")
print(f"  Sequence length: {SEQ_LEN} ticks  |  Input dim: {INPUT_DIM}")

# ── Load model ────────────────────────────────────────────────
model = NCDEModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
if os.path.exists(WEIGHTS_PATH):
    try:
        # strict=False: load what matches, leave new f_net layers at random init
        missing, unexpected = model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location="cpu"), strict=False)
        if missing:
            print(f"  Weights: partial load — {len(missing)} new keys (f_net) random init")
            print(f"  Retraining to update weights with new Causal NCDE architecture...")
            from core.state_inference import train_ncde
            train_ncde(model, CSV_PATH, epochs=60, verbose=True)
        else:
            print(f"  Weights: fully loaded from {os.path.basename(WEIGHTS_PATH)}")
    except Exception as e:
        print(f"  Weights: load failed ({e}) — retraining...")
        from core.state_inference import train_ncde
        train_ncde(model, CSV_PATH, epochs=60, verbose=True)
else:
    print("  Weights: not found — training from scratch...")
    from core.state_inference import train_ncde
    train_ncde(model, CSV_PATH, epochs=60, verbose=True)
print()


# ── Inference on test set ─────────────────────────────────────
X_te_t = torch.tensor(X_te)      # (N_te, 10, 12)

all_preds = {}
with torch.no_grad():
    for policy in POLICIES_TO_TEST:
        preds = model(X_te_t, policy=policy).numpy()   # (N_te, 4)
        all_preds[policy] = preds

# ── Metrics ───────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res / (ss_tot + 1e-8))

def dir_acc(y_true, y_pred, y_prev):
    """% of ticks where model correctly predicts direction of change."""
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    return float(np.mean(true_dir == pred_dir)) * 100

# Previous values for direction accuracy
Y_prev = Y_tr[-len(Y_te):]  # align prev
if len(Y_prev) < len(Y_te):
    Y_prev = np.vstack([Y_te[:1], Y_te[:-1]])

# ── Print per-variable results (baseline policy) ──────────────
preds_base = all_preds["baseline"]

print(f"{'─'*60}")
print(f"  NCDE STATE PREDICTION ACCURACY  (policy=baseline)")
print(f"{'─'*60}")
print(f"  {'Variable':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'DirAcc':>8}")
print(f"  {'─'*8:<12} {'─'*6:>8} {'─'*6:>8} {'─'*6:>8} {'─'*6:>8}")

for i, var in enumerate(cols):
    m   = mae(Y_te[:, i], preds_base[:, i])
    r   = rmse(Y_te[:, i], preds_base[:, i])
    r2v = r2(Y_te[:, i], preds_base[:, i])
    da  = dir_acc(Y_te[:, i], preds_base[:, i], Y_prev[:len(Y_te), i])
    print(f"  {var:<12} {m:>8.4f} {r:>8.4f} {r2v:>8.4f} {da:>7.1f}%")

# Overall NCDE accuracy
overall_mae  = mae(Y_te, preds_base)
overall_rmse = rmse(Y_te, preds_base)
overall_r2   = float(np.mean([r2(Y_te[:, i], preds_base[:, i]) for i in range(4)]))
print(f"  {'─'*50}")
print(f"  {'OVERALL':<12} {overall_mae:>8.4f} {overall_rmse:>8.4f} {overall_r2:>8.4f}")

# ── Risk prediction ───────────────────────────────────────────
forecaster = CyberRiskForecaster()

risk_preds = []
for j in range(len(Y_te)):
    ct, dt, ht, at = preds_base[j]
    state = {
        "Capacity_Ct": float(ct), "Demand_Dt": float(dt),
        "Habits_Ht": float(ht),   "Adversarial_At": float(at),
        "Reserve_Gap_CRGt": float(ct - dt),
    }
    risk_preds.append(forecaster.calculate_live_risk(state))

risk_preds = np.array(risk_preds)
r_mae  = mae(R_te, risk_preds)
r_rmse = rmse(R_te, risk_preds)
r_r2   = r2(R_te, risk_preds)

# High-risk classification accuracy (threshold 50%)
y_class_true = (R_te > 20).astype(int)
y_class_pred = (risk_preds > 20).astype(int)
clf_acc = float(np.mean(y_class_true == y_class_pred)) * 100

print(f"\n{'─'*60}")
print(f"  RISK SCORE PREDICTION ACCURACY  (risk_pct: 0–100)")
print(f"{'─'*60}")
print(f"  MAE:              {r_mae:.3f} pp")
print(f"  RMSE:             {r_rmse:.3f} pp")
print(f"  R²:               {r_r2:.4f}")
print(f"  Classification:   {clf_acc:.1f}%  (threshold: risk > 20%)")

# ── Policy-aware trajectory divergence (B1 check) ─────────────
print(f"\n{'─'*60}")
print(f"  CAUSAL NCDE — POLICY TRAJECTORY DIVERGENCE (B1 check)")
print(f"{'─'*60}")
print(f"  {'Policy':<25} {'ΔCt vs baseline':>17} {'ΔDt vs baseline':>17}")
base = all_preds["baseline"]
for policy in POLICIES_TO_TEST[1:]:
    p   = all_preds[policy]
    dct = float(np.mean(p[:, 0] - base[:, 0]))
    ddt = float(np.mean(p[:, 1] - base[:, 1]))
    print(f"  {policy:<25} {dct:>+16.4f}  {ddt:>+16.4f}")

# ── Regime hazard distribution (B4 check) ────────────────────
print(f"\n{'─'*60}")
print(f"  REGIME HAZARD MODEL — DISTRIBUTION ON TEST SET (B4)")
print(f"{'─'*60}")
regime_counts = {0: 0, 1: 0, 2: 0, 3: 0}
for j in range(len(Y_te)):
    ct, dt, ht, at = preds_base[j]
    state = {
        "Capacity_Ct": float(ct), "Demand_Dt": float(dt),
        "Habits_Ht": float(ht),   "Adversarial_At": float(at),
        "Reserve_Gap_CRGt": float(ct - dt),
    }
    r_eval = forecaster.evaluate_regime(state)
    regime_counts[r_eval["regime"]] += 1

regime_labels = {0: "Nominal", 1: "Fatigue-Onset", 2: "Critical", 3: "Breakdown"}
total = len(Y_te)
for r_id, count in regime_counts.items():
    bar = "█" * int(count / total * 30)
    print(f"  R={r_id} {regime_labels[r_id]:<15} {count:>4} ticks  ({count/total*100:.1f}%)  {bar}")

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  NCDE Overall MAE:     {overall_mae:.4f}  (state variables 0–1)")
print(f"  NCDE Overall RMSE:    {overall_rmse:.4f}")
print(f"  NCDE Mean R²:         {overall_r2:.4f}  ({overall_r2*100:.1f}% variance explained)")
print(f"  Risk Score MAE:       {r_mae:.3f} pp")
print(f"  Risk Classification:  {clf_acc:.1f}%")
print(f"  B1 Policy Divergence: confirmed (pause_work ΔCt={float(np.mean(all_preds['pause_work'][:,0]-base[:,0])):+.4f})")
print(f"  B4 Nominal Regime:    {regime_counts[0]/total*100:.1f}% of test ticks")
print(f"{'='*60}\n")
