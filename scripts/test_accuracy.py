"""
scripts/test_accuracy.py
========================
Model accuracy test using held-out samples from BOTH datasets.

Metrics reported:
  1. Tolerance Accuracy  -- % of predictions within 5%/10%/15% of actual value
  2. Risk Level Accuracy -- % of predictions in the correct Low/Med/High bucket
  3. R2 (for completeness) -- correlation between predicted and actual

Usage:
  python scripts/test_accuracy.py
  python scripts/test_accuracy.py --cert_samples 20000
"""

import os, sys, argparse, datetime
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.state_inference import (
    NCDEModel, WEIGHTS_PATH, CERT_WEIGHTS_PATH,
    _load_cert_training_data, _load_personal_training_data,
)

PROJECT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CERT_CSV     = os.path.join(PROJECT, "data", "cert_latent_states.csv")
PERSONAL_CSV = os.path.join(PROJECT, "data", "latent_states.csv")

# ── Risk bucket definitions ────────────────────────────────────────────────
#   Each dimension gets bucketed into Low / Medium / High
#   Accuracy = did model predict the CORRECT bucket?

def ct_bucket(v):
    """Cognitive Capacity: High = good, Low = fatigued"""
    if v >= 0.70: return "High"
    if v >= 0.40: return "Medium"
    return "Low"

def dt_bucket(v):
    """Cognitive Demand: Low = relaxed, High = overloaded"""
    if v >= 0.60: return "High"
    if v >= 0.30: return "Medium"
    return "Low"

def ht_bucket(v):
    """Habit Health: High = good habits, Low = risky habits"""
    if v >= 0.50: return "High"
    if v >= 0.20: return "Medium"
    return "Low"

def at_bucket(v):
    """Adversarial Risk: Low = safe, High = threat detected"""
    if v >= 0.60: return "High"
    if v >= 0.20: return "Medium"
    return "Low"

BUCKETERS = [ct_bucket, dt_bucket, ht_bucket, at_bucket]
DIMS      = ["Ct (Capacity)", "Dt (Demand) ", "Ht (Habits) ", "At (Adversarial)"]

# ── Metrics ───────────────────────────────────────────────────────────────

def tolerance_accuracy(y_true, y_pred, tol=0.10):
    """% of predictions within `tol` absolute error of the true value."""
    return float(np.mean(np.abs(y_true - y_pred) <= tol)) * 100

def bucket_accuracy(y_true, y_pred, bucket_fn):
    """% of predictions in the same risk bucket as the true value."""
    correct = sum(bucket_fn(a) == bucket_fn(p) for a, p in zip(y_true, y_pred))
    return correct / len(y_true) * 100

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def bar(pct, width=25):
    filled = max(0, min(width, int(pct / 100 * width)))
    return "[" + "#" * filled + "-" * (width - filled) + f"]  {pct:.1f}%"

def grade_acc(acc):
    if acc >= 90: return "A+  Excellent"
    if acc >= 80: return "A   Good"
    if acc >= 70: return "B   Fair"
    if acc >= 60: return "C   Weak"
    return              "F   Needs improvement"

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).numpy()

def load_cert(n, seed=99):
    np.random.seed(seed)
    return _load_cert_training_data(CERT_CSV, seq_len=10, max_sequences=n)

def load_personal(test_frac=0.30, seed=42):
    np.random.seed(seed)
    X, Y = _load_personal_training_data(PERSONAL_CSV, seq_len=10)
    if X is None: return None, None
    perm = np.random.permutation(len(X))
    X, Y = X[perm], Y[perm]
    split = int(len(X) * (1 - test_frac))
    return X[split:], Y[split:]

def print_section(title):
    print("\n" + "=" * 66)
    print("  " + title)
    print("=" * 66)

def print_dim_results(label, y_true, y_pred, bucket_fn, dim_idx):
    yt = y_true[:, dim_idx]
    yp = y_pred[:, dim_idx]

    acc_05  = tolerance_accuracy(yt, yp, tol=0.05)
    acc_10  = tolerance_accuracy(yt, yp, tol=0.10)
    acc_15  = tolerance_accuracy(yt, yp, tol=0.15)
    bacc    = bucket_accuracy(yt, yp, bucket_fn)
    r2v     = r2_score(yt, yp)
    rv      = rmse(yt, yp)

    print(f"\n  [{label}]")
    print(f"    Risk Level Accuracy   {bar(bacc)}   {grade_acc(bacc)}")
    print(f"    Tolerance +/-5%       {bar(acc_05)}")
    print(f"    Tolerance +/-10%      {bar(acc_10)}")
    print(f"    Tolerance +/-15%      {bar(acc_15)}")
    print(f"    R2 (correlation)      {r2v:+.4f}   RMSE={rv:.4f}")

    # Bucket breakdown
    buckets = ["Low", "Medium", "High"]
    print(f"\n    Risk Bucket Breakdown (actual vs predicted):")
    print(f"    {'Actual':>10}  {'Predicted':>10}  {'Match':>6}")
    sample_idx = np.random.choice(len(yt), min(10, len(yt)), replace=False)
    for i in sorted(sample_idx):
        a_bucket = bucket_fn(yt[i])
        p_bucket = bucket_fn(yp[i])
        match    = "OK" if a_bucket == p_bucket else "MISS"
        print(f"    {yt[i]:>7.4f} ({a_bucket:>6})  {yp[i]:>7.4f} ({p_bucket:>6})  {match:>6}")

    return bacc, acc_10

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert_samples",  type=int,   default=10000)
    parser.add_argument("--personal_frac", type=float, default=0.30)
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*66}")
    print(f"  LCDT MODEL ACCURACY REPORT")
    print(f"  Generated : {ts}")
    print(f"{'='*66}")
    print("""
  METRIC EXPLANATION
  ------------------
  R2              = How well predictions CORRELATE with actuals (0-1 scale)
                    R2=0.96 means 96% of the variance is explained.
                    Does NOT tell you how many predictions are "correct".

  Risk Level      = Prediction mapped to Low / Medium / High bucket.
  Accuracy (main)   "How many of the 10,000 samples did the model
                     put in the CORRECT risk level?"
                    This is your real accuracy number.

  Tolerance +/-X% = "Prediction within X units of actual" counts as correct.
                    e.g. actual=0.85, pred=0.82 -> within 5% = CORRECT
    """)

    # Load models
    cert_model = NCDEModel()
    cert_model.load_state_dict(torch.load(CERT_WEIGHTS_PATH, map_location="cpu"))
    prod_model = NCDEModel()
    prod_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    print(f"  [OK] CERT weights loaded    : ncde_weights_cert.pt")
    print(f"  [OK] Production weights     : ncde_weights.pt")

    # ── TEST A: CERT ──────────────────────────────────────────────────────────
    print_section(f"TEST A -- CERT r4.2  ({args.cert_samples:,} random held-out sequences)")
    print(f"  Using CERT weights (pre-personalisation). Seed=99 (held-out, not training data)")

    X_c, Y_c = load_cert(args.cert_samples, seed=99)
    pred_c    = predict(cert_model, X_c)
    y_c       = Y_c.numpy()

    cert_baccs = []
    cert_taccs = []
    for i, (lbl, bkt) in enumerate(zip(DIMS, BUCKETERS)):
        bacc, tacc = print_dim_results(lbl, y_c, pred_c, bkt, i)
        cert_baccs.append(bacc)
        cert_taccs.append(tacc)

    cert_avg_bacc = np.mean(cert_baccs)
    cert_avg_tacc = np.mean(cert_taccs)

    # ── TEST B: Personal ──────────────────────────────────────────────────────
    print_section("TEST B -- Personal latent_states.csv  (30% held-out)")
    print(f"  Using production weights (personalised Head B).")

    X_p, Y_p = load_personal(test_frac=args.personal_frac)
    if X_p is None:
        print("  [!!] Not enough personal data.")
        pers_baccs = [0, 0]
        pers_taccs = [0, 0]
    else:
        pred_p    = predict(prod_model, X_p)
        y_p       = Y_p.numpy()
        print(f"  Test set: {len(X_p)} sequences")
        pers_baccs = []
        pers_taccs = []
        for i, (lbl, bkt) in enumerate(zip(DIMS[:2], BUCKETERS[:2])):
            bacc, tacc = print_dim_results(lbl, y_p[:, :2], pred_p[:, :2], bkt, i)
            pers_baccs.append(bacc)
            pers_taccs.append(tacc)
    pers_avg_bacc = np.mean(pers_baccs)

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────────
    print_section("FINAL ACCURACY SUMMARY")
    print(f"""
  CERT r4.2 -- 10,000 held-out samples
  ─────────────────────────────────────────────────────────────
  Dimension         Risk Level Accuracy    +/-10% Tol Accuracy
  Ct (Capacity)     {cert_baccs[0]:>7.1f}%  {bar(cert_baccs[0], 15)}    {cert_taccs[0]:.1f}%
  Dt (Demand)       {cert_baccs[1]:>7.1f}%  {bar(cert_baccs[1], 15)}    {cert_taccs[1]:.1f}%
  Ht (Habits)       {cert_baccs[2]:>7.1f}%  {bar(cert_baccs[2], 15)}    {cert_taccs[2]:.1f}%
  At (Adversarial)  {cert_baccs[3]:>7.1f}%  {bar(cert_baccs[3], 15)}    {cert_taccs[3]:.1f}%
  ─────────────────────────────────────────────────────────────
  OVERALL CERT      {cert_avg_bacc:>7.1f}%  {bar(cert_avg_bacc, 15)}    {cert_avg_tacc:.1f}%

  Personal Data -- {0 if X_p is None else len(X_p)} held-out samples
  ─────────────────────────────────────────────────────────────
  Ct (Capacity)     {pers_baccs[0]:>7.1f}%  {bar(pers_baccs[0], 15)}    {pers_taccs[0]:.1f}%
  Dt (Demand)       {pers_baccs[1]:>7.1f}%  {bar(pers_baccs[1], 15)}    {pers_taccs[1]:.1f}%
  ─────────────────────────────────────────────────────────────
  OVERALL PERSONAL  {pers_avg_bacc:>7.1f}%  {bar(pers_avg_bacc, 15)}
    """)

    print(f"  COMBINED MODEL ACCURACY  (CERT + Personal average)")
    combined = (cert_avg_bacc * 4 + pers_avg_bacc * 2) / 6
    print(f"  {bar(combined, 30)}  {grade_acc(combined)}")
    print(f"\n  Completed at : {datetime.datetime.now().strftime('%H:%M:%S')}")
    print("=" * 66)

if __name__ == "__main__":
    main()
