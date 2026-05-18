import pandas as pd
import numpy as np
import os

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

print("=" * 65)
print("  DATASET 1: CERT r4.2  (Phase 1 training — Ht & At)")
print("=" * 65)
cert_path = os.path.join(PROJECT, "data", "cert_latent_states.csv")
df_c = pd.read_csv(cert_path)
print(f"  File    : data/cert_latent_states.csv")
print(f"  Rows    : {len(df_c):,}  user-day sessions")
print(f"  Users   : {df_c['user_id'].nunique():,}  (CERT r4.2 employees)")
print(f"  Columns : {list(df_c.columns)}")
print()
print("  Sample (first 6 rows):")
print(df_c[["user_id","timestamp","Ct","Dt","Ht","At"]].head(6).to_string(index=False))
print()
print("  Value ranges (min / mean / max):")
for col in ["Ct","Dt","Ht","At"]:
    s = df_c[col]
    print(f"    {col}: min={s.min():.3f}  mean={s.mean():.3f}  max={s.max():.3f}  std={s.std():.3f}")
mal = (df_c["At"] >= 1.0).sum()
print(f"\n  Insider threat rows (At=1.0): {mal:,}  ({mal/len(df_c)*100:.2f}%)")
print(f"  High-risk rows   (At>=0.4) : {(df_c['At']>=0.4).sum():,}")

print()
print("=" * 65)
print("  DATASET 2: Personal Biometric Data  (Phase 2/3 — Ct & Dt)")
print("=" * 65)
pers_path = os.path.join(PROJECT, "data", "latent_states.csv")
df_p = pd.read_csv(pers_path)
df_p["time"] = pd.to_datetime(df_p["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
print(f"  File    : data/latent_states.csv")
print(f"  Rows    : {len(df_p):,}  sensor readings (10-second intervals)")
duration_hrs = len(df_p) * 10 / 3600
print(f"  Duration: ~{duration_hrs:.1f} hours of active tracker use")
print(f"  Columns : {list(df_p.columns)}")
print()
print("  Sample (first 6 rows):")
print(df_p[["time","Ct","Dt","Ht","At","risk_pct"]].head(6).to_string(index=False))
print()
print("  Value ranges (min / mean / max):")
for col in ["Ct","Dt","Ht","At","risk_pct"]:
    s = df_p[col]
    print(f"    {col:8s}: min={s.min():.3f}  mean={s.mean():.3f}  max={s.max():.3f}  std={s.std():.3f}")

print()
print("=" * 65)
print("  AUGMENTED TRAINING SET  (used only in-memory during training)")
print("=" * 65)
cert_seqs = max(0, len(df_c) - 10)
pers_seqs = max(0, len(df_p) - 10)
pers_aug  = pers_seqs * 9   # 1 real + 8 copies
print(f"  CERT sequences (Phase 1) : {cert_seqs:,}  (user-day windows, seq_len=10)")
print(f"    + 7x malicious augment : +{int(mal*7):,}  noisy malicious copies")
print(f"  Personal sequences (real): {pers_seqs:,}")
print(f"    + 8x AR(1) augment     : {pers_aug:,}  total (synthetic, in-RAM only)")
print()
print("  These are NOT saved to disk — regenerated fresh each retrain.")
