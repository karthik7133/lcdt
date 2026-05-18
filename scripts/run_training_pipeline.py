"""
scripts/run_training_pipeline.py
=================================
Orchestrates the full three-phase dual-dataset NCDE training pipeline.

  Phase 1 — CERT Pre-training  (both heads, 4000 users, ~5M sequences)
  Phase 2 — Personal Fine-tune (Head A frozen, Head B + backbone on your data)
  Phase 3 — Scheduled Retrain  (runs automatically inside update_inference)

Usage (run once after receiving CERT v6.2):
  1. Generate CERT latent states first:
       python scripts/cert_feature_engineer.py --cert_dir path/to/cert_v6.2/

  2. Then run this pipeline:
       python scripts/run_training_pipeline.py

  Optional flags:
       --skip_phase1          Skip CERT pre-training (use existing cert weights)
       --cert_csv  PATH       Custom path to cert_latent_states.csv
       --personal_csv PATH    Custom path to latent_states.csv
       --phase1_epochs N      Epochs for Phase 1 (default: 100)
       --phase2_epochs N      Epochs for Phase 2 (default: 80)
"""

import os
import sys
import argparse
import torch

# Allow importing from project root
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_DIR)

from core.state_inference import (
    NCDEModel,
    train_ncde_cert,
    train_ncde_personal,
    WEIGHTS_PATH,
    CERT_WEIGHTS_PATH,
)

# Default CSV paths
DEFAULT_CERT_CSV     = os.path.join(PROJECT_DIR, "data", "cert_latent_states.csv")
DEFAULT_PERSONAL_CSV = os.path.join(PROJECT_DIR, "data", "latent_states.csv")


def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run dual-dataset three-phase NCDE training pipeline")
    parser.add_argument("--skip_phase1",    action="store_true",
                        help="Skip Phase 1 (use existing ncde_weights_cert.pt)")
    parser.add_argument("--cert_csv",       default=DEFAULT_CERT_CSV)
    parser.add_argument("--personal_csv",   default=DEFAULT_PERSONAL_CSV)
    parser.add_argument("--phase1_epochs",  type=int, default=100)
    parser.add_argument("--phase2_epochs",  type=int, default=80)
    args = parser.parse_args()

    model = NCDEModel()

    # ── PHASE 1: CERT Pre-training ────────────────────────────────────────────
    banner("PHASE 1 — CERT v6.2 Pre-training  (Head A + Head B, full loss)")

    if args.skip_phase1:
        if os.path.exists(CERT_WEIGHTS_PATH):
            print(f"[Phase 1] Skipped. Loading existing CERT weights: {CERT_WEIGHTS_PATH}")
            model.load_state_dict(torch.load(CERT_WEIGHTS_PATH, map_location="cpu"))
        else:
            print("[Phase 1] --skip_phase1 requested but no CERT weights found.")
            print(f"          Expected: {CERT_WEIGHTS_PATH}")
            print("          Run cert_feature_engineer.py first, then Phase 1.")
            sys.exit(1)
    else:
        if not os.path.exists(args.cert_csv):
            print(f"[Phase 1] CERT latent states not found: {args.cert_csv}")
            print("          Run first:")
            print("            python scripts/cert_feature_engineer.py --cert_dir <CERT_DIR>")
            sys.exit(1)

        success = train_ncde_cert(
            model,
            csv_path=args.cert_csv,
            epochs=args.phase1_epochs,
            lr=7e-4,
            verbose=True,
        )
        if not success:
            print("[Phase 1] CERT pre-training failed. Exiting.")
            sys.exit(1)

    # ── PHASE 2: Personal Fine-tuning ────────────────────────────────────────
    banner("PHASE 2 — Personal Fine-tuning  (Head A FROZEN, Head B + backbone)")

    if not os.path.exists(args.personal_csv):
        print(f"[Phase 2] Personal data not found: {args.personal_csv}")
        print("          Start the telemetry tracker to collect personal data first.")
        print("          CERT weights saved as starting point. Phase 2 will run")
        print("          automatically when the tracker next starts.")
        sys.exit(0)

    train_ncde_personal(
        model,
        csv_path=args.personal_csv,
        epochs=args.phase2_epochs,
        lr=1e-4,
        verbose=True,
    )

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    banner("TRAINING COMPLETE")
    print(f"  CERT weights (Phase 1) : {CERT_WEIGHTS_PATH}")
    print(f"  Production weights     : {WEIGHTS_PATH}")
    print()
    print("  Phase 3 (scheduled retraining) runs automatically inside")
    print("  the telemetry tracker every ~1 hour when 200+ new rows")
    print("  accumulate in latent_states.csv. Head A stays frozen.")
    print()
    print("  The telemetry tracker will now load production weights on next start.")


if __name__ == "__main__":
    main()
