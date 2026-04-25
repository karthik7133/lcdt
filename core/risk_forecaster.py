"""
LAYER 4 + LAYER 5: core/risk_forecaster.py
============================================
L4 — Causal Risk Engine (Structural Causal Model)
     Risk = f(CRG, Habits, Threat, Interventions)
     Supports do-calculus interventions with learned causal weights.

L5 — Bayesian Temporal Hazard Model (Survival Analysis)
     P(mistake in [t, t+tau])
     Weibull hazard with posterior uncertainty via Monte-Carlo.
"""

import numpy as np
import os

# ─────────────────────────────────────────────────────────────
# LAYER 4 — STRUCTURAL CAUSAL MODEL (SCM)
# Risk = sigma(B1*CRG + B2*Ht + B3*At + do(pi) + bias)
# ─────────────────────────────────────────────────────────────

# All supported do(pi) interventions with their causal effect
INTERVENTIONS = {
    # (target_var, delta_or_multiplier, is_multiplier, logit_shift)
    "reduce_notifications": ("Dt",  0.70, True,  -0.3),
    "security_training":    ("Ht",  0.20, False,  0.0),
    "improve_sleep":        ("Ct",  0.15, False, -0.2),
    "reduce_meetings":      ("Dt",  0.80, True,  -0.2),
    "pause_work":           ("Dt",  0.30, True,  -0.5),
    "adversarial_drill":    ("At",  0.50, True,   0.0),
}


class CausalRiskEngine:
    """
    Structural Causal Model:
      Risk = sigma(B1 * CRG + B2 * Ht + B3 * At + B4 * Sleep + B5 * Circadian + bias)

    Causal weights can be estimated from data via OLS on latent_states.csv,
    or kept at calibrated defaults when data is insufficient.
    """

    def __init__(self):
        # Default calibrated causal weights
        self.B = {
            "reserve":   -2.5,   # CRGt = Ct - Dt  (higher reserve → lower risk)
            "habits":    -1.8,   # Ht              (stronger habits → lower risk)
            "threat":     4.5,   # At              (active attack → sharp risk rise)
            "bias":       0.5,
        }
        self._try_fit_weights()

    def _try_fit_weights(self):
        """
        OLS estimation of causal weights from historical latent_states.csv.
        Runs only when > 50 samples available.
        """
        csv_path = os.path.join(os.path.dirname(__file__),
                                "..", "data", "latent_states.csv")
        try:
            import pandas as pd
            df = pd.read_csv(csv_path).dropna()
            if len(df) < 50:
                return

            # Build design matrix: [CRGt, Ht, At, bias]
            X = df[["CRGt", "Ht", "At"]].values
            X = np.column_stack([X, np.ones(len(X))])

            # Target: logit(risk_pct / 100)  — clipped to avoid inf
            p = np.clip(df["risk_pct"].values / 100.0, 0.01, 0.99)
            y = np.log(p / (1 - p))

            # OLS: w = (X'X)^-1 X'y
            w, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.B["reserve"] = float(w[0])
            self.B["habits"]  = float(w[1])
            self.B["threat"]  = float(w[2])
            self.B["bias"]    = float(w[3])
            print(f"[SCM] Causal weights estimated from {len(df)} observations.")
            print(f"      B_reserve={self.B['reserve']:.3f}  "
                  f"B_habits={self.B['habits']:.3f}  "
                  f"B_threat={self.B['threat']:.3f}")
        except Exception as e:
            print(f"[SCM] Using default weights ({e})")

    @staticmethod
    def _apply_intervention(ct, dt, ht, at, intervention: str):
        """Performs do(X=x) by surgically modifying the target variable."""
        if intervention not in INTERVENTIONS:
            return ct, dt, ht, at, 0.0

        var, delta, is_mult, logit_shift = INTERVENTIONS[intervention]
        if var == "Ct":
            ct = min(1.0, ct * delta if is_mult else ct + delta)
        elif var == "Dt":
            dt = max(0.0, dt * delta if is_mult else dt + delta)
        elif var == "Ht":
            ht = min(1.0, ht * delta if is_mult else ht + delta)
        elif var == "At":
            at = max(0.0, at * delta if is_mult else at + delta)

        return ct, dt, ht, at, logit_shift

    def calculate_causal_risk(self, latent_state: dict,
                               intervention: str = None) -> float:
        ct = latent_state["Capacity_Ct"]
        dt = latent_state["Demand_Dt"]
        ht = latent_state["Habits_Ht"]
        at = latent_state["Adversarial_At"]

        logit_shift = 0.0
        if intervention:
            ct, dt, ht, at, logit_shift = self._apply_intervention(
                ct, dt, ht, at, intervention)

        crg   = ct - dt
        logit = (self.B["reserve"] * crg +
                 self.B["habits"]  * ht  +
                 self.B["threat"]  * at  +
                 self.B["bias"]    +
                 logit_shift)

        return round(float(1 / (1 + np.exp(-logit))) * 100, 2)

    def list_interventions(self):
        return list(INTERVENTIONS.keys())


# ─────────────────────────────────────────────────────────────
# LAYER 5 — BAYESIAN TEMPORAL HAZARD MODEL (SURVIVAL ANALYSIS)
# P(mistake in [t, t+τ]) = 1 - exp(-(λ(Zt) * τ)^k)
# Posterior uncertainty via parameter sampling (MC approximation).
# ─────────────────────────────────────────────────────────────

class BayesianHazardModel:
    """
    Weibull Proportional Hazards model:
      h(t | Zt) = (k/lambda) * (t/lambda)^(k-1)  (hazard rate)
      S(t | Zt) = exp(-(t/lambda)^k)               (survival function)
      P(mistake in [0,tau]) = 1 - S(tau | Zt)

    lambda(Zt) is driven by the causal risk score.
    Posterior uncertainty: we treat k and lambda as uncertain with
    Gaussian priors and sample N times for CI estimation.
    """

    def __init__(self, k_mean: float = 1.5, k_std: float = 0.15,
                 n_samples: int = 500):
        # Prior on Weibull shape k (increasing hazard: k > 1 means risk grows over time)
        self.k_mean     = k_mean
        self.k_std      = k_std
        self.n_samples  = n_samples

        # Observed mistake history for online updating
        self._mistake_log: list = []

    def update_from_event(self, time_to_mistake_hours: float):
        """Call when a real security mistake is observed to update posteriors."""
        self._mistake_log.append(time_to_mistake_hours)
        if len(self._mistake_log) >= 5:
            # Method of moments update for Weibull k from observed data
            obs = np.array(self._mistake_log)
            # MoM: var/mean^2 = Gamma(1+2/k)/Gamma(1+1/k)^2 - 1
            # Approximate: k ≈ (mean/std)^1.086
            self.k_mean = float(np.clip((obs.mean() / (obs.std() + 1e-6)) ** 1.086,
                                        0.5, 5.0))
            print(f"[HazardModel] k updated to {self.k_mean:.3f} "
                  f"from {len(self._mistake_log)} events.")

    def predict(self, current_risk_pct: float, tau_hours: float) -> dict:
        """
        Returns posterior distribution of P(mistake in [t, t+tau]):
          mean, median, std, lower_5, upper_95
        """
        # Lambda: scale parameter inversely proportional to current risk
        # Higher risk → shorter expected time-to-mistake
        base_lambda = max(0.1, (100 - current_risk_pct) / 10.0)

        # Sample from posterior of k
        k_samples  = np.random.normal(self.k_mean, self.k_std, self.n_samples)
        k_samples  = np.clip(k_samples, 0.3, 5.0)

        # Weibull CDF: F(tau) = 1 - exp(-(tau/lambda)^k)
        probs = 1 - np.exp(-((tau_hours / base_lambda) ** k_samples))
        probs = np.clip(probs * 100, 0, 100)

        return {
            "mean":     round(float(probs.mean()),  2),
            "median":   round(float(np.median(probs)), 2),
            "std":      round(float(probs.std()),   2),
            "lower_5":  round(float(np.percentile(probs, 5)),  2),
            "upper_95": round(float(np.percentile(probs, 95)), 2),
        }


# ─────────────────────────────────────────────────────────────
# UNIFIED FORECASTER — public API
# ─────────────────────────────────────────────────────────────

class CyberRiskForecaster:
    """
    Public API combining:
      - Causal live risk score (L4)
      - Probabilistic mistake window (L5)
      - Counterfactual what-if analysis (L4 do-calculus)
    """

    def __init__(self):
        self.causal  = CausalRiskEngine()
        self.hazard  = BayesianHazardModel()

    # ── Live Risk (used by telemetry_tracker.py every 10s) ──
    def calculate_live_risk(self, latent_state: dict) -> float:
        return self.causal.calculate_causal_risk(latent_state)

    # ── Hazard: P(mistake in next N hours) ──
    def forecast_hazard(self, latent_state: dict,
                        window_hours: float = 24) -> dict:
        risk = self.calculate_live_risk(latent_state)
        return self.hazard.predict(risk, window_hours)

    # ── Counterfactual: do(intervention) ──
    def perform_counterfactual(self, latent_state: dict,
                               intervention: str) -> dict:
        original  = self.calculate_live_risk(latent_state)
        mitigated = self.causal.calculate_causal_risk(latent_state, intervention)
        return {
            "intervention":   intervention,
            "original_risk":  original,
            "mitigated_risk": mitigated,
            "reduction":      round(original - mitigated, 2),
            "hazard_24h_original":  self.hazard.predict(original,  24)["mean"],
            "hazard_24h_mitigated": self.hazard.predict(mitigated, 24)["mean"],
        }

    # ── All interventions at once ──
    def evaluate_all_interventions(self, latent_state: dict) -> list:
        results = []
        for iv in self.causal.list_interventions():
            results.append(self.perform_counterfactual(latent_state, iv))
        return sorted(results, key=lambda x: x["reduction"], reverse=True)

    # ── Trajectory forecast (used by Dashboard) ──
    def forecast_trajectory(self, current_state: dict, days_ahead: int = 7):
        forecasts = []
        state = dict(current_state)
        for d in range(1, days_ahead + 1):
            risk = self.calculate_live_risk(state)
            hazard = self.forecast_hazard(state, window_hours=24)
            # Burn-out drift: Ct erodes daily under sustained demand
            if state["Demand_Dt"] > 0.6:
                state["Capacity_Ct"] = max(0.1, state["Capacity_Ct"] - 0.015)
            state["Reserve_Gap_CRGt"] = state["Capacity_Ct"] - state["Demand_Dt"]
            forecasts.append({
                "day":        d,
                "mean_risk":  risk,
                "upper_bound": hazard["upper_95"],
                "lower_bound": hazard["lower_5"],
                "p_mistake_24h": hazard["mean"],
            })
        return forecasts

    # ── Record a real mistake (updates hazard posterior) ──
    def record_mistake(self, time_to_mistake_hours: float):
        self.hazard.update_from_event(time_to_mistake_hours)
