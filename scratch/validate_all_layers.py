"""
Full system validation test — all 6 layers.
Run from project root: python scratch/validate_all_layers.py
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

print("=" * 60)
print("CYBER WATCHDOG — SCIENTIFIC ARCHITECTURE VALIDATION")
print("=" * 60)

# ─── LAYER 1 + 2 + 3: NCDE Inference Engine ───────────────────
print("\n[L1/L2/L3] Initializing NeuralCDE engine (trains if needed)...")
from core.state_inference import NeuralCDEInference

engine = NeuralCDEInference()

# Simulate 15 signal packets (need > SEQ_LEN=10 to start NCDE)
signals = {
    'key_count': 45, 'mouse_entropy': 800, 'task_switches': 3,
    'notification_count': 2, 'workload_modifier': 0.3,
    'typing_error_rate': 0.08, 'insecure_http_hits': 0, 'webmail_hits': 0,
    'link_clicks': 1, 'email_frequency': 2, 'unknown_senders': 0,
    'avg_response_time': 300, 'low_strength_passwords': 0,
    'good_password_paste': 'FALSE', 'os_update_delayed': 'FALSE',
    'sleep_deficit': 'FALSE', 'vision_fatigue': 'FALSE',
    'phishing_clicked': 'FALSE', 'scam_credentials_given': 'FALSE',
    'hour_of_day': 14,
}

state = None
for i in range(15):
    state = engine.update_inference(signals)

print(f"\n[L1] Graph nodes active: {list(engine.graph.G.nodes())}")
print(f"[L1] Graph edges active: {engine.graph.G.number_of_edges()}")
print(f"[L2] NCDE output state:   {state}")
print(f"[L3] User prior mu:  {engine.profile.mu.round(3)}")
print(f"[L3] Observations:   {engine.profile.n_obs}")
assert state is not None, "FAILED: state is None"
assert 0 <= state['Capacity_Ct'] <= 1.0, "FAILED: Ct out of range"
assert 0 <= state['Demand_Dt']   <= 1.0, "FAILED: Dt out of range"
print("[L1/L2/L3] PASSED")


# ─── LAYER 4: Causal Risk Engine ──────────────────────────────
print("\n[L4] Testing Causal Risk Engine...")
from core.risk_forecaster import CyberRiskForecaster

forecaster = CyberRiskForecaster()
live_risk  = forecaster.calculate_live_risk(state)
print(f"[L4] Live risk:   {live_risk}%")
assert 0 <= live_risk <= 100, "FAILED: risk out of range"

print("[L4] Running all counterfactuals...")
all_cf = forecaster.evaluate_all_interventions(state)
for cf in all_cf:
    print(f"  do({cf['intervention']:25s}) -> {cf['original_risk']}% -> {cf['mitigated_risk']}%  (reduction: {cf['reduction']}%)")

assert all(0 <= cf['mitigated_risk'] <= 100 for cf in all_cf), "FAILED: counterfactual out of range"
print("[L4] PASSED")


# ─── LAYER 5: Bayesian Hazard Model ───────────────────────────
print("\n[L5] Testing Bayesian Hazard Model...")
hazard_24h  = forecaster.forecast_hazard(state, window_hours=24)
hazard_7d   = forecaster.forecast_hazard(state, window_hours=168)
print(f"[L5] P(mistake in 24h):  {hazard_24h}")
print(f"[L5] P(mistake in 7d):   {hazard_7d}")
assert "mean" in hazard_24h and "upper_95" in hazard_24h, "FAILED: missing CI keys"
assert hazard_7d["mean"] >= hazard_24h["mean"], "FAILED: 7d risk < 24h risk"
print("[L5] PASSED")


# ─── LAYER 6: Monte-Carlo Digital Twin ───────────────────────
print("\n[L6] Running Monte-Carlo simulation (1000 samples x 7 steps)...")
from core.simulation_engine import DigitalTwinSimulator

twin = DigitalTwinSimulator()
result = twin.run_monte_carlo_simulation(state, steps=7, policy="baseline")

print("[L6] Step-by-step results:")
for row in result:
    print(f"  Day {row['step']:2d} | Mean: {row['Mean_Risk']:5.1f}% "
          f"| 5-95 CI: [{row['Risk_Lower_5']}, {row['Risk_Upper_95']}]"
          f"| Burnout: {row['Burnout_Confidence']}%")

burnout = result[0].get("Burnout_Horizon_Step")
print(f"[L6] Burnout Horizon: Step {burnout}")

assert len(result) == 7, "FAILED: wrong number of steps"
assert "Risk_Upper_95" in result[0], "FAILED: missing CI"
assert "Burnout_Confidence" in result[0], "FAILED: missing burnout confidence"

# Counterfactual delta
print("\n[L6] Counterfactual: security_training vs baseline...")
delta = twin.counterfactual_delta(state, "security_training", days=7)
print(f"  Mean risk reduction from training: {delta['mean_reduction']}%")
print("[L6] PASSED")

print("\n" + "=" * 60)
print("ALL LAYERS VALIDATED SUCCESSFULLY")
print("=" * 60)
