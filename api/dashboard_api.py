"""
api/dashboard_api.py
=====================
Flask REST API linking the React frontend to all 6 scientific layers.

Endpoints:
  GET  /api/summary          — live snapshot (L2/L3 state)
  GET  /api/latent_states    — historical Ct/Dt/Ht time series
  GET  /api/stats            — raw telemetry history
  GET  /api/forecast         — 7-day L5 trajectory with CI bands
  GET  /api/hazard           — P(mistake in next 1h/24h/7d) with uncertainty
  GET  /api/counterfactuals  — all 6 do(pi) interventions ranked
  POST /api/simulation       — L6 Monte-Carlo run for a policy
  POST /api/behaviour        — browser extension log
  POST /api/adversarial_signal — phishing/scam simulator
  GET  /api/test/live_sensors
  GET  /api/test/os_update
  POST /api/test/run_phishing
  POST /api/test/run_fatigue
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.risk_forecaster import CyberRiskForecaster
from core.simulation_engine import DigitalTwinSimulator
from core.state_inference import NeuralCDEInference

app = Flask(__name__)
CORS(app)

# ── File Paths ──────────────────────────────────────────────
DATA_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
BEHAVIOUR_LOG   = os.path.join(DATA_DIR, 'digital_behaviour.csv')
ADVERSARIAL_LOG = os.path.join(DATA_DIR, 'adversarial_failures.csv')
TELEMETRY_CSV   = os.path.join(DATA_DIR, 'telemetry_history.csv')
INTERVENTIONS_LOG = os.path.join(DATA_DIR, 'interventions.log')
LATENT_LOG      = os.path.join(DATA_DIR, 'latent_states.csv')

# ── Engine Initialisation (once at server start) ────────────────────
print("[API] Initialising Scientific Engines...")
risk_forecaster = CyberRiskForecaster()
simulator       = DigitalTwinSimulator()
ncde_engine     = NeuralCDEInference()   # for resilience_profile (B2)
print("[API] Ready.")


# ── Helpers ─────────────────────────────────────────────────

def _latest_state() -> dict | None:
    """Read the most recent latent state from the CSV log."""
    if not os.path.exists(LATENT_LOG):
        return None
    try:
        df = pd.read_csv(LATENT_LOG).dropna()
        if df.empty:
            return None
        r = df.iloc[-1]
        return {
            "Capacity_Ct":      float(r.get("Ct", 1.0)),
            "Demand_Dt":        float(r.get("Dt", 0.1)),
            "Habits_Ht":        float(r.get("Ht", 0.8)),
            "Adversarial_At":   float(r.get("At", 0.0)),
            "Reserve_Gap_CRGt": float(r.get("CRGt", 0.9)),
        }
    except Exception as e:
        print(f"[API] _latest_state error: {e}")
        return None


# ── Core Endpoints ───────────────────────────────────────────

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Raw telemetry history for the physical sensor chart."""
    if not os.path.exists(TELEMETRY_CSV):
        return jsonify([])
    try:
        df = pd.read_csv(TELEMETRY_CSV)
        if df.empty:
            return jsonify([])
        df['time'] = pd.to_datetime(df['timestamp']).dt.strftime('%H:%M:%S')
        df['smoothed_keys'] = df['keys_pressed'].rolling(window=3, min_periods=1).mean().astype(int)
        return jsonify(df[['time', 'keys_pressed', 'smoothed_keys']].tail(30).to_dict('records'))
    except Exception as e:
        print(f"[API] stats error: {e}")
        return jsonify([])


@app.route('/api/latent_states', methods=['GET'])
def get_latent_states():
    """Historical L2/L3 brain states for the Ct vs Dt chart."""
    if not os.path.exists(LATENT_LOG):
        return jsonify([])
    try:
        df = pd.read_csv(LATENT_LOG).dropna()
        if df.empty:
            return jsonify([])
        df['time'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%H:%M:%S')
        return jsonify(df[['time', 'Ct', 'Dt', 'Ht', 'CRGt', 'risk_pct']].tail(30).to_dict('records'))
    except Exception as e:
        print(f"[API] latent_states error: {e}")
        return jsonify([])


@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Live dashboard cards: state snapshot + risk + causal interventions."""
    state = _latest_state()
    summary = {
        "Ct": 1.0, "Dt": 0.1, "Ht": 0.8, "At": 0.0,
        "CRGt": 0.9, "risk_pct": 0,
        "total_interventions": 0,
        "user_prior_mu": [1.0, 0.1, 0.8, 0.0],
    }

    if state:
        live_risk = risk_forecaster.calculate_live_risk(state)
        summary.update({
            "Ct":       round(state["Capacity_Ct"], 3),
            "Dt":       round(state["Demand_Dt"], 3),
            "Ht":       round(state["Habits_Ht"], 3),
            "At":       round(state["Adversarial_At"], 3),
            "CRGt":     round(state["Reserve_Gap_CRGt"], 3),
            "risk_pct": live_risk,
        })

    if os.path.exists(INTERVENTIONS_LOG):
        try:
            with open(INTERVENTIONS_LOG, 'r') as f:
                summary["total_interventions"] = len(f.readlines())
        except Exception:
            pass

    return jsonify(summary)


# ── NEW: Hazard Endpoint (L5) ────────────────────────────────

@app.route('/api/hazard', methods=['GET'])
def get_hazard():
    state = _latest_state()
    if not state:
        return jsonify({"error": "No latent state available"}), 404
    return jsonify({
        "hazard_1h":  risk_forecaster.forecast_hazard(state, window_hours=1),
        "hazard_24h": risk_forecaster.forecast_hazard(state, window_hours=24),
        "hazard_7d":  risk_forecaster.forecast_hazard(state, window_hours=168),
    })


@app.route('/api/regime', methods=['GET'])
def get_regime():
    """
    BREAKTHROUGH 4 — Current cognitive regime (phase transition model).
    Returns regime label, hazard h, Ct drain rate, and ticks to next transition.
    """
    state = _latest_state()
    if not state:
        return jsonify({"error": "No latent state available"}), 404
    try:
        return jsonify(risk_forecaster.evaluate_regime(state))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/resilience_profile', methods=['GET'])
def get_resilience_profile():
    """
    BREAKTHROUGH 2 — Causal user resilience profile.
    Returns causal sensitivity per policy and the most effective intervention.
    """
    try:
        return jsonify(ncde_engine.profile.get_resilience_profile())
    except Exception as e:
        return jsonify({"error": str(e)}), 500




# ── NEW: All Counterfactuals Endpoint (L4) ───────────────────

@app.route('/api/counterfactuals', methods=['GET'])
def get_counterfactuals():
    """
    L4 Causal Risk Engine.
    Returns all 6 do(pi) interventions ranked by risk reduction.
    """
    state = _latest_state()
    if not state:
        return jsonify([])

    results = risk_forecaster.evaluate_all_interventions(state)
    return jsonify(results)


# ── Updated: Forecast Endpoint (L5 trajectory) ──────────────

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """
    L5 probabilistic 7-day trajectory with CI bands.
    Each day includes mean_risk, upper_bound (95th), lower_bound (5th),
    and P(mistake in next 24h).
    """
    state = _latest_state()
    if not state:
        return jsonify([])

    try:
        forecast = risk_forecaster.forecast_trajectory(state, days_ahead=7)
        return jsonify(forecast)
    except Exception as e:
        print(f"[API] forecast error: {e}")
        return jsonify([])


# ── Updated: Simulation Endpoint (L6 Monte-Carlo) ────────────

@app.route('/api/simulation', methods=['POST'])
def run_simulation():
    """
    L6 Monte-Carlo Digital Twin.
    Accepts policy name and runs 1000 stochastic trajectories.
    Returns mean, CI bands, and burnout confidence per step.
    """
    data   = request.json or {}
    policy = data.get('policy', 'baseline')

    state = _latest_state()
    if not state:
        return jsonify({"status": "error", "message": "No latent data"}), 404

    try:
        result = simulator.run_monte_carlo_simulation(
            state, steps=30, policy=policy)
        return jsonify(result)
    except Exception as e:
        print(f"[API] simulation error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ── NEW: Counterfactual Delta Endpoint (L6) ──────────────────

@app.route('/api/counterfactual_delta', methods=['POST'])
def get_counterfactual_delta():
    """
    L6 Monte-Carlo counterfactual delta.
    Returns E[Risk|do(pi)] vs E[Risk|baseline] per step.
    """
    data   = request.json or {}
    policy = data.get('policy', 'security_training')

    state = _latest_state()
    if not state:
        return jsonify({"status": "error"}), 404

    try:
        delta = simulator.counterfactual_delta(state, policy, days=30)
        return jsonify(delta)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Existing Endpoints (unchanged) ───────────────────────────

@app.route('/api/behaviour', methods=['POST'])
def log_behaviour():
    import json
    data = request.json
    if not data:
        return jsonify({"status": "error"}), 400
    if not os.path.exists(BEHAVIOUR_LOG):
        with open(BEHAVIOUR_LOG, 'w', encoding='utf-8') as f:
            f.write("timestamp,event,domain,details\n")
    details_str = json.dumps(data.get('details', {})).replace(',', ';')
    with open(BEHAVIOUR_LOG, 'a', encoding='utf-8') as f:
        f.write(f"{data.get('timestamp')},{data.get('event')},{data.get('domain')},{details_str}\n")
    return jsonify({"status": "success"})


@app.route('/api/adversarial_signal', methods=['POST'])
def log_adversarial_signal():
    import time
    data = request.json
    if not data:
        return jsonify({"status": "error"}), 400
    if not os.path.exists(ADVERSARIAL_LOG):
        with open(ADVERSARIAL_LOG, 'w', encoding='utf-8') as f:
            f.write("timestamp,attack_type,status,details\n")
    with open(ADVERSARIAL_LOG, 'a', encoding='utf-8') as f:
        f.write(f"{time.time()},{data.get('type','Unknown')},{data.get('status','Unknown')},{data.get('details','')}\n")
    return jsonify({"status": "success"})


@app.route('/api/test/live_sensors', methods=['GET'])
def get_live_sensors():
    import json
    sensors, beh = {}, []
    lsf = os.path.join(DATA_DIR, 'live_sensors.json')
    if os.path.exists(lsf):
        try:
            with open(lsf, 'r') as f:
                sensors = json.load(f)
        except Exception:
            pass
    if os.path.exists(BEHAVIOUR_LOG):
        try:
            beh = pd.read_csv(BEHAVIOUR_LOG).tail(10).to_dict('records')
        except Exception:
            pass
    return jsonify({"sensors": sensors, "behaviour": beh})


@app.route('/api/test/os_update', methods=['GET'])
def get_os_update():
    from core.baseline_engine import get_system_update_risk
    try:
        import winreg, datetime
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SOFTWARE\Microsoft\WindowsUpdate\UX\Settings")
        value, _ = winreg.QueryValueEx(key, "LastToastActionTime")
        winreg.CloseKey(key)
        last_update = datetime.datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        last_update = "Unknown"
    return jsonify({"last_update": last_update, "risk_score": get_system_update_risk()})


@app.route('/api/test/run_phishing', methods=['POST'])
def test_run_phishing():
    import subprocess
    subprocess.Popen([sys.executable,
                      os.path.join(os.path.dirname(__file__), '..',
                                   'simulations', 'simulate_phishing_click.py')])
    return jsonify({"status": "success"})


@app.route('/api/test/run_fatigue', methods=['POST'])
def test_run_fatigue():
    import subprocess
    subprocess.Popen([sys.executable, 'live_predictor.py'],
                     cwd=os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                       '..', 'sensors',
                                                       'fatigue_model_project')))
    return jsonify({"status": "success"})


if __name__ == '__main__':
    print("Scientific Dashboard API (Layers 1-6) running on port 5000")
    app.run(host='0.0.0.0', debug=True, port=5000)
