from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os
import math
import sys

# Add project root to path for core logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.risk_forecaster import CyberRiskForecaster
from core.simulation_engine import DigitalTwinSimulator

app = Flask(__name__)
CORS(app) 

# File Paths
BEHAVIOUR_LOG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'digital_behaviour.csv'))
ADVERSARIAL_LOG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'adversarial_failures.csv'))
CSV_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'telemetry_history.csv'))
LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'interventions.log'))
LATENT_LOG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'latent_states.csv'))

# Engines
risk_forecaster = CyberRiskForecaster()
simulator = DigitalTwinSimulator()

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Returns the historical telemetry data (physical sensors)."""
    if not os.path.exists(CSV_FILE):
        return jsonify([])
    try:
        df = pd.read_csv(CSV_FILE)
        if len(df) == 0: return jsonify([])
        df['time'] = pd.to_datetime(df['timestamp']).dt.strftime('%H:%M:%S')
        df['smoothed_keys'] = df['keys_pressed'].rolling(window=3, min_periods=1).mean().astype(int)
        chart_data = df[['time', 'keys_pressed', 'smoothed_keys']].tail(30).to_dict('records')
        return jsonify(chart_data)
    except:
        return jsonify([])

@app.route('/api/latent_states', methods=['GET'])
def get_latent_states():
    """Returns the historical brain states (Capacity, Demand, Risk)."""
    if not os.path.exists(LATENT_LOG):
        return jsonify([])
    try:
        df = pd.read_csv(LATENT_LOG)
        if len(df) == 0: return jsonify([])
        # Format time and tail
        df['time'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%H:%M:%S')
        data = df[['time', 'Ct', 'Dt', 'Ht', 'CRGt', 'risk_pct']].tail(30).to_dict('records')
        return jsonify(data)
    except Exception as e:
        print(f"Error reading latent log: {e}")
        return jsonify([])

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Returns live snapshot metrics for dashboard cards."""
    summary = {
        "Ct": 1.0, "Dt": 0.1, "Ht": 0.8, "risk_pct": 0, "total_interventions": 0
    }
    # 1. Get Latest Latent State
    if os.path.exists(LATENT_LOG):
        try:
            df = pd.read_csv(LATENT_LOG)
            if len(df) > 0:
                latest = df.iloc[-1]
                summary.update({
                    "Ct": round(latest['Ct'], 2),
                    "Dt": round(latest['Dt'], 2),
                    "Ht": round(latest['Ht'], 2),
                    "risk_pct": round(latest['risk_pct'], 1)
                })
        except: pass
            
    # 2. Get Interventions
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                summary["total_interventions"] = len(f.readlines())
        except: pass
            
    return jsonify(summary)

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """Calculates a 7-day risk trajectory based on the current state."""
    if not os.path.exists(LATENT_LOG):
        return jsonify([])
    try:
        df = pd.read_csv(LATENT_LOG)
        if len(df) == 0: return jsonify([])
        latest = df.iloc[-1]

        # Handle older logs that may not have the 'At' column
        at_val = float(latest['At']) if 'At' in df.columns else 0.0

        # Format as expected by forecaster
        current_state = {
            "Capacity_Ct": float(latest['Ct']),
            "Demand_Dt": float(latest['Dt']),
            "Habits_Ht": float(latest['Ht']),
            "Adversarial_At": at_val,
            "Reserve_Gap_CRGt": float(latest['CRGt'])
        }
        forecast = risk_forecaster.forecast_trajectory(current_state, days_ahead=7)
        return jsonify(forecast)
    except Exception as e:
        print(f"[API] Forecast error: {e}")
        return jsonify([])

@app.route('/api/simulation', methods=['POST'])
def run_simulation():
    """Runs a counterfactual simulation (do-calculus) for a specific policy."""
    data = request.json
    policy = data.get('policy', 'baseline') # baseline, security_training, ui_policy_change
    
    if not os.path.exists(LATENT_LOG):
        return jsonify({"status": "error", "message": "No latent data available"})
        
    df = pd.read_csv(LATENT_LOG)
    latest = df.iloc[-1]
    
    current_state = {
        "Ct": latest['Ct'], "Dt": latest['Dt'], "Ht": latest['Ht'], "At": latest['At']
    }
    
    # Run 30-day simulation
    history = simulator.run_simulation(current_state, steps=30, policy=policy)
    return jsonify(history)

@app.route('/api/behaviour', methods=['POST'])
def log_behaviour():
    import json
    data = request.json
    if not data: return jsonify({"status": "error"}), 400
    if not os.path.exists(BEHAVIOUR_LOG):
        with open(BEHAVIOUR_LOG, 'w', encoding='utf-8') as f: 
            f.write("timestamp,event,domain,details\n")
    
    # Store details as a proper JSON string, replacing commas with semicolons to avoid breaking simple CSV parsers if we just append
    # Wait, we can just quote it properly for standard CSV, or use the replace approach but json dump first
    details_dict = data.get('details', {})
    details_str = json.dumps(details_dict).replace(',', ';')
    
    with open(BEHAVIOUR_LOG, 'a', encoding='utf-8') as f:
        f.write(f"{data.get('timestamp')},{data.get('event')},{data.get('domain')},{details_str}\n")
    return jsonify({"status": "success"})

@app.route('/api/adversarial_signal', methods=['POST'])
def log_adversarial_signal():
    """Endpoint for simulating phishing clicks and other adversarial failures."""
    data = request.json
    if not data: return jsonify({"status": "error"}), 400
    
    import time
    if not os.path.exists(ADVERSARIAL_LOG):
        with open(ADVERSARIAL_LOG, 'w', encoding='utf-8') as f:
            f.write("timestamp,attack_type,status,details\n")
            
    with open(ADVERSARIAL_LOG, 'a', encoding='utf-8') as f:
        # The telemetry tracker looks for "Phishing" or "Scam" in the attack_type column
        attack_type = data.get('type', 'Unknown')
        status = data.get('status', 'Unknown')
        details = data.get('details', '')
        f.write(f"{time.time()},{attack_type},{status},{details}\n")
        
    return jsonify({"status": "success"})

@app.route('/api/test/live_sensors', methods=['GET'])
def get_live_sensors():
    import json
    data = {}
    live_sensor_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'live_sensors.json')
    if os.path.exists(live_sensor_file):
        try:
            with open(live_sensor_file, 'r') as f:
                data = json.load(f)
        except:
            pass
            
    beh = []
    if os.path.exists(BEHAVIOUR_LOG):
        try:
            df = pd.read_csv(BEHAVIOUR_LOG).tail(10)
            beh = df.to_dict('records')
        except:
            pass
            
    return jsonify({"sensors": data, "behaviour": beh})

@app.route('/api/test/os_update', methods=['GET'])
def get_os_update():
    from core.baseline_engine import get_system_update_risk
    try:
        import winreg
        import datetime
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\WindowsUpdate\UX\Settings")
        value, _ = winreg.QueryValueEx(key, "LastToastActionTime")
        winreg.CloseKey(key)
        
        last_update = datetime.datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        last_update = "Unknown (Could not read registry)"
        
    risk = get_system_update_risk()
    return jsonify({"last_update": last_update, "risk_score": risk})

@app.route('/api/test/run_phishing', methods=['POST'])
def test_run_phishing():
    import subprocess
    subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), '..', 'simulations', 'simulate_phishing_click.py')])
    return jsonify({"status": "success"})

@app.route('/api/test/run_fatigue', methods=['POST'])
def test_run_fatigue():
    import subprocess
    subprocess.Popen([sys.executable, 'live_predictor.py'], cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fatigue_model_project')))
    return jsonify({"status": "success"})

if __name__ == '__main__':
    print("Dashboard API Server (Layer 5) running on port 5000")
    app.run(debug=True, port=5000)
