import sys
import os

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.state_inference import LatentStateInference
from core.risk_forecaster import CyberRiskForecaster

def run_prediction_demo():
    # Initialize both engines
    layer3 = LatentStateInference()
    layer4 = CyberRiskForecaster()

    print("--- [Layer 4: Predictive Risk Demo] ---")
    
    # Scenario: High Stress + Sleep Deficit
    print("\n[Scenario]: High Workload + Sleep Deficit")
    signals = {
        'task_switches': 18,
        'notification_count': 8,
        'sleep_deficit': 'TRUE'
    }

    # 1. Update Psychological State (Layer 3)
    # We'll run a few steps to let the dynamics settle
    for i in range(5):
        latent_state = layer3.update_inference(signals)
    
    print(f"Current State: {latent_state}")

    # 2. Calculate Live Risk Probability (Layer 4)
    risk_pct = layer4.calculate_live_risk(latent_state)
    print(f"!!! LIVE CYBER MISTAKE PROBABILITY: {risk_pct}%")

    # 3. Forecast Trajectory (7-Day Forecast)
    forecast = layer4.forecast_trajectory(latent_state, days_ahead=7)
    print("\n[7-Day Risk Forecast (Burnout Simulation)]:")
    for day in forecast:
        print(f"  Day {day['day']}: {day['forecasted_risk_pct']}% risk")

    # Scenario: Good Habits Protective Effect
    print("\n[Scenario]: Adding Good Habits (Password Manager)")
    signals['good_password_paste'] = 'TRUE'
    for i in range(5):
        latent_state = layer3.update_inference(signals)
    
    risk_pct_new = layer4.calculate_live_risk(latent_state)
    print(f"OK: NEW RISK (with habits): {risk_pct_new}%")
    print(f"Reduction: {round(risk_pct - risk_pct_new, 2)}%")

if __name__ == "__main__":
    run_prediction_demo()
