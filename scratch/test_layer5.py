import sys
import os

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.simulation_engine import DigitalTwinSimulator

def test_causal_inference():
    simulator = DigitalTwinSimulator()
    
    # Starting state: A fatigued user (Low Capacity, High Demand)
    initial_state = {
        "Ct": 0.6,    # Capacity drained (e.g. poor sleep)
        "Dt": 0.5,    # Moderate demand
        "Ht": 0.4,    # Poor habits
        "At": 0.0,
        "CRGt": -0.1
    }

    print("--- [LAYER 5: CAUSAL SIMULATION TEST] ---")
    print(f"Starting Fatigue State: Capacity=0.6, Habits=0.4")
    print("-" * 50)

    # 1. Run Baseline Simulation (No intervention)
    baseline = simulator.run_simulation(initial_state, steps=30, policy=None)
    
    # 2. Run Counterfactual: do(security_training)
    training = simulator.run_simulation(initial_state, steps=30, policy="security_training")

    print(f"{'Day':<5} | {'Baseline Risk':<15} | {'Training Risk':<15} | {'Divergence'}")
    print("-" * 50)

    # Sample results every 5 days for brevity
    for i in [0, 4, 9, 14, 19, 24, 29]:
        b = baseline[i]
        t = training[i]
        divergence = round(b['Risk_Pct'] - t['Risk_Pct'], 2)
        print(f"{b['step']:<5} | {b['Risk_Pct']:<15}% | {t['Risk_Pct']:<15}% | {divergence}% saved")

    print("-" * 50)
    final_baseline = baseline[-1]['Risk_Pct']
    final_training = training[-1]['Risk_Pct']
    
    print(f"RESULT: After 30 days, Security Training reduced mistake probability from {final_baseline}% to {final_training}%.")
    print("This demonstrates the Digital Twin's ability to calculate E[M | do(training)].")

if __name__ == "__main__":
    test_causal_inference()
