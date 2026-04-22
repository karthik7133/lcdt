import sys
import os
import time

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.state_inference import LatentStateInference

def test_dynamics():
    engine = LatentStateInference()
    print("--- Starting Layer 3 Dynamics Test ---")
    print(f"Initial State: {engine.state}")
    print("-" * 40)

    # Phase 1: High Workload (10 intervals)
    print("Phase 1: High Workload (Task Switching & Notifications)")
    for i in range(1, 11):
        signals = {
            'task_switches': 15,
            'notification_count': 5,
            'typing_error_rate': 0.05
        }
        res = engine.update_inference(signals)
        print(f"Step {i:02d} | Demand: {res['Demand_Dt']:.2f} | Capacity: {res['Capacity_Ct']:.2f} | CRG: {res['Reserve_Gap_CRGt']:.2f}")

    print("-" * 40)
    # Phase 2: Rest/Idle (10 intervals)
    print("Phase 2: Rest (Zero signals, observing cooling-off)")
    for i in range(1, 11):
        signals = {} # No work
        res = engine.update_inference(signals)
        print(f"Rest {i:02d} | Demand: {res['Demand_Dt']:.2f} | Capacity: {res['Capacity_Ct']:.2f} | CRG: {res['Reserve_Gap_CRGt']:.2f}")

    print("-" * 40)
    # Phase 3: Sleep Deficit Impact
    print("Phase 3: Impact of Sleep Deficit on Capacity Drift")
    signals = {'sleep_deficit': 'TRUE'}
    for i in range(1, 11):
        res = engine.update_inference(signals)
        print(f"Deficit {i:02d} | Capacity: {res['Capacity_Ct']:.2f} | Demand: {res['Demand_Dt']:.2f}")

if __name__ == "__main__":
    test_dynamics()
