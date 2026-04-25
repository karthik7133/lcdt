from core.state_inference import NeuralCDEInference
import torch

def test_inference():
    print("Initializing NCDE Inference Engine...")
    engine = NeuralCDEInference()
    
    signals = {
        'key_count': 50,
        'mouse_entropy': 1000,
        'task_switches': 2,
        'workload_modifier': 0.3
    }
    
    print("Running first update...")
    state = engine.update_inference(signals)
    print(f"State 1: {state}")
    
    print("Running second update (Path evolution)...")
    state2 = engine.update_inference(signals)
    print(f"State 2: {state2}")
    
    print("✅ NCDE Engine Test Passed!")

if __name__ == "__main__":
    test_inference()
