import numpy as np
import copy
from core.state_inference import LatentStateInference
from core.risk_forecaster import CyberRiskForecaster

class DigitalTwinSimulator:
    def __init__(self):
        # We use instances to access hyperparameters and formulas
        self.inference_engine = LatentStateInference()
        self.risk_forecaster = CyberRiskForecaster()

    def run_simulation(self, initial_state, steps=30, signals_template=None, policy=None):
        """
        The core simulation loop. 
        Calculates E[Mt:T | do(pi)] by iteratively updating Layer 3 & Layer 4.
        """
        # Ensure we work with internal short keys for simulation logic
        sim_state = {
            "Ct": initial_state.get("Capacity_Ct", initial_state.get("Ct", 1.0)),
            "Dt": initial_state.get("Demand_Dt", initial_state.get("Dt", 0.1)),
            "Ht": initial_state.get("Habits_Ht", initial_state.get("Ht", 0.8)),
            "At": initial_state.get("Adversarial_At", initial_state.get("At", 0.0))
        }
        
        history = []
        
        # Default signals if none provided
        if signals_template is None:
            signals_template = {}

        for t in range(1, steps + 1):
            # 1. Apply do(pi) Interventions to signals or parameters
            current_signals = copy.deepcopy(signals_template)
            
            # Policy-specific logic
            if policy == "increased_workload":
                current_signals['task_switches'] = 25 # Force max stress
                current_signals['notification_count'] = 15
            
            if policy == "ui_policy_change":
                # Zero out task switches as if the UI prevents context switching
                current_signals['task_switches'] = 0
            
            # 2. Layer 3 Dynamics (Euler Method approximation)
            # Capacity (Ct) update
            mu = 1.0
            if policy == "aging_progression":
                # Steadily decay the baseline capacity over the simulation
                mu = max(0.4, 1.0 - (t * 0.02))
            
            if current_signals.get('sleep_deficit') == 'TRUE':
                mu -= 0.3
            
            # Slow Dynamics C_t
            # Split dynamics: recover fast, drain slow
            diff = mu - sim_state["Ct"]
            alpha = self.inference_engine.alpha_up if diff > 0 else self.inference_engine.alpha_down
            
            dCt = alpha * diff * self.inference_engine.dt
            sim_state["Ct"] = max(0.0, min(1.0, sim_state["Ct"] + dCt))

            # Demand (Dt) update
            f_work = 0.0
            f_work += min(current_signals.get('task_switches', 0) / 20.0, 0.3)
            f_work += min(current_signals.get('notification_count', 0) / 10.0, 0.2)
            
            # Fast Dynamics D_t
            dDt = (f_work - self.inference_engine.beta * (sim_state["Dt"] - 0.1)) * self.inference_engine.dt
            sim_state["Dt"] = max(0.1, min(1.0, sim_state["Dt"] + dDt))

            # Habit (Ht) update
            reward_r = 0.0
            if policy == "security_training":
                # Inject a massive reward spike each day
                reward_r += 0.2
            
            # Medium Dynamics H_t
            sim_state["Ht"] = max(0.0, min(1.0, sim_state["Ht"] + self.inference_engine.eta * reward_r))

            # 3. Layer 4: Get Risk Probability (Mapping to expected keys)
            input_state = {
                "Capacity_Ct": sim_state["Ct"],
                "Demand_Dt": sim_state["Dt"],
                "Habits_Ht": sim_state["Ht"],
                "Adversarial_At": sim_state["At"],
                "Reserve_Gap_CRGt": sim_state["Ct"] - sim_state["Dt"]
            }
            
            risk_pct = self.risk_forecaster.calculate_live_risk(input_state)
            
            history.append({
                "step": t,
                "Capacity": round(sim_state["Ct"], 3),
                "Demand": round(sim_state["Dt"], 3),
                "Habits": round(sim_state["Ht"], 3),
                "Risk_Pct": risk_pct
            })
            
        return history

    def simulate_scenarios(self, current_state, days=30):
        """Runs the four standard architecture interventions for comparison."""
        results = {
            "baseline": self.run_simulation(current_state, steps=days),
            "do_workload": self.run_simulation(current_state, steps=days, policy="increased_workload"),
            "do_training": self.run_simulation(current_state, steps=days, policy="security_training"),
            "do_ui_change": self.run_simulation(current_state, steps=days, policy="ui_policy_change"),
            "do_aging": self.run_simulation(current_state, steps=days, policy="aging_progression")
        }
        return results
