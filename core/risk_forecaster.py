import numpy as np

class CyberRiskForecaster:
    def __init__(self):
        # The Beta Weights (Tuning Parameters)
        # CRGt = Ct - Dt (positive = spare capacity, HEALTHY)
        # Weight reduced from -4.0 to -3.0 to be more forgiving of moderate load
        self.beta1 = -3.0  # High reserve (Ct > Dt) = LOWER risk
        self.beta2 = -3.0  # Strong REDUCTION in risk from good Habits (Protective factor)
        self.beta3 = 6.0   # Extreme impact from active Adversarial attacks (Immediate danger)

        # Bias reduced from 2.0 to 0.5
        # Calibrated so a healthy active person (CRGt=0.4, Ht=0.8) = ~5-7% risk.
        self.bias = 0.5

    def sigmoid(self, x):
        """Mathematical function to squash any value between 0.0 and 1.0"""
        return 1 / (1 + np.exp(-x))

    def calculate_live_risk(self, latent_state):
        """
        Calculates short-term P(Mt) based on the current Layer 3 State.
        """
        # Extract states from the Layer 3 output
        # Note: keys must match the output of LatentStateInference.update_inference
        crg = latent_state["Reserve_Gap_CRGt"]
        ht = latent_state["Habits_Ht"]
        at = latent_state["Adversarial_At"]

        # The Linear Combination: (B1*CRG) + (B2*H) + (B3*A) + Bias
        # This is the 'Logit' or Z-score for the probability
        x = (self.beta1 * crg) + (self.beta2 * ht) + (self.beta3 * at) + self.bias
        
        # Apply the Sigmoid curve to get P(Mt) in range [0, 1]
        probability = self.sigmoid(x)
        
        # Convert to a readable percentage (e.g., 84.5%)
        risk_percentage = round(probability * 100, 2)
        return risk_percentage

    def forecast_trajectory(self, current_state, days_ahead):
        """
        Calculates Mid/Long-term risk by simulating future cognitive degradation.
        (Outputs the 'Cyber Risk Trajectory' from the architecture document)
        """
        forecasts = []
        # Create a working copy of the psychological state
        # We use a simple degradation heuristic: chronic stress drains capacity
        simulated_state = {
            "Capacity_Ct": current_state["Capacity_Ct"],
            "Demand_Dt": current_state["Demand_Dt"],
            "Habits_Ht": current_state["Habits_Ht"],
            "Adversarial_At": current_state["Adversarial_At"],
            "Reserve_Gap_CRGt": current_state["Reserve_Gap_CRGt"]
        }
        
        for day in range(1, days_ahead + 1):
            # Simulation Heuristic:
            # If demand is consistently high (>0.6), capacity drops slightly each day due to burnout
            if simulated_state["Demand_Dt"] > 0.6:
                simulated_state["Capacity_Ct"] = max(0.1, simulated_state["Capacity_Ct"] - 0.02)

            # CRGt = Ct - Dt (positive = healthy reserve, matches the inference engine)
            simulated_state["Reserve_Gap_CRGt"] = simulated_state["Capacity_Ct"] - simulated_state["Demand_Dt"]

            # Calculate future risk based on the simulated degradation
            future_risk = self.calculate_live_risk(simulated_state)
            forecasts.append({"day": day, "forecasted_risk_pct": future_risk})
            
        return forecasts
