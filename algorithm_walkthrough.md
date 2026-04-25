# 🛡️ Deep Dive: The 5-Layer Cognitive-Cyber Algorithm

This document provides a comprehensive technical walkthrough of how the Cyber Watchdog translates human activity into a predictive Digital Twin.

---

## 📂 Layer 1: The Sensor Array (The Raw Inputs)
Every 10 seconds, the system gathers 12 distinct parameters from your digital and physical environment. These are the "raw signals."

1.  **Keystrokes**: Total count of keys pressed. Indicates active work.
2.  **Mouse Entropy**: Total distance and complexity of mouse movement.
3.  **Typing Errors**: Number of backspaces and deletes. Indicates cognitive friction.
4.  **Task Switching**: Frequency of changing windows/tabs. Indicates context-switching load.
5.  **Notifications**: Audio peaks from system alerts. Indicates external distraction.
6.  **Email Patterns**: Frequency of incoming emails, especially from unknown senders.
7.  **Password Habits**: Detection of password manager use (Good) vs. manual typing (Risk).
8.  **Browser Exposure**: Time spent on insecure (HTTP) sites or webmail.
9.  **OS Update Behavior**: Whether the system has delayed critical security patches.
10. **Sleep Deficit**: Detected when working during normal rest hours.
11. **Phishing Simulations**: Interactions with "bait" links or credential prompts.
12. **Vision Fatigue**: Eye-blinks and slumping detected by the AI camera (Vision Core).

---

## 🧠 Layer 2: State Inference (Mapping Signals to Roots)
Layer 2 maps those 12 raw signals into 4 **Hidden Roots** (The Internal State).

### Root 1: Cognitive Capacity (Cyan - Energy)
*   **Drivers**: Vision Fatigue, Sleep Deficit, Late-night work.
*   **Result**: A value between 0.0 (Exhausted) and 1.0 (Full Energy).

### Root 2: Cognitive Demand (Magenta - Stress)
*   **Drivers**: Task Switching, Notifications, Typing Errors.
*   **Result**: A value between 0.1 (Idle) and 1.0 (Max Stress).

### Root 3: Habit Quality (Green - Protection)
*   **Drivers**: Password habits, browser hits.
*   **Result**: A value between 0.0 (Unsafe) and 1.0 (Cyber-Expert).

### Root 4: Adversarial Threat (Red - Danger)
*   **Drivers**: Phishing interactions.
*   **Result**: A value between 0.0 (Safe) and 1.0 (Under Attack).

---

## 🔬 Layer 3: Multi-Timescale Dynamics (The Formulas)
This level ensures that states have "Momentum." We use the **Euler Method** to calculate the change in each state over a 10-second interval.

### 1. The Energy Formula (Slow Dynamics)
**Change in Capacity = Recovery Speed × (Target Baseline - Current Capacity)**
*   *Target Baseline*: Usually 1.0, but drops if you are tired.
*   *Recovery Speed*: A very small number (0.01), ensuring it takes hours to recharge.

### 2. The Stress Formula (Fast Dynamics)
**Change in Demand = (Work Pressure - Cooling Rate × (Current Demand - 0.1))**
*   *Work Pressure*: Calculated from your task switches and notifications.
*   *Cooling Rate*: 0.2, ensuring stress fades away over a few minutes of rest.

### 3. The Habit Formula (Medium Dynamics)
**New Habits = Old Habits + Learning Rate × Behavior Reward**
*   *Learning Rate*: 0.1.
*   *Reward*: Positive (+0.05) for good security, Negative (-0.1) for bad security.

### 4. The Reserve Gap (The Core Logic)
**Reserve Gap = Capacity minus Demand**
*   **Positive Gap**: You have spare brainpower (Healthy).
*   **Negative Gap**: You are in cognitive debt (Risky).

---

## 📈 Layer 4: Predictive Analytics (The Risk Formulas)
This level converts the Hidden Roots into a **Mistake Probability**.

### 1. The Logit Score (The Linear Combine)
We multiply each root by a "Beta Weight" (Importance Factor) and add a baseline Bias:
**Score = (-3 × Reserve Gap) + (-3 × Habits) + (6 × Threat) + 0.5**

### 2. The Probability Formula (Sigmoid Curve)
We "squash" the Score into a percentage using the Sigmoid function:
**Risk Percentage = 1 divided by (1 + Exponential of -Score)**

---

## 🔮 Layer 5: Digital Twin Simulation (The "What-If" Engine)
This layer projects the risk forward by 7 to 30 days by repeating the Layer 3 and 4 formulas under specific "Counterfactual" policies like **Aging Progression** or **Increased Workload**.

---

## 📝 Worked Example: The "Crunch Mode" Crisis
Let's walk through a manual calculation for a user under heavy stress.

### Layer 1: The Sensors
*   **Task Switches**: 20 (The user is jumping between tabs).
*   **Notifications**: 10 (System is pinging constantly).
*   **Phishing Link**: User just clicked an unknown email link.

### Layer 2: State Mapping
*   **Work Pressure**: 0.5 (Normalized from the 20 switches and 10 pings).
*   **Initial States**: Capacity = 1.0, Demand = 0.1, Habits = 0.8, Threat = 0.0.

### Layer 3: Dynamics Calculation
1.  **New Demand**: Change = (0.5 Pressure - 0) = 0.5. **New Demand = 0.6.**
2.  **New Capacity**: Stays at **1.0** (too slow to change in 10 seconds).
3.  **New Threat**: Instant spike to **1.0** (due to the phishing click).
4.  **Reserve Gap**: 1.0 (Energy) - 0.6 (Stress) = **0.4.**

### Layer 4: Risk Prediction
1.  **Calculate Score**:
    *   (-3 × 0.4 Gap) + (-3 × 0.8 Habits) + (6 × 1.0 Threat) + 0.5 Bias
    *   (-1.2) + (-2.4) + (6.0) + 0.5 = **2.9**
2.  **Calculate Percentage**:
    *   Sigmoid(2.9) = 1 / (1 + exp(-2.9)) 
    *   Sigmoid(2.9) = 1 / (1 + 0.055) = 1 / 1.055 = **0.947**
*   **RESULT**: The system displays a **94.7% Mistake Risk**.

### Layer 5: Digital Twin Simulation
The system simulates this same "Crunch" behavior for 30 days. It sees that **Capacity (Cyan)** will drain by 0.02 every day. By Day 15, the Reserve Gap becomes negative (-0.2), and the risk stays at **99%** even without a phishing attack. The Digital Twin warns the user to take a 3-day break now to avoid a catastrophic breach.
