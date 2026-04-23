import numpy as np

class LatentStateInference:
    def __init__(self):
        # --- Layer 3 Hyperparameters (The "Biological" Constants) ---
        self.dt = 1.0          # Time step (per interval)
        self.alpha_up = 0.01   # Slow recovery (requires sustained rest)
        self.alpha_down = 0.08 # Aggressive drain (Capacity can crash in ~2 mins)
        self.beta = 0.2        # Fast Dynamics (Demand cooling rate)
        self.eta = 0.1         # Medium Dynamics (Habit learning rate)
        self.sigma = 0.0

        # Initialize the Hidden Human State Zt
        self.state = {
            "Ct": 1.0,   # Capacity (Slow)       - cognitive health [0,1]
            "Dt": 0.1,   # Demand (Fast)         - current workload [0.1,1]
            "Ht": 0.8,   # Habits (Medium)       - security habit quality [0,1]
            "At": 0.0,   # Adversarial (Threat)  - exposure to attacks [0,1]
            "CRGt": 0.9  # Reserve Gap (Output)  - Ct - Dt (positive = safe)
        }

        # --- Session-level flags to prevent continuous draining ---
        # os_update_delayed should only penalise ONCE per session, not every 10s.
        self._os_update_penalised = False

    def update_inference(self, signals):
        """
        Calculates the Hidden Human State Zt using Multi-Timescale Dynamical Systems.
        Strictly follows the Euler Method approximations for dCt, dDt, and Ht+1.

        CRT Model Order:
          Ct  (Slow)   <- sleep / vision fatigue only
          Dt  (Fast)   <- real task events only (keys, task switches, notifications)
          Ht  (Medium) <- discrete security behaviour events only
          At  (Threat) <- adversarial events
          CRGt         <- Ct - Dt (positive reserve = healthy)
        """

        # ------------------------------------------------------------------ #
        # 1. SLOW DYNAMICS — Cognitive Capacity (Ct)
        #    dCt = alpha * (mu - Ct) * dt
        #    Driven by biological state AND sustained work session fatigue.
        #    At rest with no deficits, Ct drifts smoothly back to 1.0.
        # ------------------------------------------------------------------ #
        mu = 1.0  # Healthy biological baseline
        if signals.get('sleep_deficit') == 'TRUE':
            mu -= 0.3
        if signals.get('vision_fatigue') == 'TRUE':
            mu -= 0.6 # Increased impact
            
        # --- PERFORMANCE DROP DETECTION ---
        # If user is barely typing/moving relative to baseline, it's exhaustion.
        base_keys = signals.get('base_keys', 0)
        key_count = signals.get('key_count', 0)
        mouse_entropy = signals.get('mouse_entropy', 0.0)
        
        if base_keys > 10: # Only check if there is a meaningful baseline
            perf_ratio = key_count / base_keys
            if perf_ratio < 0.4 and mouse_entropy < 150:
                mu -= 0.5 # Serious capacity penalty for "stalling"
        elif key_count == 0 and mouse_entropy < 50:
            # If no baseline but absolute zero activity, drift mu lower
            mu -= 0.2

        # --- CIRCADIAN RHYTHM IMPACT ---
        # Late night/Early morning work (23:00 to 05:00) reduces capacity baseline
        hour = signals.get('hour_of_day', 12) # Default to midday if unknown
        if hour >= 23 or hour <= 5:
            mu -= 0.2
            # Also increase drain rate at night
            self.alpha_down = 0.12 # 50% faster drain than default 0.08
        else:
            self.alpha_down = 0.08 # Reset to default

        # Split dynamics: recover fast, drain slow
        diff = mu - self.state["Ct"]
        alpha = self.alpha_up if diff > 0 else self.alpha_down
        
        # --- RECOVERY INHIBITION ---
        # If we are recovering (diff > 0) but the user is actively typing or moving mouse,
        # freeze or significantly slow down the recovery.
        if diff > 0 and (key_count > 0 or mouse_entropy > 200.0):
            alpha = self.alpha_up * 0.1 # 90% reduction in recovery speed while working
            
        dCt = alpha * diff * self.dt
        self.state["Ct"] = max(0.1, min(1.0, self.state["Ct"] + dCt))

        # ------------------------------------------------------------------ #
        # 2. FAST DYNAMICS — Cognitive Demand (Dt)
        #    dDt = (f_work - beta * (Dt - D_floor)) * dt
        #
        #    FIX: mouse_entropy was using raw pixel distance (e.g. 1337px)
        #    which equated to 1337/5000 = 0.27 of demand even at idle jitter.
        #    Now: mouse activity only contributes when movement exceeds the
        #    "meaningful work" threshold of 200px per 10-second interval.
        #    If idle (0 keys, <200px mouse), f_work stays 0 and beta cooling
        #    pulls Dt back to its 0.1 floor — no phantom demand.
        # ------------------------------------------------------------------ #
        f_work = 0.0

        # Task switching pressure (window changes per minute interval)
        # Divisor increased to 40.0 to reduce sensitivity to minor switches
        task_switches = signals.get('task_switches', 0)
        if task_switches > 0:
            f_work += min(task_switches / 40.0, 0.2)

        # Interruption pressure (audio notifications)
        notification_count = signals.get('notification_count', 0)
        if notification_count > 0:
            f_work += min(notification_count / 10.0, 0.2)

        # Typing friction (only meaningful if user is actually typing)
        # Weight reduced from 2.0 to 1.0; capped at 0.1
        key_count = signals.get('key_count', 0)
        if key_count > 0:
            typing_error_rate = signals.get('typing_error_rate', 0.0)
            f_work += min(typing_error_rate * 1.0, 0.1)

        # Mouse activity: only counts as demand when actively working (>200px)
        # This prevents idle mouse jitter from inflating Dt.
        mouse_entropy = signals.get('mouse_entropy', 0.0)
        if mouse_entropy > 200.0:
            # Scale: 200px=small, 2000px=moderate, 5000px+=max contribution
            f_work += min((mouse_entropy - 200.0) / 4800.0, 0.2)

        # Email & link activity
        email_frequency = signals.get('email_frequency', 0)
        if email_frequency > 0:
            f_work += min(email_frequency / 10.0, 0.2)

        link_clicks = signals.get('link_clicks', 0)
        if link_clicks > 0:
            f_work += min(link_clicks / 20.0, 0.2)

        # Fast response time = reactive high-pressure work
        avg_rt = signals.get('avg_response_time', 0)
        if 0 < avg_rt < 500:
            f_work += 0.1

        # Euler step: beta cooling naturally returns Dt to 0.1 floor when idle
        dDt = (f_work - self.beta * (self.state["Dt"] - 0.1)) * self.dt
        self.state["Dt"] = max(0.1, min(1.0, self.state["Dt"] + dDt))

        # ------------------------------------------------------------------ #
        # 3. MEDIUM DYNAMICS — Habit Quality (Ht)
        #    Ht+1 = Ht + eta * reward_r
        #
        #    FIX: os_update_delayed was checking the registry on every 10-second
        #    interval and always returning TRUE (machine is always a bit outdated),
        #    causing a persistent -0.01/tick drain that looked random.
        #    Now: os_update_delayed only fires its penalty ONCE per session.
        #    All other signals only fire when genuinely NEW events occur this
        #    interval, so Ht is perfectly stable when nothing happens.
        # ------------------------------------------------------------------ #
        reward_r = 0.0

        # Insecure browsing events (new hits this interval)
        insecure_hits = signals.get('insecure_http_hits', 0)
        if insecure_hits > 0:
            reward_r -= 0.05 * insecure_hits

        # Webmail hits
        webmail_hits = signals.get('webmail_hits', 0)
        if webmail_hits > 0:
            reward_r -= 0.02 * webmail_hits

        # OS update delay: penalise ONCE per session, not every 10 seconds.
        if signals.get('os_update_delayed') == 'TRUE' and not self._os_update_penalised:
            reward_r -= 0.1
            self._os_update_penalised = True  # lock — won't fire again this session

        # Reset the flag when updates are no longer delayed (user updated)
        if signals.get('os_update_delayed') == 'FALSE':
            self._os_update_penalised = False

        # Good password habits (positive reinforcement)
        if signals.get('good_password_paste') == 'TRUE':
            reward_r += 0.05

        # Weak passwords detected
        low_passwords = signals.get('low_strength_passwords', 0)
        if low_passwords > 0:
            reward_r -= 0.1 * low_passwords

        # Unknown email senders (phishing risk indicator)
        unknown_senders = signals.get('unknown_senders', 0)
        if unknown_senders > 0:
            reward_r -= 0.05 * unknown_senders

        # Only update Ht when a genuine event occurred — stable when idle
        if reward_r != 0.0:
            self.state["Ht"] = max(0.0, min(1.0, self.state["Ht"] + self.eta * reward_r))
        else:
            # --- SLOW HABIT RECOVERY ---
            # If no negative events this interval, habits drift back toward 0.8 baseline.
            # 0.005 per interval (10s) = ~0.03 per minute (~25 mins to full recovery).
            if self.state["Ht"] < 0.8:
                self.state["Ht"] = min(0.8, self.state["Ht"] + 0.005)

        # ------------------------------------------------------------------ #
        # 4. ADVERSARIAL EXPOSURE (At) — Event-Driven with Decay
        #    Spikes to 1.0 on adversarial event, decays by 0.05/interval.
        # ------------------------------------------------------------------ #
        if signals.get('phishing_clicked') == 'TRUE' or signals.get('scam_credentials_given') == 'TRUE':
            self.state["At"] = 1.0
        else:
            self.state["At"] = max(0.0, self.state["At"] - 0.05)

        # ------------------------------------------------------------------ #
        # 5. COGNITIVE RESERVE GAP (CRGt) — The Core Output
        #    CRGt = Ct - Dt
        #    Positive = spare capacity (safe), Negative = overloaded (risky).
        #    FIX: was Dt - Ct (inverted), now correctly Ct - Dt per CRT model.
        # ------------------------------------------------------------------ #
        self.state["CRGt"] = self.state["Ct"] - self.state["Dt"]

        return {
            "Capacity_Ct": round(self.state["Ct"], 2),
            "Demand_Dt": round(self.state["Dt"], 2),
            "Habits_Ht": round(self.state["Ht"], 2),
            "Adversarial_At": round(self.state["At"], 2),
            "Reserve_Gap_CRGt": round(self.state["CRGt"], 2)
        }
