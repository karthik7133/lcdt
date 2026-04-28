"""
fuzzy_engine.py
================
Pure Fuzzy Logic inference engine for fatigue detection.
NO external fuzzy library needed — implemented from scratch using
standard triangular / trapezoidal membership functions + centroid defuzzification.

Architecture:
  Inputs  : EAR, MAR, Pitch, EAR_MA (10-frame moving average)
  Output  : Fatigue Index ∈ [0, 100]
  Decision: Fatigue Index > 50  →  Label = 1 (Tired)

Fuzzy Sets per Input:
  EAR   : {open, half_closed, closed}
  MAR   : {normal, yawning}
  Pitch : {upright, nodding}
  EAR_MA: {stable_open, drifting, stable_closed}

Rules (19 expert-defined rules):
  Based on published fatigue detection research + clinical thresholds.
  EAR < 0.20 is clinically associated with drowsiness.
  MAR > 0.50 indicates yawning.
  Pitch < 0.55 indicates head nodding.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MEMBERSHIP FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def trimf(x, a, b, c):
    """Triangular membership function: rises from a to b, falls from b to c."""
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a + 1e-9)
    return (c - x) / (c - b + 1e-9)


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function: flat top between b and c."""
    if x <= a or x >= d:
        return 0.0
    if x >= b and x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a + 1e-9)
    return (d - x) / (d - c + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# FUZZIFICATION: Membership Degrees per Variable
# ─────────────────────────────────────────────────────────────────────────────

def fuzzify_ear(ear):
    """
    EAR (Eye Aspect Ratio) — higher = more open.
    Clinical thresholds:
      < 0.18 : closed / micro-sleep
      0.18–0.25 : half-closed / drowsy
      > 0.25 : open / alert
    """
    closed      = trapmf(ear, 0.00, 0.05, 0.16, 0.20)
    half_closed = trimf(ear,  0.16, 0.22, 0.28)
    open_eye    = trapmf(ear, 0.24, 0.28, 0.50, 1.00)
    return {"closed": closed, "half_closed": half_closed, "open": open_eye}


def fuzzify_mar(mar):
    """
    MAR (Mouth Aspect Ratio) — higher = more open (yawning).
    Thresholds:
      < 0.30 : mouth closed (normal)
      > 0.50 : yawning
    """
    normal  = trapmf(mar, 0.00, 0.00, 0.28, 0.45)
    yawning = trapmf(mar, 0.38, 0.55, 1.00, 2.00)
    return {"normal": normal, "yawning": yawning}


def fuzzify_pitch(pitch):
    """
    Pitch (head tilt ratio) — lower = head nodding down.
    Thresholds:
      < 0.55 : nodding (fatigue indicator)
      > 0.60 : upright
    """
    nodding = trapmf(pitch, 0.00, 0.10, 0.52, 0.62)
    upright = trapmf(pitch, 0.55, 0.65, 2.00, 3.00)
    return {"nodding": nodding, "upright": upright}


def fuzzify_ear_ma(ear_ma):
    """
    EAR moving average over 10 frames — captures sustained droopiness.
    """
    stable_closed = trapmf(ear_ma, 0.00, 0.05, 0.18, 0.22)
    drifting      = trimf(ear_ma,  0.18, 0.24, 0.30)
    stable_open   = trapmf(ear_ma, 0.26, 0.30, 0.50, 1.00)
    return {"stable_closed": stable_closed, "drifting": drifting, "stable_open": stable_open}


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT MEMBERSHIP FUNCTIONS (for defuzzification universe)
# ─────────────────────────────────────────────────────────────────────────────

# Universe of discourse for Fatigue Index: 0–100
UNIVERSE = np.linspace(0, 100, 500)


def output_alert(x):
    """Alert (low fatigue): 0–35"""
    return trapmf(x, 0, 0, 20, 40)


def output_moderate(x):
    """Moderate fatigue: 30–70"""
    return trimf(x, 25, 50, 75)


def output_tired(x):
    """Tired (high fatigue): 60–100"""
    return trapmf(x, 55, 70, 100, 100)


# ─────────────────────────────────────────────────────────────────────────────
# RULE BASE (19 Expert Rules)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_rules(ear_mf, mar_mf, pitch_mf, ear_ma_mf):
    """
    Returns activation strengths for (alert, moderate, tired) output sets.

    Rules based on clinical fatigue indicators:
    - EAR closed / half-closed = strong fatigue signal
    - MAR yawning = moderate fatigue signal
    - Pitch nodding = fatigue signal (head drop)
    - EAR_MA stable_closed = sustained drowsiness
    """
    e  = ear_mf
    m  = mar_mf
    p  = pitch_mf
    em = ear_ma_mf

    alert    = 0.0
    moderate = 0.0
    tired    = 0.0

    # ── ALERT RULES (person is awake) ───────────────────────────────────────
    # R1: IF EAR open AND MAR normal AND Pitch upright → alert
    alert = max(alert, min(e["open"], m["normal"], p["upright"]))

    # R2: IF EAR open AND EAR_MA stable_open → alert
    alert = max(alert, min(e["open"], em["stable_open"]))

    # R3: IF EAR open AND MAR normal → alert
    alert = max(alert, min(e["open"], m["normal"]) * 0.8)

    # ── MODERATE RULES ──────────────────────────────────────────────────────
    # R4: IF EAR half_closed AND MAR normal → moderate
    moderate = max(moderate, min(e["half_closed"], m["normal"]))

    # R5: IF EAR open AND Pitch nodding → moderate
    moderate = max(moderate, min(e["open"], p["nodding"]))

    # R6: IF EAR open AND MAR yawning → moderate
    moderate = max(moderate, min(e["open"], m["yawning"]))

    # R7: IF EAR open AND EAR_MA drifting → moderate
    moderate = max(moderate, min(e["open"], em["drifting"]))

    # R8: IF EAR half_closed AND Pitch upright → moderate
    moderate = max(moderate, min(e["half_closed"], p["upright"]) * 0.9)

    # R9: IF EAR_MA drifting AND MAR normal → moderate
    moderate = max(moderate, min(em["drifting"], m["normal"]) * 0.8)

    # ── TIRED RULES ─────────────────────────────────────────────────────────
    # R10: IF EAR closed → tired (strongest single cue)
    tired = max(tired, e["closed"])

    # R11: IF EAR half_closed AND MAR yawning → tired
    tired = max(tired, min(e["half_closed"], m["yawning"]))

    # R12: IF EAR half_closed AND Pitch nodding → tired
    tired = max(tired, min(e["half_closed"], p["nodding"]))

    # R13: IF EAR half_closed AND EAR_MA stable_closed → tired
    tired = max(tired, min(e["half_closed"], em["stable_closed"]))

    # R14: IF EAR_MA stable_closed AND Pitch nodding → tired
    tired = max(tired, min(em["stable_closed"], p["nodding"]))

    # R15: IF EAR_MA stable_closed AND MAR yawning → tired
    tired = max(tired, min(em["stable_closed"], m["yawning"]))

    # R16: IF EAR closed AND MAR yawning → tired (very strong)
    tired = max(tired, min(e["closed"], m["yawning"]))

    # R17: IF EAR closed AND Pitch nodding → tired (very strong)
    tired = max(tired, min(e["closed"], p["nodding"]))

    # R18: IF EAR half_closed AND MAR yawning AND Pitch nodding → tired
    tired = max(tired, min(e["half_closed"], m["yawning"], p["nodding"]))

    # R19: IF EAR_MA stable_closed AND MAR yawning AND Pitch nodding → tired
    tired = max(tired, min(em["stable_closed"], m["yawning"], p["nodding"]))

    return alert, moderate, tired


# ─────────────────────────────────────────────────────────────────────────────
# DEFUZZIFICATION: Centroid Method (CoG)
# ─────────────────────────────────────────────────────────────────────────────

def defuzzify(alert_strength, moderate_strength, tired_strength):
    """
    Mamdani defuzzification using centroid (Center of Gravity) method.
    Aggregates clipped output membership functions and finds centroid.
    """
    aggregated = np.zeros_like(UNIVERSE)

    for x in range(len(UNIVERSE)):
        v = UNIVERSE[x]
        # Clip each output MF by its activation strength (Mamdani min-implication)
        a_clip = min(alert_strength,    output_alert(v))
        m_clip = min(moderate_strength, output_moderate(v))
        t_clip = min(tired_strength,    output_tired(v))
        # Aggregate by max
        aggregated[x] = max(a_clip, m_clip, t_clip)

    total = np.sum(aggregated)
    if total < 1e-9:
        return 50.0  # fallback: uncertain

    centroid = float(np.sum(UNIVERSE * aggregated) / total)
    return round(centroid, 2)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

class FuzzyFatigueEngine:
    """
    Main Fuzzy Logic inference engine.
    Replaces the SVM classifier entirely — no training required.

    Usage:
        engine = FuzzyFatigueEngine()
        fatigue_index = engine.predict(ear=0.22, mar=0.08, pitch=0.65, ear_ma=0.23)
        label = 1 if fatigue_index > 50 else 0
    """

    # Threshold separating Awake (0) from Tired (1)
    THRESHOLD = 50.0

    def predict(self, ear: float, mar: float, pitch: float,
                ear_ma: float = None) -> dict:
        """
        Runs full fuzzy inference pipeline.

        Args:
            ear     : Eye Aspect Ratio (current frame)
            mar     : Mouth Aspect Ratio (current frame)
            pitch   : Head pitch ratio (current frame)
            ear_ma  : EAR 10-frame moving average (defaults to ear if None)

        Returns:
            dict with keys: fatigue_index, label, alert, moderate, tired
        """
        if ear_ma is None:
            ear_ma = ear

        # Fuzzify
        ear_mf    = fuzzify_ear(ear)
        mar_mf    = fuzzify_mar(mar)
        pitch_mf  = fuzzify_pitch(pitch)
        ear_ma_mf = fuzzify_ear_ma(ear_ma)

        # Rule evaluation
        alert_s, moderate_s, tired_s = evaluate_rules(
            ear_mf, mar_mf, pitch_mf, ear_ma_mf)

        # Defuzzify
        fatigue_index = defuzzify(alert_s, moderate_s, tired_s)
        label = 1 if fatigue_index >= self.THRESHOLD else 0

        return {
            "fatigue_index":     fatigue_index,
            "label":             label,
            "strength_alert":    round(alert_s,    3),
            "strength_moderate": round(moderate_s, 3),
            "strength_tired":    round(tired_s,    3),
        }

    def predict_batch(self, df) -> "pd.Series":
        """
        Vectorised prediction over a DataFrame with columns: EAR, MAR, Pitch, EAR_MA.
        Returns a Series of labels (0 or 1).
        """
        import pandas as pd
        labels = []
        for _, row in df.iterrows():
            result = self.predict(
                ear    = row["EAR"],
                mar    = row["MAR"],
                pitch  = row["Pitch"],
                ear_ma = row.get("EAR_MA", row["EAR"])
            )
            labels.append(result["label"])
        return pd.Series(labels, index=df.index)
