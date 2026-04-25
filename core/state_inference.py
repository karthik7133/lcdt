"""
LAYER 1 + LAYER 2 + LAYER 3: core/state_inference.py
=======================================================
L1 — Multimodal Behaviour Graph
      Full 20-signal temporal graph with proper feature embedding.

L2 — Neural Controlled Differential Equations (NCDE)
      dZt = f_theta(Zt)dt + g_phi(Zt)dXt
      Solves via torchcde.cdeint with Hermite cubic spline path.
      Includes offline training from historical latent_states.csv.

L3 — Personalized Latent Cognitive State Space
      Zt(user) ~ N(mu_user, Sigma_user)
      Per-user MAP calibration with online Bayesian shrinkage.
"""

import os
import time
import json
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchcde
import networkx as nx

# ─────────────────────────────────────────────────────────────
# LAYER 1 — MULTIMODAL TEMPORAL BEHAVIOUR GRAPH
# ─────────────────────────────────────────────────────────────

# All 20 raw Layer-1 signals mapped to graph event categories
SIGNAL_TO_EVENT = {
    "key_count":               ("User",    "Workstation", "typing"),
    "mouse_entropy":           ("User",    "Workstation", "mouse_movement"),
    "typing_error_rate":       ("User",    "Workstation", "typing_friction"),
    "task_switches":           ("Apps",    "User",        "context_switch"),
    "notification_count":      ("System",  "User",        "interruption"),
    "workload_modifier":       ("Schedule","User",        "calendar_pressure"),
    "insecure_http_hits":      ("Browser", "Risk",        "insecure_browse"),
    "webmail_hits":            ("Browser", "Risk",        "webmail_access"),
    "link_clicks":             ("Browser", "Risk",        "link_click"),
    "email_frequency":         ("Email",   "User",        "email_activity"),
    "unknown_senders":         ("Email",   "Risk",        "phishing_signal"),
    "avg_response_time":       ("Email",   "User",        "email_response"),
    "low_strength_passwords":  ("Security","Risk",        "weak_password"),
    "good_password_paste":     ("Security","User",        "good_habit"),
    "os_update_delayed":       ("System",  "Risk",        "update_delay"),
    "sleep_deficit":           ("Bio",     "User",        "sleep_debt"),
    "vision_fatigue":          ("Bio",     "User",        "vision_fatigue"),
    "phishing_clicked":        ("Threat",  "Risk",        "phishing_click"),
    "scam_credentials_given":  ("Threat",  "Risk",        "credential_theft"),
    "hour_of_day":             ("Clock",   "User",        "circadian"),
}

# Node types for categorical encoding
NODE_TYPES = ["User", "Workstation", "Apps", "System", "Schedule",
              "Browser", "Email", "Security", "Bio", "Threat", "Risk", "Clock"]


class BehaviourGraphEngine:
    """
    Builds a live Temporal Directed Multigraph from all 20 raw sensor signals.
    Provides a proper 12-dimensional feature embedding for the NCDE.
    """

    def __init__(self, memory_window: int = 30):
        """memory_window: number of past intervals to retain for pruning."""
        self.G = nx.MultiDiGraph()
        self.edge_history: deque = deque(maxlen=memory_window)
        self._init_nodes()

    def _init_nodes(self):
        for n in NODE_TYPES:
            self.G.add_node(n, type=n)

    def update_graph(self, signals: dict):
        ts = time.time()

        for sig_key, (src, dst, event) in SIGNAL_TO_EVENT.items():
            raw = signals.get(sig_key, 0)

            # Normalise boolean strings
            if isinstance(raw, str):
                raw = 1.0 if raw.upper() == "TRUE" else 0.0

            if raw and raw != 0:
                eid = self.G.add_edge(src, dst, event=event,
                                      weight=float(raw), ts=ts)
                self.edge_history.append((src, dst, eid, ts))

        # Prune edges older than 300 seconds (5 minutes)
        # This ensures that Demand (Dt) drops when the user is idle, 
        # even if no new signals are arriving.
        while self.edge_history:
            src, dst, eid, edge_ts = self.edge_history[0]
            if ts - edge_ts > 300: # 5 minute TTL
                self.edge_history.popleft()
                if self.G.has_edge(src, dst, key=eid):
                    self.G.remove_edge(src, dst, key=eid)
            else:
                break

    def get_embedding(self) -> torch.Tensor:
        """
        12-dimensional graph feature vector:
          [0]  Total active edges
          [1]  Mean edge weight
          [2]  Weighted in-degree of User node
          [3]  Weighted in-degree of Risk node
          [4]  Weighted out-degree of Threat node
          [5]  Weighted out-degree of Browser node
          [6]  Weighted out-degree of Email node
          [7]  Weighted in-degree of Security node
          [8]  Weighted out-degree of Bio node
          [9]  Weighted in-degree of Workstation
          [10] Number of distinct event types active
          [11] Recency score (edges in last 3 intervals / total)
        """
        edges = list(self.G.edges(data=True))
        n_edges = len(edges)

        if n_edges == 0:
            return torch.zeros(12, dtype=torch.float32)

        weights = [d.get("weight", 1.0) for _, _, d in edges]
        mean_w = float(np.mean(weights))

        def w_in(node):
            return sum(d.get("weight", 1.0)
                       for _, v, d in edges if v == node)

        def w_out(node):
            return sum(d.get("weight", 1.0)
                       for u, _, d in edges if u == node)

        distinct_events = len({d.get("event") for _, _, d in edges})
        now = time.time()
        recent = sum(1 for _, _, d in edges if now - d.get("ts", 0) < 30)
        recency = recent / max(1, n_edges)

        feat = [
            float(n_edges),
            mean_w,
            w_in("User"),
            w_in("Risk"),
            w_out("Threat"),
            w_out("Browser"),
            w_out("Email"),
            w_in("Security"),
            w_out("Bio"),
            w_in("Workstation"),
            float(distinct_events),
            recency,
        ]
        return torch.tensor(feat, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# LAYER 2 — NEURAL CONTROLLED DIFFERENTIAL EQUATIONS (NCDE)
# dZt = f_theta(Zt)dt + g_phi(Zt)dXt
# ─────────────────────────────────────────────────────────────

INPUT_DIM  = 12   # Graph embedding dimension (L1 output)
HIDDEN_DIM = 8    # Latent state Zt dimension (Ct, Dt, Ht, At + extras)
OUTPUT_DIM = 4    # Semantic outputs: [Ct, Dt, Ht, At]


class CDEFunc(nn.Module):
    """
    Parameterises g_phi(Zt): R^H → R^(H × X)
    The drift component f_theta(Zt) is the autonomous part inside the net.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # Two-layer MLP with tanh activation (bounded, suitable for dynamics)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim * input_dim),
            nn.Tanh(),
        )

    def forward(self, t, z):
        # z: (batch, hidden_dim)
        out = self.net(z)                              # (batch, H*X)
        return out.view(z.size(0), self.hidden_dim, self.input_dim)


class ReadoutHead(nn.Module):
    """Maps the NCDE hidden state Zt → [Ct, Dt, Ht, At]."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),   # keep all outputs in (0, 1)
        )

    def forward(self, z):
        return self.fc(z)


class NCDEModel(nn.Module):
    """Full NCDE model: initial encoder + CDE function + readout head."""

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 output_dim=OUTPUT_DIM):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.initial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.cde_func = CDEFunc(input_dim, hidden_dim)
        self.readout   = ReadoutHead(hidden_dim, output_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (batch, seq_len, input_dim)
        Returns: (batch, output_dim)  — final Zt semantic values
        """
        # --- Build spline path X(t) ---
        t_span = torch.linspace(0, 1, x_seq.size(1))
        coeffs  = torchcde.hermite_cubic_coefficients_with_backward_differences(
            x_seq, t_span)
        path    = torchcde.CubicSpline(coeffs)

        # --- Initial hidden state z0 from first observation ---
        z0 = self.initial_encoder(x_seq[:, 0, :])  # (batch, H)

        # --- Integrate CDE over [0, 1] ---
        t_eval = torch.tensor([0.0, 1.0])
        z_t    = torchcde.cdeint(X=path, func=self.cde_func,
                                  z0=z0, t=t_eval,
                                  method="rk4")      # (batch, 2, H)

        # --- Readout at final time ---
        z_final = z_t[:, -1, :]                      # (batch, H)
        return self.readout(z_final)                  # (batch, 4)


# ─────────────────────────────────────────────────────────────
# LAYER 3 — PERSONALIZED STATE SPACE (Bayesian Shrinkage)
# Zt(user) ~ N(mu_user, Sigma_user)
# ─────────────────────────────────────────────────────────────

class UserProfile:
    """
    Holds per-user MAP estimate of cognitive priors.
    Updated online via exponential moving average (a lightweight
    approximate Bayesian update without full VI).
    """

    def __init__(self, user_id: str = "default", alpha: float = 0.05):
        self.user_id = user_id
        # mu_user: [Ct_baseline, Dt_baseline, Ht_baseline, At_baseline]
        self.mu     = np.array([1.0, 0.1, 0.8, 0.0], dtype=np.float32)
        # sigma_user: diagonal covariance (stress sensitivity per dimension)
        self.sigma  = np.array([0.05, 0.1, 0.05, 0.02], dtype=np.float32)
        # EMA learning rate for online update
        self.alpha  = alpha
        self.n_obs  = 0

    def update(self, z_obs: np.ndarray):
        """
        Online MAP update: exponential moving average.
        mu_new = (1-alpha)*mu + alpha*z_obs
        """
        self.mu    = (1 - self.alpha) * self.mu + self.alpha * z_obs
        residual   = (z_obs - self.mu) ** 2
        self.sigma = (1 - self.alpha) * self.sigma + self.alpha * residual
        self.n_obs += 1

    def calibrate_from_history(self, history: np.ndarray):
        """
        Batch calibration from historical [Ct, Dt, Ht, At] data.
        Uses running mean/std as the initial prior.
        """
        if len(history) > 5:
            self.mu    = history.mean(axis=0).astype(np.float32)
            self.sigma = history.std(axis=0).astype(np.float32) + 1e-4
            self.n_obs = len(history)

    def shrink(self, z_model: np.ndarray, shrinkage: float = 0.85) -> np.ndarray:
        """
        Posterior estimate: blend model output with personalised prior.
        z_posterior = shrinkage * z_model + (1-shrinkage) * mu_user
        Shrinkage → 1.0 trusts model; → 0.0 trusts prior only.
        Increases toward 1.0 as n_obs grows.
        """
        adaptive = min(shrinkage, 0.5 + self.n_obs / 200.0)
        return adaptive * z_model + (1 - adaptive) * self.mu


# ─────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "ncde_weights.pt")
USER_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "user_profile.json")


def _load_training_data(csv_path: str, seq_len: int = 10):
    """
    Reads latent_states.csv and builds (X_seq, Y_target) tensors for training.
    X = sequence of [Ct, Dt, Ht, At] at time t-seq_len..t-1 (padded to INPUT_DIM via zeros)
    Y = [Ct, Dt, Ht, At] at time t
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path).dropna()
    if len(df) < seq_len + 2:
        return None, None

    cols = ["Ct", "Dt", "Ht", "At"]
    data = df[cols].values.astype(np.float32)

    X_seqs, Y_targets = [], []
    for i in range(seq_len, len(data)):
        x_raw = data[i - seq_len:i]             # (seq_len, 4)
        # Pad from 4 → INPUT_DIM (12) with zeros for the remaining graph features
        pad   = np.zeros((seq_len, INPUT_DIM - 4), dtype=np.float32)
        x_pad = np.concatenate([x_raw, pad], axis=1)  # (seq_len, 12)
        X_seqs.append(x_pad)
        Y_targets.append(data[i])

    return (torch.tensor(np.array(X_seqs)),
            torch.tensor(np.array(Y_targets)))


def train_ncde(model: NCDEModel,
               csv_path: str,
               epochs: int = 50,
               lr: float = 1e-3,
               seq_len: int = 10,
               verbose: bool = True):
    """
    Trains the NCDE model on historical latent state sequences.
    Loss: MSE between predicted [Ct, Dt, Ht, At] and next observed values.
    """
    X, Y = _load_training_data(csv_path, seq_len)
    if X is None:
        if verbose:
            print("[NCDE] Not enough historical data for training. Using random init.")
        return

    dataset  = torch.utils.data.TensorDataset(X, Y)
    loader   = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [NCDE Train] Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(loader):.5f}")

    torch.save(model.state_dict(), WEIGHTS_PATH)
    if verbose:
        print(f"[NCDE] Model weights saved -> {WEIGHTS_PATH}")


# ─────────────────────────────────────────────────────────────
# MAIN INFERENCE ENGINE (wires L1 + L2 + L3)
# ─────────────────────────────────────────────────────────────

class NeuralCDEInference:
    """
    Live inference engine.
      Step 1 — Update Behaviour Graph (L1)
      Step 2 — Append graph embedding to rolling window (L2 control path)
      Step 3 — Solve NCDE over the rolling window (L2)
      Step 4 — Personalise with user prior (L3)
    """

    SEQ_LEN    = 10
    DATA_CSV   = os.path.join(os.path.dirname(__file__),
                               "..", "data", "latent_states.csv")

    def __init__(self, user_id: str = "default", retrain: bool = False):
        self.graph   = BehaviourGraphEngine()
        self.model   = NCDEModel()
        self.profile = UserProfile(user_id)

        # Rolling window of graph embeddings (L2 control path)
        self.window: deque = deque(maxlen=self.SEQ_LEN)

        # --- Load or train weights ---
        if os.path.exists(WEIGHTS_PATH) and not retrain:
            try:
                self.model.load_state_dict(
                    torch.load(WEIGHTS_PATH, map_location="cpu"))
                print("[NCDE] Loaded pre-trained weights.")
            except Exception as e:
                print(f"[NCDE] Could not load weights ({e}). Retraining...")
                self._train()
        else:
            self._train()

        self.model.eval()

        # --- Calibrate user profile from history ---
        self._calibrate_profile()

        # Fallback state when window is not yet full
        self._last_state = {
            "Capacity_Ct": 1.0, "Demand_Dt": 0.1,
            "Habits_Ht": 0.8,   "Adversarial_At": 0.0,
            "Reserve_Gap_CRGt": 0.9
        }

    def _train(self):
        print("[NCDE] Training on historical data...")
        train_ncde(self.model, self.DATA_CSV, epochs=60, verbose=True)
        self.model.eval()

    def _calibrate_profile(self):
        """Fit user priors from historical latent_states.csv."""
        import pandas as pd
        if not os.path.exists(self.DATA_CSV):
            return
        df = pd.read_csv(self.DATA_CSV).dropna()
        if len(df) >= 5:
            cols = ["Ct", "Dt", "Ht", "At"]
            history = df[cols].values.astype(np.float32)
            self.profile.calibrate_from_history(history)
            print(f"[L3] User prior calibrated from {len(history)} samples.")
            print(f"     mu    = {self.profile.mu.round(3)}")
            print(f"     sigma = {self.profile.sigma.round(3)}")

    # ── Exponential Moving Average state for stability ──
    EMA_ALPHA = 0.10   # How fast EMA tracks new values (lower = smoother)
    MAX_SLEW  = 0.05   # Max change per tick (10s) for Ct, Ht, At


    def _compute_signal_driven_dt(self, signals: dict) -> float:
        """
        Compute Demand (Dt) directly from raw signals so it drops to ~0.1
        when the user is idle, rather than relying on the NCDE which can
        memorise high-demand patterns from training history.

        Formula (all clipped to [0,1]):
          Dt = 0.1 (idle floor)
               + 0.35 * typing_pressure   (keys vs baseline)
               + 0.20 * task_switch_norm  (context switches above noise floor)
               + 0.15 * notification_norm (audio interruptions)
               + 0.10 * calendar_pressure (workload_modifier if present)
               + 0.10 * sleep_penalty     (sleep deficit raises perceived demand)
        """
        # --- Typing pressure: how busy the user is vs their own baseline ---
        key_count  = float(signals.get('key_count', 0))
        base_keys  = max(1.0, float(signals.get('base_keys', 60)))
        typing_pressure = min(1.0, key_count / base_keys)

        # --- Task switching (VS Code noise already subtracted in tracker) ---
        task_sw = float(signals.get('task_switches', 0))
        task_switch_norm = min(1.0, task_sw / 10.0)  # >10 switches = max pressure

        # --- Notification pressure ---
        notifs = float(signals.get('notification_count', 0))
        notification_norm = min(1.0, notifs / 5.0)  # >5 = max

        # --- Calendar / workload modifier (0‒1 passed from context_api) ---
        workload = float(signals.get('workload_modifier', 0))
        calendar_pressure = min(1.0, workload)

        # --- Sleep deficit raises perceived demand ---
        sleep_str = str(signals.get('sleep_deficit', 'FALSE')).upper()
        sleep_penalty = 0.10 if sleep_str == 'TRUE' else 0.0

        dt_raw = (0.10
                  + 0.35 * typing_pressure
                  + 0.20 * task_switch_norm
                  + 0.15 * notification_norm
                  + 0.10 * calendar_pressure
                  + sleep_penalty)
        return float(np.clip(dt_raw, 0.10, 1.0))

    def _is_flat_embedding(self, window_list) -> bool:
        """
        Returns True when the rolling window is nearly constant —
        i.e. the user has been idle for several ticks and the NCDE
        path X(t) is so flat that RK4 integration becomes numerically
        unreliable (causes the Ct/Ht flicker observed in production).
        Threshold: std across the window is below 0.02.
        """
        stacked = torch.stack(window_list).numpy()   # (T, 12)
        return float(stacked.std(axis=0).mean()) < 0.05

    def update_inference(self, signals: dict) -> dict:
        # ── L1: Update Behaviour Graph ──
        self.graph.update_graph(signals)
        embedding = self.graph.get_embedding()   # (12,)
        self.window.append(embedding)

        # --- Signal-driven Dt (always reliable, never stuck at 1.0 during idle) ---
        dt_signal = self._compute_signal_driven_dt(signals)

        # Return last state (with updated Dt) if window not yet full
        if len(self.window) < 2:
            prev = dict(self._last_state)
            prev["Demand_Dt"] = round(dt_signal, 3)
            prev["Reserve_Gap_CRGt"] = round(prev["Capacity_Ct"] - dt_signal, 3)
            self._last_state = prev
            return self._last_state

        # ── Stability Gate: if embedding is flat (idle), skip NCDE and
        #    let the EMA gently decay Ct/Ht toward healthy baseline. ──
        window_list = list(self.window)
        if self._is_flat_embedding(window_list):
            # Blend last_state toward the user prior slowly (soft recovery)
            last_ct = self._last_state["Capacity_Ct"]
            last_ht = self._last_state["Habits_Ht"]
            last_at = self._last_state["Adversarial_At"]

            # Target: user prior (mu) if known, else healthy defaults
            target_ct = float(self.profile.mu[0])
            target_ht = float(self.profile.mu[2])
            target_at = float(self.profile.mu[3])

            RECOVERY_RATE = 0.05  # 5% per tick toward baseline during idle
            ct = last_ct + RECOVERY_RATE * (target_ct - last_ct)
            ht = last_ht + RECOVERY_RATE * (target_ht - last_ht)
            at = last_at + RECOVERY_RATE * (target_at - last_at)

            ct = float(np.clip(ct, 0.1, 1.0))
            ht = float(np.clip(ht, 0.0, 1.0))
            at = float(np.clip(at, 0.0, 1.0))
            dt = dt_signal  # always use signal-driven Dt

            self._last_state = {
                "Capacity_Ct":      round(ct, 3),
                "Demand_Dt":        round(dt, 3),
                "Habits_Ht":        round(ht, 3),
                "Adversarial_At":   round(at, 3),
                "Reserve_Gap_CRGt": round(ct - dt, 3),
            }
            return self._last_state

        # ── L2: Solve NCDE (only when the path has enough variation) ──
        x_seq = torch.stack(window_list).unsqueeze(0)  # (1, T, 12)

        with torch.no_grad():
            z_out = self.model(x_seq)   # (1, 4) — [Ct, Dt, Ht, At]

        z_np = z_out.squeeze(0).numpy()  # (4,)

        # ── L3: Personalise (Bayesian shrinkage toward user prior) ──
        z_personal = self.profile.shrink(z_np)

        # Online update of user profile
        self.profile.update(z_personal)

        # ── EMA Smoothing: prevent tick-to-tick oscillations ──
        # Initialise EMA state on first call
        if not hasattr(self, '_ema_state'):
            self._ema_state = np.array([1.0, 0.1, 0.8, 0.0], dtype=np.float32)

        alpha = self.EMA_ALPHA
        # NCDE contributes Ct, Ht, At; we override Dt with signal-driven value
        ncde_ct = float(np.clip(z_personal[0], 0.1, 1.0))
        ncde_ht = float(np.clip(z_personal[2], 0.0, 1.0))
        ncde_at = float(np.clip(z_personal[3], 0.0, 1.0))

        # --- Slew Rate Limit (Stability Force) ---
        # Don't allow values to jump more than MAX_SLEW from last state
        def apply_slew(current, target, max_delta):
            delta = target - current
            if abs(delta) > max_delta:
                return current + (max_delta if delta > 0 else -max_delta)
            return target

        last = self._last_state
        ncde_ct = apply_slew(last["Capacity_Ct"], ncde_ct, self.MAX_SLEW)
        ncde_ht = apply_slew(last["Habits_Ht"],   ncde_ht, self.MAX_SLEW)
        ncde_at = apply_slew(last["Adversarial_At"], ncde_at, self.MAX_SLEW)

        raw = np.array([ncde_ct, dt_signal, ncde_ht, ncde_at], dtype=np.float32)

        self._ema_state = (1 - alpha) * self._ema_state + alpha * raw

        ct = float(np.clip(self._ema_state[0], 0.1, 1.0))
        dt = float(np.clip(self._ema_state[1], 0.1, 1.0))
        ht = float(np.clip(self._ema_state[2], 0.0, 1.0))
        at = float(np.clip(self._ema_state[3], 0.0, 1.0))

        self._last_state = {
            "Capacity_Ct":      round(ct, 3),
            "Demand_Dt":        round(dt, 3),
            "Habits_Ht":        round(ht, 3),
            "Adversarial_At":   round(at, 3),
            "Reserve_Gap_CRGt": round(ct - dt, 3),
        }
        return self._last_state


# ─────────────────────────────────────────────────────────────
# LEGACY COMPAT — keep telemetry_tracker.py working unchanged
# ─────────────────────────────────────────────────────────────

class LatentStateInference(NeuralCDEInference):
    def __init__(self):
        super().__init__()
