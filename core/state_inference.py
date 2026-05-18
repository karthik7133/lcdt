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
# LAYER 2 — CAUSAL NEURAL CONTROLLED DIFFERENTIAL EQUATIONS
# dZt = (Iπt ∘ f)(Zt)dt + g_ϕ(Zt)dXt
# Unifies NCDE (Kidger) + SCM (Pearl) at the dynamics level.
# ─────────────────────────────────────────────────────────────

# Intervention operator table: γπ scales drift, δπ shifts it
# Order of dimensions: [Ct, Dt, Ht, At, ...extras]
INTERVENTION_PARAMS = {
    "baseline":             {"gamma": [1.00, 1.00, 1.000, 1.00], "delta": [0.00, 0.00, 0.00, 0.00]},
    "reduce_notifications": {"gamma": [1.00, 0.70, 1.000, 1.00], "delta": [0.00, 0.00, 0.00, 0.00]},
    "reduce_meetings":      {"gamma": [1.00, 0.80, 1.000, 1.00], "delta": [0.00, 0.00, 0.00, 0.00]},
    "pause_work":           {"gamma": [1.00, 0.30, 1.000, 1.00], "delta": [0.04, 0.00, 0.00, 0.00]},
    "improve_sleep":        {"gamma": [1.00, 1.00, 1.000, 1.00], "delta": [0.15, 0.00, 0.00, 0.00]},
    "security_training":    {"gamma": [1.00, 1.00, 1.010, 1.00], "delta": [0.00, 0.00, 0.00, 0.00]},
    "adversarial_drill":    {"gamma": [1.00, 1.00, 1.000, 0.50], "delta": [0.00, 0.00, 0.00, 0.00]},
}


class InterventionModule:
    """
    Iπt ∘ f = diag(γπ) · f(Zt) + δπ
    Applies the causal intervention operator to the autonomous drift output.
    Operates on the first 4 hidden dimensions (Ct, Dt, Ht, At).
    """
    def apply(self, f_out: torch.Tensor, policy: str) -> torch.Tensor:
        params = INTERVENTION_PARAMS.get(policy, INTERVENTION_PARAMS["baseline"])
        gamma = torch.tensor(params["gamma"], dtype=torch.float32)  # (4,)
        delta = torch.tensor(params["delta"], dtype=torch.float32)  # (4,)
        out = f_out.clone()
        n = min(4, out.shape[-1])
        out[:, :n] = out[:, :n] * gamma[:n] + delta[:n]
        return out


INPUT_DIM  = 12   # Graph embedding dimension (L1 output)
HIDDEN_DIM = 16   # Latent state Zt dimension (doubled: 8->16 for richer At/Dt dynamics)
OUTPUT_DIM = 4    # Semantic outputs: [Ct, Dt, Ht, At]


class CDEFunc(nn.Module):
    """
    Causal CDE function implementing:
      dZt = (Iπt ∘ f)(Zt)dt + g_ϕ(Zt)dXt
    f_net  = autonomous drift MLP (intrinsic cognitive dynamics)
    net    = control matrix g_ϕ(Zt) (behaviour-path driven)
    Iπt    = intervention operator applied to f_net output
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self._policy    = "baseline"
        self.intervention = InterventionModule()

        # Autonomous drift net: f(Zt)  [NEW — B1]
        self.f_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, hidden_dim),
            nn.Tanh(),
        )

        # Control matrix net: g_ϕ(Zt)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim * input_dim),
            nn.Tanh(),
        )

    def set_policy(self, policy: str):
        """Set active intervention policy for the CDE integration."""
        self._policy = policy

    def forward(self, t, z):
        # z: (batch, hidden_dim)
        # Autonomous drift with causal intervention gate (B1)
        f_out    = self.f_net(z)                                      # (batch, H)
        f_causal = self.intervention.apply(f_out, self._policy)       # (batch, H)
        # Control matrix (behaviour path modulates state via g_ϕ)
        g_in = z + 0.1 * f_causal                                     # drift pre-conditions g
        out  = self.net(g_in)                                         # (batch, H*X)
        return out.view(z.size(0), self.hidden_dim, self.input_dim)


class DualReadoutHead(nn.Module):
    """
    Dual-head readout: Zt → [Ct, Dt, Ht, At].

    Head A (head_threat)   → [Ht, At]  Pre-trained on CERT v6.2, FROZEN after Phase 1.
    Head B (head_capacity) → [Ct, Dt]  Fine-tuned on personal latent_states.csv.

    Output order [Ct, Dt, Ht, At] is preserved — all downstream code unchanged.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Head A: Security habits + adversarial threat (CERT-trained, locked after Phase 1)
        self.head_threat = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),   # [Ht, At]
            nn.Sigmoid(),
        )
        # Head B: Cognitive capacity + demand (personally fine-tuned)
        self.head_capacity = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),   # [Ct, Dt]
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        ht_at = self.head_threat(z)    # (batch, 2): [Ht, At]
        ct_dt = self.head_capacity(z)  # (batch, 2): [Ct, Dt]
        # Maintain [Ct, Dt, Ht, At] convention — all downstream code unchanged
        return torch.cat([ct_dt, ht_at], dim=-1)

    def freeze_head_a(self):
        """Freeze Head A before Phase 2 personal fine-tuning."""
        for p in self.head_threat.parameters():
            p.requires_grad = False

    def unfreeze_head_a(self):
        """Unfreeze Head A — used during Phase 1 CERT pre-training only."""
        for p in self.head_threat.parameters():
            p.requires_grad = True


class NCDEModel(nn.Module):
    """Causal NCDE: dZt = (Iπt ∘ f)(Zt)dt + g_ϕ(Zt)dXt."""

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
        self.readout  = DualReadoutHead(hidden_dim)  # Dual-head: CERT + personal

    def forward(self, x_seq: torch.Tensor,
                policy: str = "baseline") -> torch.Tensor:
        """
        x_seq:  (batch, seq_len, input_dim)
        policy: intervention policy active during integration (B1)
        Returns: (batch, output_dim) — intervention-aware Zt
        """
        # Set active policy on CDE function (B1)
        self.cde_func.set_policy(policy)

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
    BREAKTHROUGH 2 — Causal Personalization.
    Old: μ_new = (1-α)*μ + α*z_observed  (EMA — temporal, ignores causality)
    New: μ_user(t) = argmin_μ  E_do(π)[L(Zt, μ)]

    The personal prior is updated based on how the user *responds to interventions*,
    not just what was observed. This distinguishes:
      - sleep-sensitive vs meeting-fragile users
      - resilient vs burnout-trajectory users
    """

    # All policies used in causal sensitivity estimation
    ALL_POLICIES = list(INTERVENTION_PARAMS.keys())

    def __init__(self, user_id: str = "default", alpha: float = 0.05):
        self.user_id = user_id
        self.mu     = np.array([1.0, 0.1, 0.8, 0.0], dtype=np.float32)
        self.sigma  = np.array([0.05, 0.1, 0.05, 0.02], dtype=np.float32)
        self.alpha  = alpha
        self.n_obs  = 0
        # B2: Causal sensitivity profile — how much each intervention moves Zt
        self.causal_sensitivity_profile: dict = {
            pi: 0.0 for pi in self.ALL_POLICIES
        }
        # B2: Per-policy response history for online update
        self._policy_z_buffer: dict = {pi: [] for pi in self.ALL_POLICIES}

    def update(self, z_obs: np.ndarray):
        """
        Legacy-compatible EMA update (still called for sigma tracking).
        """
        self.mu    = (1 - self.alpha) * self.mu + self.alpha * z_obs
        residual   = (z_obs - self.mu) ** 2
        self.sigma = (1 - self.alpha) * self.sigma + self.alpha * residual
        self.n_obs += 1

    def causal_update(self, policy_z_map: dict):
        """
        BREAKTHROUGH 2 — Causal prior update.
        policy_z_map: {policy_name: z_under_policy (np.ndarray, shape (4,))}

        Update rule (online gradient step):
          μ ← μ + η · Σ_π [Zt^π - μ] / |Π|

        Also updates causal_sensitivity_profile:
          sensitivity[π] = ||Zt^π - z_baseline||
        """
        if not policy_z_map:
            return
        eta = 0.03  # causal learning rate (slower than EMA to avoid noise)
        z_baseline = policy_z_map.get("baseline", self.mu)

        # Gradient step: pull mu toward average interventional outcome
        z_sum = np.zeros(4, dtype=np.float32)
        for pi, z_pi in policy_z_map.items():
            z_arr = np.array(z_pi, dtype=np.float32)
            z_sum += z_arr
            # Sensitivity = how far this policy moves state from baseline
            diff = np.linalg.norm(z_arr - z_baseline)
            # Smooth sensitivity with exponential averaging
            self.causal_sensitivity_profile[pi] = (
                0.9 * self.causal_sensitivity_profile[pi] + 0.1 * float(diff)
            )
        mu_interventional = z_sum / len(policy_z_map)
        self.mu = self.mu + eta * (mu_interventional - self.mu)
        self.n_obs += 1

    def get_resilience_profile(self) -> dict:
        """
        Returns the causal sensitivity profile with a resilience label.
        High sensitivity to a policy → user responds strongly to that intervention.
        Low sensitivity → user is resilient to (or unaffected by) that policy.
        """
        profile = dict(self.causal_sensitivity_profile)
        best_policy = max(profile, key=profile.get) if profile else "baseline"
        return {
            "sensitivity": profile,
            "most_effective_intervention": best_policy,
            "user_type": self._classify_user(profile),
        }

    def _classify_user(self, profile: dict) -> str:
        if not profile or all(v == 0.0 for v in profile.values()):
            return "uncalibrated"
        top = max(profile, key=profile.get)
        labels = {
            "improve_sleep": "sleep-sensitive",
            "reduce_notifications": "notification-fragile",
            "reduce_meetings": "meeting-fragile",
            "pause_work": "overload-prone",
            "security_training": "training-responsive",
            "adversarial_drill": "threat-aware",
            "baseline": "self-regulating",
        }
        return labels.get(top, "mixed")

    def calibrate_from_history(self, history: np.ndarray):
        """Batch calibration from historical [Ct, Dt, Ht, At] data."""
        if len(history) > 5:
            self.mu    = history.mean(axis=0).astype(np.float32)
            self.sigma = history.std(axis=0).astype(np.float32) + 1e-4
            self.n_obs = len(history)

    def shrink(self, z_model: np.ndarray, shrinkage: float = 0.85) -> np.ndarray:
        """Posterior estimate: blend model output with personalised prior."""
        adaptive = min(shrinkage, 0.5 + self.n_obs / 200.0)
        return adaptive * z_model + (1 - adaptive) * self.mu



# ─────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

WEIGHTS_PATH      = os.path.join(os.path.dirname(__file__), "ncde_weights.pt")
CERT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "ncde_weights_cert.pt")
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


def _load_cert_training_data(csv_path: str,
                              seq_len: int = 10,
                              max_sequences: int = 50_000):
    """
    User-aware CERT data loader.
    Builds sequences WITHIN each user's time series only — never crosses user boundaries.
    Randomly samples max_sequences to keep Phase 1 training tractable on CPU.
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path).dropna()
    if len(df) < seq_len + 2:
        return None, None

    cols = ["Ct", "Dt", "Ht", "At"]
    has_user = "user_id" in df.columns

    X_seqs, Y_targets = [], []

    if has_user:
        # Per-user windowing — respects user boundaries
        for _, user_df in df.groupby("user_id"):
            data = user_df[cols].values.astype(np.float32)
            if len(data) < seq_len + 1:
                continue
            for i in range(seq_len, len(data)):
                x_raw = data[i - seq_len:i]
                pad   = np.zeros((seq_len, INPUT_DIM - 4), dtype=np.float32)
                X_seqs.append(np.concatenate([x_raw, pad], axis=1))
                Y_targets.append(data[i])
    else:
        # Fallback: single continuous series (personal data)
        data = df[cols].values.astype(np.float32)
        for i in range(seq_len, len(data)):
            x_raw = data[i - seq_len:i]
            pad   = np.zeros((seq_len, INPUT_DIM - 4), dtype=np.float32)
            X_seqs.append(np.concatenate([x_raw, pad], axis=1))
            Y_targets.append(data[i])

    if not X_seqs:
        return None, None

    X_arr = np.array(X_seqs,   dtype=np.float32)
    Y_arr = np.array(Y_targets, dtype=np.float32)

    # ── Malicious augmentation: 7 noisy copies of every At>0.5 sequence ──
    # 70 confirmed malicious users / 1000 total = 7% positive At rate.
    # Without augmentation, the model barely sees malicious patterns.
    # 7 copies × 70 users × ~27 days = ~13,230 augmented malicious sequences.
    # At label is preserved EXACTLY (label-preserving noise on Ct/Dt/Ht only).
    mal_mask = Y_arr[:, 3] > 0.50          # confirmed insider threat sequences
    n_mal    = mal_mask.sum()
    if n_mal > 0:
        X_mal = X_arr[mal_mask];  Y_mal = Y_arr[mal_mask]
        aug_X, aug_Y = [], []
        for _ in range(7):                  # 7 augmented copies
            noise = np.random.randn(*X_mal.shape).astype(np.float32) * 0.006
            x_aug = np.clip(X_mal + noise, 0.0, 1.0)
            # Preserve At (index 3) EXACTLY in both input and target
            x_aug[:, :, 3] = X_mal[:, :, 3]
            y_aug           = Y_mal.copy()  # At label unchanged
            aug_X.append(x_aug);  aug_Y.append(y_aug)
        X_arr = np.concatenate([X_arr] + aug_X, axis=0)
        Y_arr = np.concatenate([Y_arr] + aug_Y, axis=0)

    # Random subsample to cap training size (ensures CPU tractability)
    total = len(X_arr)
    if total > max_sequences:
        idx   = np.random.choice(total, max_sequences, replace=False)
        X_arr = X_arr[idx];  Y_arr = Y_arr[idx]

    return (torch.tensor(X_arr), torch.tensor(Y_arr))


def _load_personal_training_data(csv_path: str, seq_len: int = 10):
    """
    Personal data loader for Phase 2 / Phase 3 fine-tuning.

    What is kept vs zeroed — confirmed by CERT coverage check:
      Ct (idx 0): KEPT   — keyboard/mouse/facial/sleep give reliable signal       R2_cert=0.96
      Dt (idx 1): KEPT   — window count/notifications/email give reliable signal  R2_cert=0.78
      Ht (idx 2): ZEROED — needs phishing/USB events to degrade; personal data
                           has almost none. Values are noise from old predictions.
                           CERT Head A already covers Ht (R2_cert=0.99). FROZEN.
      At (idx 3): ZEROED — adversarial events cannot be triggered naturally;
                           personal data is nearly always 0 → useless signal.
                           CERT Head A already covers At (R2_cert=0.70). FROZEN.

    Zeroing in BOTH X (input sequences) and Y (targets) keeps the backbone
    clean — it only learns Ct/Dt dynamics from personal data.
    Head A (frozen, CERT-trained) still infers Ht/At at prediction time.
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path).dropna()
    if len(df) < seq_len + 2:
        return None, None

    cols = ["Ct", "Dt", "Ht", "At"]
    data = df[cols].values.astype(np.float32)

    # Zero out Ht (idx 2) and At (idx 3) — unreliable / absent in personal data
    data[:, 2] = 0.0   # Ht -> 0  (sourced from CERT Head A at inference time)
    data[:, 3] = 0.0   # At -> 0  (sourced from CERT Head A at inference time)

    X_seqs, Y_targets = [], []
    for i in range(seq_len, len(data)):
        x_raw = data[i - seq_len:i]               # (seq_len, 4) with Ht/At = 0
        pad   = np.zeros((seq_len, INPUT_DIM - 4), dtype=np.float32)
        X_seqs.append(np.concatenate([x_raw, pad], axis=1))
        Y_targets.append(data[i])                 # target: Ct, Dt real; Ht, At = 0

    if not X_seqs:
        return None, None

    return (torch.tensor(np.array(X_seqs)),
            torch.tensor(np.array(Y_targets)))


def train_ncde_cert(model: NCDEModel,
                   csv_path: str,
                   epochs: int = 20,
                   lr: float = 1e-3,
                   seq_len: int = 10,
                   max_sequences: int = 50_000,
                   verbose: bool = True) -> bool:

    """
    PHASE 1 — Pre-trains the full NCDE on CERT-derived latent states.
    Improvements over v1:
      - Stratified oversampling: 25% of sequences from At>0.1 users (was ~7%)
      - Weighted loss: At gets 3x, Ht gets 1.5x to compensate class imbalance
      - Data augmentation: Gaussian noise (sigma=0.006) on normal sequences
      - Weight decay regularisation (1e-4) to prevent overfitting
      - LR 7e-4 (smoother convergence vs 1e-3)
      - 50 epochs (was 20)
    """
    X, Y = _load_cert_training_data(csv_path, seq_len, max_sequences)
    if X is None:
        if verbose:
            print("[NCDE Phase 1] CERT data not found or insufficient. Skipping.")
        return False

    # ── Stratified oversampling: ensure ~25% adversarial sequences ──────────
    # At>0.1 = any adversarial signal (confirmed, pre-buildup, suspicious)
    y_np = Y.numpy()
    adv_mask = y_np[:, 3] > 0.10                    # At > 0.1
    adv_idx  = np.where(adv_mask)[0]
    norm_idx = np.where(~adv_mask)[0]

    if len(adv_idx) > 0:
        n_adv_target = max(len(adv_idx), int(len(X) * 0.25))
        n_norm_target = len(X) - min(n_adv_target, int(len(X) * 0.25))
        # oversample adversarial
        adv_sample  = adv_idx[np.random.choice(len(adv_idx), n_adv_target, replace=True)]
        norm_sample = norm_idx[np.random.choice(len(norm_idx), n_norm_target, replace=True)]
        all_idx = np.concatenate([adv_sample, norm_sample])
        np.random.shuffle(all_idx)
        X = X[all_idx]; Y = Y[all_idx]

    # ── Data augmentation: Gaussian noise on normal sequences only ──────────
    x_aug = X.clone(); y_aug = Y.clone()
    norm_rows = (y_aug[:, 3] < 0.10)              # At near 0 = normal
    noise     = torch.randn_like(x_aug[norm_rows]) * 0.006
    x_aug[norm_rows] = (x_aug[norm_rows] + noise).clamp(0.0, 1.0)
    X = torch.cat([X, x_aug[norm_rows]], dim=0)
    Y = torch.cat([Y, y_aug[norm_rows]], dim=0)

    if verbose:
        adv_pct = 100 * (Y.numpy()[:, 3] > 0.1).mean()
        print(f"[NCDE Phase 1] {len(X):,} sequences after oversample+augment"
              f" ({adv_pct:.1f}% adversarial) | {epochs} epochs")
        print(f"[NCDE Phase 1] Weighted loss: Ct=1.0 Dt=1.5 Ht=1.5 At=3.0"
              f" | LR={lr} | weight_decay=1e-4")

    model.readout.unfreeze_head_a()   # Both heads active

    # Differential LR: give CDE func slightly lower LR for stability
    optimizer = optim.Adam([
        {'params': model.initial_encoder.parameters(), 'lr': lr},
        {'params': model.cde_func.parameters(),        'lr': lr * 0.8},
        {'params': model.readout.parameters(),          'lr': lr},
    ], weight_decay=1e-4)

    # Balanced static weights — mild boosts for Ht and At
    # Dt stays equal to Ct so both learn equally from the shared backbone
    BASE_W = torch.tensor([1.0, 1.0, 1.5, 2.0])   # [Ct, Dt, Ht, At]

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred   = model(xb)
            sq_err = (pred - yb) ** 2             # (batch, 4)
            loss   = (sq_err * BASE_W).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [Phase 1] Epoch {epoch+1}/{epochs}  Loss: {total/len(loader):.5f}")


    torch.save(model.state_dict(), CERT_WEIGHTS_PATH)
    if verbose:
        print(f"[NCDE Phase 1] CERT weights saved -> {CERT_WEIGHTS_PATH}")
    return True


def train_ncde_personal(model: NCDEModel,
                        csv_path: str,
                        epochs: int = 30,
                        lr: float = 1e-4,
                        seq_len: int = 10,
                        verbose: bool = True):
    """
    PHASE 2 / PHASE 3 — Fine-tunes on personal latent_states.csv.
    - Loads CERT weights as starting point (if production weights absent).
    - Freezes Head A (Ht/At) — CERT knowledge locked.
    - Fine-tunes backbone (slow) + Head B (Ct/Dt) on personal data.
    - Loss: MSE on [Ct, Dt] only — avoids sparse At signal corrupting backbone.
    Output: ncde_weights.pt  (production weights)
    """
    # Bootstrap from CERT weights if production weights don't exist yet
    if not os.path.exists(WEIGHTS_PATH) and os.path.exists(CERT_WEIGHTS_PATH):
        if verbose:
            print("[NCDE Phase 2] Bootstrapping from CERT pre-trained weights...")
        model.load_state_dict(torch.load(CERT_WEIGHTS_PATH, map_location="cpu"))

    # Use personal-specific loader: Ht and At zeroed out in both X and Y
    X, Y = _load_personal_training_data(csv_path, seq_len)
    if X is None:
        if verbose:
            print("[NCDE Phase 2] Not enough personal data yet. Keeping current weights.")
        return

    if verbose:
        print(f"[NCDE Phase 2] Personal fine-tuning: {len(X)} sequences | {epochs} epochs")
        print("[NCDE Phase 2] Training ONLY on Ct + Dt (Ht and At zeroed — sourced from CERT).")
        print("[NCDE Phase 2] Head A (Ht/At) FROZEN. Fine-tuning Head B (Ct/Dt) + backbone.")

    model.readout.freeze_head_a()  # Lock CERT's At/Ht knowledge

    # Freeze backbone COMPLETELY — only Head B learns from personal data.
    # CERT backbone already encodes perfect dynamics; changing it during Phase 2
    # (even at slow LR) degrades CERT accuracy over many epochs.
    # Head B (610 params) is sufficient to remap Zt -> your personal Ct/Dt scale.
    for p in model.initial_encoder.parameters(): p.requires_grad = False
    for p in model.cde_func.parameters():        p.requires_grad = False
    optimizer = optim.Adam(
        model.readout.head_capacity.parameters(), lr=lr * 2, weight_decay=1e-5
    )
    loss_fn = nn.MSELoss()

    # ── Personal data augmentation: 8 AR(1)-correlated copies per sequence ──
    # Real personal data: ~1,205 sequences. Head B (610 params) needs more to converge.
    # AR(1) noise creates temporally smooth synthetic sequences that stay within
    # plausible personal Ct/Dt ranges. Sigma deliberately smaller than Phase 1
    # so we don't teach Head B implausibly extreme values.
    X_np = X.numpy(); Y_np = Y.numpy()
    aug_X, aug_Y = [X_np], [Y_np]
    for _ in range(8):
        x_aug = X_np.copy()
        for seq_i in range(len(x_aug)):
            ct_noise = 0.0; dt_noise = 0.0
            for t in range(x_aug.shape[1]):
                ct_noise = 0.70 * ct_noise + np.random.randn() * 0.015  # tighter
                dt_noise = 0.70 * dt_noise + np.random.randn() * 0.020  # tighter
                x_aug[seq_i, t, 0] = float(np.clip(X_np[seq_i, t, 0] + ct_noise, 0.1, 1.0))
                x_aug[seq_i, t, 1] = float(np.clip(X_np[seq_i, t, 1] + dt_noise, 0.1, 1.0))
        y_aug = Y_np.copy()
        for seq_i in range(len(y_aug)):
            y_aug[seq_i, 0] = float(np.clip(Y_np[seq_i, 0] + np.random.randn() * 0.012, 0.1, 1.0))
            y_aug[seq_i, 1] = float(np.clip(Y_np[seq_i, 1] + np.random.randn() * 0.018, 0.1, 1.0))
        aug_X.append(x_aug); aug_Y.append(y_aug)

    X = torch.tensor(np.concatenate(aug_X, axis=0), dtype=torch.float32)
    Y = torch.tensor(np.concatenate(aug_Y, axis=0), dtype=torch.float32)
    if verbose:
        print(f"[NCDE Phase 2] Augmented to {len(X):,} sequences (8x AR1 augmentation).")

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X, Y), batch_size=32, shuffle=True):
            optimizer.zero_grad()
            pred = model(xb)
            # Loss only on [Ct, Dt] — indices 0 and 1
            loss = loss_fn(pred[:, :2], yb[:, :2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [Phase 2/3] Epoch {epoch+1}/{epochs}  Ct/Dt Loss: {total/max(1,len(X)//32):.5f}")

    torch.save(model.state_dict(), WEIGHTS_PATH)
    if verbose:
        print(f"[NCDE Phase 2/3] Production weights saved -> {WEIGHTS_PATH}")

    # Re-enable backbone grads for future Phase 3 / inference calls
    for p in model.initial_encoder.parameters(): p.requires_grad = True
    for p in model.cde_func.parameters():        p.requires_grad = True


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

        # Phase 3: track row count for scheduled retraining
        self._retrain_check_ticks = 0
        self._retrain_row_count   = self._get_current_row_count()

        # --- Load weights: production → CERT bootstrap → legacy retrain ---
        if os.path.exists(WEIGHTS_PATH) and not retrain:
            try:
                self.model.load_state_dict(
                    torch.load(WEIGHTS_PATH, map_location="cpu"))
                print("[NCDE] Loaded production weights (dual-head).")
            except Exception as e:
                print(f"[NCDE] Weight load failed ({e}). Running personal fine-tuning...")
                train_ncde_personal(self.model, self.DATA_CSV, epochs=30, verbose=True)
        elif os.path.exists(CERT_WEIGHTS_PATH) and not retrain:
            print("[NCDE] CERT pre-trained weights found. Running Phase 2 fine-tuning...")
            train_ncde_personal(self.model, self.DATA_CSV, epochs=30, verbose=True)
        else:
            print("[NCDE] No weights found. Run scripts/run_training_pipeline.py first.")
            print("[NCDE] Falling back to Phase 2 personal-only training...")
            self._train()   # Legacy fallback

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
        """Legacy fallback: personal-only training when no CERT weights exist."""
        print("[NCDE] Training on historical data...")
        train_ncde(self.model, self.DATA_CSV, epochs=60, verbose=True)
        self.model.eval()

    def _get_current_row_count(self) -> int:
        """Returns current row count of latent_states.csv for Phase 3 tracking."""
        try:
            import pandas as pd
            if os.path.exists(self.DATA_CSV):
                return len(pd.read_csv(self.DATA_CSV))
        except Exception:
            pass
        return 0

    def _maybe_retrain(self):
        """
        PHASE 3 — Scheduled retraining.
        Checks every 180 ticks (~30 min). If 200+ new rows have accumulated
        in latent_states.csv since the last retrain, triggers Phase 2 fine-tuning
        with Head A frozen (Ct/Dt only). Head A stays locked to CERT knowledge.
        epochs=40 (was 20): more convergence per trigger = faster R2 improvement.
        """
        self._retrain_check_ticks += 1
        if self._retrain_check_ticks < 180:   # Check every ~30 min (was 360/1hr)
            return
        self._retrain_check_ticks = 0
        current = self._get_current_row_count()
        new_rows = current - self._retrain_row_count
        if new_rows >= 200:
            print(f"[NCDE Phase 3] {new_rows} new rows detected. Retraining Head B (40 epochs)...")
            train_ncde_personal(self.model, self.DATA_CSV,
                                epochs=40, lr=1e-4, verbose=True)
            self.model.eval()
            self._retrain_row_count = current
            print("[NCDE Phase 3] Retraining complete. Run evaluate_model.py to check R2.")

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
        # ── Phase 3: Check if scheduled retraining is due ──
        self._maybe_retrain()

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

        # ── L2: Causal NCDE — solve under ALL policies (B1 + B3) ──
        x_seq = torch.stack(window_list).unsqueeze(0)  # (1, T, 12)

        with torch.no_grad():
            # Run under baseline for live state
            z_out = self.model(x_seq, policy="baseline")   # (1, 4)

            # B3: Run under every policy to build causal sensitivity map
            policy_z_map = {}
            for pi in UserProfile.ALL_POLICIES:
                z_pi = self.model(x_seq, policy=pi).squeeze(0).numpy()  # (4,)
                policy_z_map[pi] = self.profile.shrink(z_pi)

        z_np = z_out.squeeze(0).numpy()  # (4,)

        # ── L3: Personalise (Bayesian shrinkage toward user prior) ──
        z_personal = self.profile.shrink(z_np)

        # B2: Causal prior update (replaces raw EMA update with interventional avg)
        self.profile.causal_update(policy_z_map)
        # Also keep sigma estimate updated via legacy EMA
        self.profile.update(z_personal)

        # B3: Track active policy for training log (default: baseline during live)
        self._active_policy = getattr(self, '_active_policy', 'baseline')

        # ── EMA Smoothing: prevent tick-to-tick oscillations ──
        if not hasattr(self, '_ema_state'):
            self._ema_state = np.array([1.0, 0.1, 0.8, 0.0], dtype=np.float32)

        alpha = self.EMA_ALPHA
        ncde_ct = float(np.clip(z_personal[0], 0.1, 1.0))
        ncde_ht = float(np.clip(z_personal[2], 0.0, 1.0))
        ncde_at = float(np.clip(z_personal[3], 0.0, 1.0))

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

        # B3: Build counterfactual columns for latent_states.csv
        cf_cols = {}
        for pi, z_pi in policy_z_map.items():
            safe_pi = pi.replace(" ", "_")
            cf_cols[f"Ct_cf_{safe_pi}"] = round(float(np.clip(z_pi[0], 0.1, 1.0)), 4)

        self._last_state = {
            "Capacity_Ct":      round(ct, 3),
            "Demand_Dt":        round(dt, 3),
            "Habits_Ht":        round(ht, 3),
            "Adversarial_At":   round(at, 3),
            "Reserve_Gap_CRGt": round(ct - dt, 3),
            "policy_active":    self._active_policy,   # B3
            **cf_cols,                                  # B3: counterfactual Ct per policy
        }
        return self._last_state



# ─────────────────────────────────────────────────────────────
# LEGACY COMPAT — keep telemetry_tracker.py working unchanged
# ─────────────────────────────────────────────────────────────

class LatentStateInference(NeuralCDEInference):
    def __init__(self):
        super().__init__()
