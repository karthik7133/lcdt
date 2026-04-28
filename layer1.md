# 🔬 Layer 1 — Multimodal Behaviour Graph
**File:** `core/state_inference.py` → `BehaviourGraphEngine`

---

## ❌ Does Layer 1 Use SVM?

**No. SVM (Support Vector Machine) is not used anywhere in Layer 1 — or anywhere in the entire LCDT system.**

Layer 1 is purely **graph-theoretic**. It uses a **Temporal Directed Multigraph** (NetworkX `MultiDiGraph`) to model the
employee's behavioural signals as live, time-stamped edges between semantic nodes. No classifier of any kind (SVM, 
decision tree, logistic regression) runs inside Layer 1. The only ML model in the pipeline begins at Layer 2 (NCDE).

---

## 🧠 What Layer 1 Actually Does

Layer 1 has **three jobs**:
1. **Receive** all 20 raw sensor signals every 10 seconds
2. **Encode** each signal as a directed edge in a live temporal graph
3. **Embed** the graph topology into a compact **12-dimensional numeric vector** that Layer 2 (NCDE) can consume

---

## 🗺️ Graph Structure

The graph is a **NetworkX `MultiDiGraph`** — directed (edges have a source and target), multi (multiple parallel
edges are allowed between the same pair of nodes), and temporal (each edge carries a timestamp).

### Nodes (12 Semantic Entities)

| Node | Represents |
|------|-----------|
| `User` | The employee — receives cognitive load signals |
| `Workstation` | Keyboard/mouse activity destination |
| `Apps` | Application context switches |
| `System` | OS-level signals (notifications, updates) |
| `Schedule` | Google Calendar workload data |
| `Browser` | Web browsing behaviour |
| `Email` | Email client activity |
| `Security` | Password hygiene signals |
| `Bio` | Biometric / physiological signals (sleep, vision fatigue) |
| `Threat` | Active adversarial events (phishing, credential theft) |
| `Risk` | Risk accumulation sink node |
| `Clock` | Circadian/time-of-day signals |

---

## 📡 The 20 Signals — Full Topology Table

Each raw sensor signal is mapped to a **directed edge** `(Source → Target)` with an `event_type` label and the
raw sensor value as the **edge weight**.

| Signal | Source → Target | Event Type | What It Measures |
|--------|----------------|-----------|-----------------|
| `key_count` | User → Workstation | `typing` | Keystrokes per interval |
| `mouse_entropy` | User → Workstation | `mouse_movement` | Randomness/speed of mouse path |
| `typing_error_rate` | User → Workstation | `typing_friction` | Backspace-corrected error rate |
| `task_switches` | Apps → User | `context_switch` | App-switching frequency |
| `notification_count` | System → User | `interruption` | Audio / popup interrupts |
| `workload_modifier` | Schedule → User | `calendar_pressure` | Meeting density from Google Calendar |
| `insecure_http_hits` | Browser → Risk | `insecure_browse` | HTTP (non-HTTPS) page visits |
| `webmail_hits` | Browser → Risk | `webmail_access` | Webmail site visits |
| `link_clicks` | Browser → Risk | `link_click` | Any external link clicks |
| `email_frequency` | Email → User | `email_activity` | Emails sent/received per interval |
| `unknown_senders` | Email → Risk | `phishing_signal` | Emails from unrecognized senders |
| `avg_response_time` | Email → User | `email_response` | Mean time to reply to emails |
| `low_strength_passwords` | Security → Risk | `weak_password` | Weak passwords in use |
| `good_password_paste` | Security → User | `good_habit` | Clipboard manager / good password usage |
| `os_update_delayed` | System → Risk | `update_delay` | OS patch outstanding > threshold |
| `sleep_deficit` | Bio → User | `sleep_debt` | Boolean: slept < 6h |
| `vision_fatigue` | Bio → User | `vision_fatigue` | EAR/MAR/head-pose score from camera |
| `phishing_clicked` | Threat → Risk | `phishing_click` | Clicked a simulated or real phishing link |
| `scam_credentials_given` | Threat → Risk | `credential_theft` | Submitted credentials to a scam site |
| `hour_of_day` | Clock → User | `circadian` | Hour of the day (0–23) for fatigue curve |

---

## ⏱️ 5-Minute TTL (Time-To-Live) — Edge Pruning

Every edge is stamped with a Unix timestamp when added.
A background loop checks the **oldest edge** in the deque at every update cycle.
If its timestamp is older than **300 seconds (5 minutes)**, it is **deleted from the graph**.

```python
while self.edge_history:
    src, dst, eid, edge_ts = self.edge_history[0]
    if ts - edge_ts > 300:  # 5 minute TTL
        self.edge_history.popleft()
        if self.G.has_edge(src, dst, key=eid):
            self.G.remove_edge(src, dst, key=eid)
    else:
        break
```

**Why this matters:** Without TTL pruning, the `Risk` node would accumulate stale edges from past sessions and
Cognitive Demand (Dt) would never fall to idle even when the user is not working — causing the "stuck high risk"
bug observed in early versions.

---

## 📐 The 12-Dimensional Graph Embedding

After the graph is updated, `get_embedding()` reads the live graph topology and extracts 12 numeric features
into a `torch.Tensor` of shape `(12,)`. This is the **input to the NCDE in Layer 2**.

| Index | Feature | Calculation |
|-------|---------|------------|
| `[0]` | Total active edges | `len(G.edges)` |
| `[1]` | Mean edge weight | `mean(all edge weights)` |
| `[2]` | Weighted in-degree of `User` | Sum of weights of all edges pointing **into** `User` |
| `[3]` | Weighted in-degree of `Risk` | Sum of weights of all edges pointing **into** `Risk` |
| `[4]` | Weighted out-degree of `Threat` | Sum of weights of all edges **from** `Threat` |
| `[5]` | Weighted out-degree of `Browser` | Sum of weights of all edges **from** `Browser` |
| `[6]` | Weighted out-degree of `Email` | Sum of weights of all edges **from** `Email` |
| `[7]` | Weighted in-degree of `Security` | Sum of weights of all edges pointing **into** `Security` |
| `[8]` | Weighted out-degree of `Bio` | Sum of weights of all edges **from** `Bio` |
| `[9]` | Weighted in-degree of `Workstation` | Sum of weights of all edges pointing **into** `Workstation` |
| `[10]` | Distinct active event types | `len({edge.event for edge in G.edges})` |
| `[11]` | Recency score | `(edges updated in last 30s) / total_edges` |

If the graph is empty (idle state with no active edges), all 12 dimensions return `0.0`.

---

## 📅 Google Calendar Integration (L1 Workload Signal)

`core/context_api.py` → `GoogleContextEngine` authenticates via **OAuth 2.0** and fetches today's Google Calendar
events before the session begins. The meeting count is converted into a `workload_modifier` scalar:

| Meetings Today | `workload_modifier` |
|:---:|:---:|
| 0 | 0.0 |
| 1–2 | 0.1 |
| 3–4 | 0.3 |
| 5+ | 0.5 |

This value is then injected as a `Schedule → User` edge (`calendar_pressure`) directly into the graph.

---

## 🔄 Data Flow Summary

```
Raw Sensors (10-second tick)
        │
        ▼
 BehaviourGraphEngine.update_graph(signals)
        │  ├─ Add each non-zero signal as a directed edge
        │  ├─ Stamp with Unix timestamp
        │  └─ Prune edges older than 300 seconds
        │
        ▼
 BehaviourGraphEngine.get_embedding()
        │  └─ Compute 12 graph-structural features
        │
        ▼
 torch.Tensor  shape=(12,)
        │
        ▼
  Appended to rolling window (deque, maxlen=10)
        │
        ▼
  [ Layer 2 — NCDE ] consumes window as path X(t)
```

---

## 🚫 What Layer 1 Does NOT Use

| Technique | Used? | Reason |
|-----------|-------|--------|
| SVM | ❌ No | No classification step; Layer 1 only builds and reads a graph |
| Logistic Regression | ❌ No | That appears in Layer 4 (SCM) |
| Decision Tree / Random Forest | ❌ No | Not used anywhere |
| Any neural network | ❌ No | Neural model starts at Layer 2 |
| Feature scaling / normalization | ❌ No | Raw weights passed; NCDE handles scale via tanh |

---

## 📦 Key Classes & Methods

| Class / Method | Role |
|----------------|------|
| `BehaviourGraphEngine.__init__()` | Creates the `MultiDiGraph` and initialises all 12 semantic nodes |
| `BehaviourGraphEngine.update_graph(signals)` | Adds edges for non-zero signals; prunes stale edges |
| `BehaviourGraphEngine.get_embedding()` | Computes the 12-dim structural feature vector |
| `SIGNAL_TO_EVENT` (dict) | Maps each of the 20 signal names to `(source_node, target_node, event_type)` |
| `NODE_TYPES` (list) | The 12 semantic node names used to initialise the graph |
