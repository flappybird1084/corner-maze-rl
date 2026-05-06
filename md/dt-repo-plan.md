# Corner-Maze-RL Repo Plan

**Status:** draft, pre-build
**Last updated:** 2026-05-05 (ported into new `corner-maze-rl` repo)
**Purpose:** standalone repo combining the yoked-data pipeline, the maze environment, and a Decision Transformer + plug-in PPO/SR baselines, for student use via Colab and VS Code.

**Companion docs in this repo:**
- [environment-architecture.md](environment-architecture.md) — env spec the new code is built against
- [maze-behavior-spec.md](maze-behavior-spec.md) — 2S2C task rules
- [reward-structure-analysis.md](reward-structure-analysis.md) — reward shaping rationale
- [sr-yoked-negative-results.md](sr-yoked-negative-results.md) — known SR failure case

**Legacy repo (source of code/data to port):** `https://github.com/ryangrg/corner-maze-rl-legacy`. See §16 for the file-by-file source manifest.

---

## 1. Goals

1. Single-file-style implementations of **DT**, **PPO**, **SR** that share an env, an encoder layer, and a run-management layer.
2. Decision Transformer trained on yoked rat behavior with **real env-derived rewards** (not the fake "session-progress" RTG in the current notebook).
3. Pluggable **state encoders** (grid-cells, visual-CNN, one-hot tabular, reward-history, egocentric image) that can be combined.
4. **Pip-installable from GitHub** so Colab notebooks are thin (`pip install git+https://...`; no sys.path hacks).
5. Reproducible run-saving with seeds, configs, and per-step trajectories.
6. **Compute-aware kill switch** — terminate a run early if the learning curve is provably flat, but let "creeping" runs continue.
7. Student-friendly: clone-and-run locally in VS Code, *or* Colab-only, with the same code.
8. Evaluation protocol grounded in the empirical-RL methodology canon (IQM, stratified bootstrap, drawdown — see [LLM Wiki references](#10-evaluation-protocol)).

## 2. Non-goals

- Changing maze behavior design (sparse +1/trial scoring is fixed).
- Building a new yoking pipeline (reuse the mature one in [yoking/](../yoking/)).
- Multi-rat training (one rat per experiment, ≤ 80 sessions).
- Retraining the visual CNN by default (ship pretrained; allow on-the-fly retrain as advanced option).

## 3. Repo layout

```
corner-maze-dt/
├── README.md                            # students: install + 5-min quickstart
├── pyproject.toml                       # pip-installable package
├── colab/
│   ├── 01_explore_env.ipynb             # env walkthrough, manual control
│   ├── 02_explore_yoked_data.ipynb      # load dataset, plot trajectories
│   ├── 03_compute_returns.ipynb         # build reward + RTG cache
│   ├── 04_train_dt.ipynb                # DT train + eval
│   ├── 05_train_ppo.ipynb               # PPO baseline
│   ├── 06_train_sr.ipynb                # SR baseline
│   └── 07_compare_models.ipynb          # IQM, performance profiles, drawdown
├── src/corner_maze_dt/
│   ├── env/                             # trimmed env + constants + trial_seq
│   ├── data/
│   │   ├── load.py                      # DuckDB queries against 3-table dataset
│   │   ├── compute_returns.py           # env-replay → reward, RTG (per-trial, ITI-start)
│   │   ├── windows.py                   # context-window builder for DT
│   │   └── session_types.py             # 4 hardcoded paradigms; PI / VC / PI+VC / PI+VC_f1
│   ├── encoders/
│   │   ├── base.py                      # StateEncoder Protocol, CompositeEncoder
│   │   ├── grid_cells.py                # ported from DatasetBuilder.ipynb cell 1 (deterministic)
│   │   ├── visual_cnn.py                # pretrained 60D CNN; load + optional retrain
│   │   ├── one_hot_pose.py              # 196-dim baseline
│   │   ├── reward_history.py            # 4×n_wm decaying timers (from custom_rl.py)
│   │   ├── egocentric_image.py          # raw 21×21×3 MiniGrid view (for CnnPolicy)
│   │   └── registry.py                  # build_encoder(["grid_cells", "visual_cnn"]) → composite
│   ├── models/
│   │   ├── base.py                      # TrainableAgent Protocol
│   │   ├── decision_transformer.py
│   │   ├── ppo.py                       # adapted from custom_rl.py PPOAgent
│   │   └── sr.py                        # adapted from custom_rl.py SRAgent
│   ├── train/
│   │   ├── runner.py                    # adapted from session_runner.py
│   │   ├── kill_switch.py               # learning-curve early termination
│   │   └── callbacks.py                 # logging, checkpointing
│   ├── eval/
│   │   ├── rollout.py                   # online rollout in env
│   │   ├── metrics.py                   # IQM, drawdown, success rate
│   │   ├── viz.py                       # learning curves, performance profiles
│   │   └── replay.py                    # play back yoked or model trajectories (no video)
│   └── utils/
│       ├── run_io.py                    # set_global_seed, save_run_config (port from seed_utils.py)
│       └── git.py                       # capture commit SHA in run_config
├── scripts/
│   ├── build_returns_dataset.py         # one-time: actions_with_returns.parquet
│   └── train.py                         # CLI: --model {dt,ppo,sr} --session-type ... --seeds N
├── data/                                # gitignored; setup script downloads
└── tests/
    ├── test_env_replay_determinism.py
    ├── test_returns_correctness.py
    ├── test_encoders_dim.py
    └── test_kill_switch.py
```

## 4. Reward & return-to-go (the core integration)

### 4.1 Behavior recap (do not change)
- Every trial ends in either reward or stuck (no time-out kill of trials).
- Wrong well **does not** end a trial; only correct well does.
- Per-trial score: +1 if rat completed the trial without ever visiting a non-rewarded well, else 0. Max 32/session.
- **The agent does not see this score.** The agent sees only the env's `step()` reward, which is currently:
  - `STEP_FORWARD_COST = -0.0005`
  - `STEP_TURN_COST = -0.001`
  - `WELL_REWARD_SCR = 1.061` on correct well
  - `-0.005` on empty/wrong well
- Score is for human evaluation, not policy learning.

### 4.2 Preprocessing pass (`compute_returns.py`)
One-time, deterministic, replays each yoked session through `CornerMazeEnv` and adds two columns to a new `actions_with_returns.parquet`:

| Column | Description |
|--------|-------------|
| `reward` | Per-step scalar from `env.step()` (uses constants above). |
| `return_to_go` | Sum of rewards from this step to **end of current trial** (ITI-start mode — see below). |

**Per-trial RTG with ITI-start.** RTG is computed within the boundaries of `[ITI_start_t, trial_end_t]`. Reasoning:
- Per-session RTG drowns goal credit in 32 trials of step costs.
- Per-trial RTG resets at trial start, but agents trained that way sometimes get stuck in ITI (no reward credit available there).
- ITI-start: include the ITI before the trial in the same RTG window. ITI traversal earns credit toward the upcoming trial reward → encourages forward movement through ITI.

Boundary detection: the env emits trial-phase tags via `_compute_session_scores()` and trial_tags. Bake those into the preprocessing pass.

### 4.3 Sparse-reward acknowledgement
RTG signal is dominated by the +1.061 spike at trial end. Successful trials produce strongly positive RTG curves; failed trials produce flat slightly-negative RTG. DT learns to discriminate these regimes but credit assignment within a successful trial is coarse. We document this honestly and treat it as a teaching opportunity ("this is *why* DT papers use dense rewards").

### 4.4 Inference-time RTG (the part students find counterintuitive)
At rollout, the student specifies a *target* return; RTG decrements as rewards are observed:

```python
target_return = 5.0  # student-set
for t in range(max_steps):
    context.append((target_return, encode(obs), prev_action))
    action = dt(context[-K:]).argmax()
    obs, reward, done, _ = env.step(action)
    target_return -= reward
    if done: break
```

The eval notebook ships a slider over target_return so students can see the optimism→imitation knob in action.

## 5. Encoders (the composability layer)

### 5.1 Protocol
```python
class StateEncoder(Protocol):
    output_dim: int
    def encode(self, layout: str, x: int, y: int, direction: int) -> np.ndarray: ...
```

### 5.2 Provided encoders (each defaults to 60D when applicable)

| Name | Dim | Source | Trainable? | Notes |
|------|-----|--------|-----------|-------|
| `grid_cells` | 60 | port of [notebooks/DatasetBuilder.ipynb](../notebooks/DatasetBuilder.ipynb) cell 1 | No (fixed neural code) | 5 modules × 12 phase bins, von Mises directional tuning. |
| `visual_cnn` | 60 | port of [notebooks/Minigrid_Maze_Yoked_Training.ipynb](../notebooks/Minigrid_Maze_Yoked_Training.ipynb) cells 11–12 | Pretrained ships; optional retrain | Local-view PNGs → 60D latent. |
| `one_hot_pose` | 196 | new | No | 196-tile one-hot (matches existing tabular setup). |
| `reward_history` | 4·n_wm | port from [custom_rl.py](../src/custom_rl.py) `generate_state_vector()` | No | Decaying timers per well; injects partial observability. |
| `egocentric_image` | 21×21×3 | env's native obs | No | For CnnPolicy variants. |

### 5.3 Composition
```python
encoder = build_encoder(["grid_cells", "visual_cnn"])   # → 120-dim composite
encoder = build_encoder(["one_hot_pose", "reward_history"])   # → 236-dim, matches your tabular runs
```

The DT's state-token Linear is sized to `encoder.output_dim`. PPO/SR consume the same vector. **One encoder definition drives all three model types**, which is the clean ablation surface students need.

### 5.4 Tabular vs. egocentric switch
A first-class config flag:
```yaml
observation_mode: "vector"    # uses encoder registry, runs MLP/transformer
# or
observation_mode: "image"     # raw 21×21×3, runs CnnPolicy
```
The env already supports both modes via [src/sb3_agents.py](../src/sb3_agents.py)'s wrappers. We expose them through a single switch.

## 6. Models

### 6.1 Plug-in protocol
```python
class TrainableAgent(Protocol):
    def train_one_session(self, env, callbacks=None) -> SessionResult: ...
    def act(self, obs) -> int: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "TrainableAgent": ...
```

`md-files/adding_a_model.md` will document the contract. The env never changes.

### 6.2 Decision Transformer (`models/decision_transformer.py`)
Adapted from [DTtrainer.ipynb](../2S2C_task/colab/DTtrainer.ipynb) with these changes:
- Real RTG from preprocessing pass (not the fake `1 - steps_to_go/500`).
- **Drop** the `ACTION_WEIGHTS = [1,1,1,20,1]` hack.
- **Configurable positional encoding** (`pos_encoding: 'learned' | 'sinusoidal' | 'spatial' | 'none'`):
  - `learned` (default) — learned timestep embedding per token position.
  - `sinusoidal` — Vaswani-style fixed sinusoidal timestep encoding.
  - `spatial` — the notebook's quirky "add pose-vector to R/S/A". Kept for ablation parity with original implementation.
  - `none` — no positional info (instructive baseline; shows why position matters).
  - All four available out of the box; students compare them as a built-in ablation.
- 5-action space already aligned with `actions_synthetic_pretrial.parquet`; no remap.
- Variable context size, default 64.

### 6.3 PPO (`models/ppo.py`)
Adapted from [src/custom_rl.py](../src/custom_rl.py) `PPOAgent`. Generic over `encoder.output_dim`. Add SB3 path for students who want CnnPolicy/MaskablePPO.

### 6.4 SR (`models/sr.py`)
Adapted from [src/custom_rl.py](../src/custom_rl.py) `SRAgent`. Document the [SR yoked negative results](../md-files/sr-yoked-negative-results.md) up front so students don't expect it to work — it's an instructive failure case (see "negative results" framing below).

## 7. Run management & saving

### 7.1 Per-run output directory
```
runs/{model}_{session_type}_{timestamp}/run_{i}/
  run_config.json        # seed, hyperparams, encoder list, session_type, git SHA, dataset hash
  model.pt               # checkpoint
  trajectory.parquet     # per-step: layout, x, y, dir, action, reward, RTG, trial_idx, trial_tag, done
  scores.parquet         # per-trial + per-session aggregates
  metrics.json           # eval results
  curves.parquet         # rolling per-session score for kill-switch + plotting
  killed_at.json         # if early-terminated, why and when
```

### 7.2 What to capture per session (canonical schema)

**Per-step:** `layout_name, x, y, direction, action, reward, return_to_go, done, trial_index, trial_phase, trial_tag, model_logits` (optional — useful for DT/PPO inspection).

**Per-trial:** `trial_index, trial_config (start_arm, cue, route, goal, tag), success, steps_to_well, error_well_visits, turn_score, target_rtg` (DT-only).

**Per-session:** `total_return, perfect_trial_count, success_rate_by_tag, mean_steps_per_trial, training_criterion_met, killed_early`.

**Run metadata:** `seed, git_sha, hyperparams, encoder_list, session_type, model_type, wall_clock_seconds, dataset_hash`.

### 7.3 Student workflow
**Recommended:** clone repo locally → `pip install -e .` → open in VS Code → connect Colab to the local folder for GPU bursts. Runs save to local `runs/` which survives Colab disconnects.

**Colab-only fallback:** `pip install git+https://github.com/<user>/corner-maze-dt.git` → mount Drive → set `RUNS_DIR=/content/drive/MyDrive/.../runs`. Same code, just a different output path.

Both paths documented in README; "VS Code + Colab kernel" is the recommended primary route.

## 8. Kill switch (the compute-aware early stop)

### 8.1 Requirements (from user)
- Kill if learning is **flat**.
- Want learning visible by ~25 sessions.
- **Don't kill if curve is creeping up**, even slowly.
- Max 80 sessions.

### 8.2 Detection logic (`train/kill_switch.py`)
Track per-session `perfect_trial_count` (0–32). After each session, evaluate:

```python
# Inputs: scores = [s_0, s_1, ..., s_n]   (per-session perfect-trial count)

WARMUP = 10                  # never kill before this
SLOPE_WINDOW = 10            # regression window for "creeping" detection
FLAT_SLOPE_EPS = 0.05        # trials/session — below this is "flat"
ABSOLUTE_FLOOR = 4           # if mean of last window < this, eligible to kill
DEAD_WINDOW = 8              # if no successful trial in last K sessions, kill
HARD_CAP = 80                # never run past this

if n < WARMUP:
    return CONTINUE

if all(s == 0 for s in scores[-DEAD_WINDOW:]):
    return KILL("dead — no successful trial in {DEAD_WINDOW} sessions")

window = scores[-SLOPE_WINDOW:]
slope = linear_regression_slope(range(len(window)), window)
recent_mean = np.mean(window)

if slope < FLAT_SLOPE_EPS and recent_mean < ABSOLUTE_FLOOR and n >= WARMUP + SLOPE_WINDOW:
    return KILL(f"flat: slope={slope:.3f}, mean={recent_mean:.1f} after {n} sessions")

if recent_mean >= 24:        # high success — let early-stop on positive criterion handle
    return CRITERION_MET

if n >= HARD_CAP:
    return KILL("hard cap reached")

return CONTINUE
```

### 8.3 Why these defaults
- **Linear regression on a window** is the cheapest way to test "creeping up vs. flat" while ignoring noise. A purely-flat curve has slope ≈ 0; a curve creeping from 2 → 6 over 10 sessions has slope ≈ 0.4.
- **WARMUP = 10** because variance over the first few sessions is high; killing at session 5 risks killing a slow starter.
- **ABSOLUTE_FLOOR = 4** prevents killing a model that's stuck low but technically slope-positive due to noise. Tunable.
- **DEAD_WINDOW** catches catastrophic failure (no perfect trial at all in 8 sessions).
- This logic is the *across-time* analogue of [Chan 2020]'s drawdown / detrended-dispersion metric — applied as a stopping rule, not as a metric.

### 8.4 Config exposure
All thresholds live in `train/kill_switch.py` as module constants and can be overridden per-run via `run_config.kill_switch_overrides`. Defaults are documented in the README so students understand what's terminating their runs.

### 8.5 What gets saved on kill
`killed_at.json` with: `{ "session": n, "reason": ..., "scores_at_kill": [...], "slope": ..., "recent_mean": ... }`. Not a failure — a logged decision.

## 9. Session types (4 hardcoded options)

User picks one as a CLI/config arg. Defaults to PI+VC. Each choice = a *training-group filter* + an *ordered sequence of experimental sessions* (acquisition → probes → reversal → etc.).

### 9.1 Mapping: (training_group, yoked_session_type) → env_paradigm

The yoked dataset's `session_type` column holds experiment-design names; the same yoked session_type maps to a *different env paradigm* depending on the rat's training_group:

| Yoked `session_type` | training_group=PI+VC | training_group=PI | training_group=VC |
|---------------------|----------------------|-------------------|-------------------|
| `Rotate Train` | `PI+VC f2 rotate` | — | `VC acquisition` |
| `Fixed Cue 1` | `PI+VC f2 novel route` | `PI novel route cue` | `VC novel route fixed` |
| `Dark Train` | `PI+VC f2 no cue` | `PI acquisition` | — (TBD) |
| `Fixed Cue 1 Twist` | TBD | TBD | TBD |

**Open data points:** `Fixed Cue 1 Twist` mapping not yet provided; VC × Dark Train cell empty. Resolve before locking; treat unknowns as "skip this session in this group's sequence" with a warning.

### 9.2 Student-facing choices

| Choice | training_group filter | Yoked data status | Notes |
|--------|------------------------|--------------------|-------|
| `pi_vc` (default) | `PI+VC` | ✅ available | Most-tested data path. |
| `pi` | `PI` | ✅ available | |
| `vc` | `VC` | ✅ available | |
| `pi_vc_f1` | `PI+VC` (f1 subjects: CM057, CM058, CM059, CM060, CM061, CM063, CM064) | ⚠️ **not yet yoked** — different correct routes; pipeline extension required | Stub the option, raise `NotImplementedError` with clear message until data is added. |

### 9.3 `data/session_types.py` structure

```python
# (env_paradigm, yoked_session_type) ordered tuples per group
SESSION_SEQUENCES: dict[str, list[tuple[str, str]]] = {
    "pi_vc": [
        ("PI+VC f2 rotate",      "Rotate Train"),
        ("PI+VC f2 novel route", "Fixed Cue 1"),
        ("PI+VC f2 no cue",      "Dark Train"),
        # ... extend as paradigm sequence is finalized
    ],
    "pi": [
        ("PI acquisition",       "Dark Train"),
        ("PI novel route cue",   "Fixed Cue 1"),
        # ...
    ],
    "vc": [
        ("VC acquisition",       "Rotate Train"),
        ("VC novel route fixed", "Fixed Cue 1"),
        # ...
    ],
    "pi_vc_f1": None,   # raises NotImplementedError until yoking extended
}

# subjects.parquet filter for the f1 subgroup, when added:
F1_SUBJECT_NAMES = {"CM057", "CM058", "CM059", "CM060", "CM061", "CM063", "CM064"}
```

The runner iterates through this sequence per chosen group, pulling the matching yoked rows for each (env_paradigm, yoked_session_type) pair.

## 10. Evaluation protocol

Grounded in the empirical-RL methodology canon ([LLM Wiki - AI Reproducibility](file:///Users/ryangrgurich/Code/llm-wiki/vaults/ai-reproducibility/Wiki/index.md)).

### 10.1 What we report per (model, session_type, encoder_config) cell

**Performance (across runs):**
- **IQM** of final per-session perfect-trial count (last 5 sessions averaged) as primary metric. Justification: [agarwal-2021]. Robust to outlier seeds.
- **Median** and **mean** as supplementary.
- **Stratified bootstrap 95% CIs** via Rliable (`pip install rliable`).
- **Performance profile**: empirical CDF of final scores across seeds. Reveals algorithm crossings ([agarwal-2021], [jordan-2020]).

**Reliability (within model):**
- **DR** — IQR across seeds at the final checkpoint ([chan-2020]).
- **LRT** — CVaR(α=0.05) of drawdown from running peak. Catches catastrophic mid-training drops.
- **DF** — variability of evaluation rollouts of the trained policy (run 30 deterministic rollouts of saved checkpoint, report IQR of total return).

**Pairwise comparison (model A vs B):**
- Paired bootstrap on IQM-difference. Bonferroni correction when comparing > 2 models.
- For very small N (< 20), use Welch's t-test instead — bootstrap CIs under-cover at small N ([colas-2018]).

### 10.2 Seed counts and run profiles

`scripts/train.py --profile {pilot,compare,headline}` selects a coordinated bundle of seed count + kill-switch config + reporting expectations. Defaults documented in README.

| Profile | N seeds | Kill switch | Use case |
|---------|---------|-------------|----------|
| `pilot` (default) | 5 | aggressive (default thresholds) | initial design iteration, hyperparameter sanity checks; explicitly underpowered |
| `compare` | 10–20 | default | pairwise A/B between models or encoder configs |
| `headline` | 30 | relaxed thresholds (longer WARMUP, lower DEAD_WINDOW sensitivity) | published-quality runs; final negative-results claims (Wilson 95% upper bound ≤ 0.10) |

Individual flags (`--seeds N`, `--kill-switch off`, `--warmup 15`) override profile defaults. `--profile` sets the *defaults*, never silently overrides explicit flags.

Justification: [colas-2018] (N≥10 for power), [agarwal-2021] (N≥30 for headline IQM), [patterson-2024] (N=50+ when feasible).

The kill switch *adds* statistical power per-seed (more compute reaches the seeds that learn), but doesn't substitute for seed count. Document this trade-off in `eval/metrics.md`.

### 10.3 Negative results as first-class output
Three of our model × encoder × paradigm cells will likely fail (SR on yoked already known to fail — see [sr-yoked-negative-results.md](sr-yoked-negative-results.md)). Frame these as scientific findings, not bugs. Per [patterson-2024]:

> "Under [observation modality] and the tuning ranges in Table N, [model] did not reach criterion on [session_type] within [budget] sessions (0/N runs; Wilson 95% upper bound: 3/N). The same model reliably reached criterion on [adjacent paradigm] (M/N runs)."

Bake this template into the comparison notebook (07).

### 10.4 Two-stage tuning
Avoid maximization bias:
- **Stage 1** — separate seed pool for hyperparameter selection (5–10 seeds per config).
- **Stage 2** — fresh seeds for the reported numbers (10–30 seeds, never overlapping Stage 1).
- Document which seed pool any given metric was computed on.

`scripts/train.py` accepts `--stage {tune,report}` to make this explicit.

### 10.5 What we deliberately don't do
- Don't report max-during-training (biased upward, protocol-incompatible across runs).
- Don't tune-and-report on the same seeds.
- Don't use bare bootstrap CIs at N < 20.
- Don't claim "model A is better" without paired statistics.

## 11. Data pipeline

The yoking pipeline stays in `corner-maze-rl-legacy/yoking/`. This repo *consumes* the resulting 3-table dataset (copied per §16.2) and adds one downstream stage:

```
data/yoked/dataset/                                  (copied from legacy, read-only)
  ├── subjects.parquet
  ├── sessions.parquet
  └── actions_synthetic_pretrial.parquet
                ↓ scripts/build_returns_dataset.py    (this repo)
data/yoked/dataset/actions_with_returns.parquet      (new — adds reward + RTG columns)
```

The new repo depends on `actions_with_returns.parquet`. The build script is one-time, deterministic, hashable — the dataset hash goes into every run_config.json so we know which dataset version a run was trained on.

When the f1 subjects (CM057–064, see §9.2) need to be yoked, that work happens in the legacy repo's `yoking/` pipeline; the resulting parquets are then copied into this repo.

## 12. Environment integration

The env is ported from `corner-maze-rl-legacy/src/env/corner_maze_env.py` (see manifest §16.1) — trimmed for student readability but **no behavioral changes**.

The model-pluggability contract: a new model only needs to implement `TrainableAgent` and consume `obs` + return `action`. The env's obs space exposes:
- `view` (21×21×3 image) — for CnnPolicy.
- `embedding` (vector from registry) — for MLP/transformer.
- `stereo` (2-channel image) — for the StereoFeaturesExtractor pipeline.

A new model may add a *new* obs key (env-side change). The pipeline above the env (encoder, runner, eval) is unchanged.

## 13. Open questions / pre-build tasks

In priority order:

1. **Repo name** — public GitHub, need a name before `pyproject.toml`. See suggestions in conversation.

2. **Complete session_type mapping for two cells** — `Fixed Cue 1 Twist` (all groups) and `Dark Train × VC` are still TBD. Quick query against the analysis-side data; not a blocker for first-pass build (we can ship with `pi_vc` only).

3. **Decide which env paradigms make up each group's full sequence** — the user gave the *mapping* but not the *order* of sessions within each group's training arc (e.g., does PI+VC always run rotate → novel → no_cue, or different?). Required for the runner to advance through sessions correctly.

### Closed (this iteration)

- ~~PI+VC_f1 data~~ — separate subjects (CM057–064), different correct routes, **not yet yoked**. Stub the option, raise `NotImplementedError`. Yoking-pipeline extension scheduled for later.
- ~~Positional encoding choice~~ — keep flexible: 4 styles (`learned`, `sinusoidal`, `spatial`, `none`) ship out of the box. Default `learned`.
- ~~Repo hosting~~ — public GitHub confirmed.
- ~~Seed count for headline runs~~ — N=30 for `headline` profile, N=5 for `pilot` (the default), N=10–20 for `compare`. Implemented as `--profile` flag.

## 14. Phased build order

**Phase 0 — pre-build investigation (1 session)**
- Resolve open questions 1, 2, 3 above. Update this plan.

**Phase 1 — foundation (independent, parallelizable)**
- Port `encoders/grid_cells.py` from `DatasetBuilder.ipynb` cell 1.
- Write `data/compute_returns.py` + `scripts/build_returns_dataset.py`.
- Write `train/kill_switch.py` + tests.
- Port `utils/run_io.py` from `seed_utils.py`.

**Phase 2 — first model (DT)**
- `models/decision_transformer.py` (clean rewrite from `DTtrainer.ipynb`).
- `train/runner.py` (adapt from `session_runner.py`).
- `eval/rollout.py`, `eval/replay.py`.
- Notebook 04 (DT train) + Notebook 03 (compute returns).

**Phase 3 — baselines**
- `models/ppo.py`, `models/sr.py` (adapt from `custom_rl.py`).
- Notebooks 05, 06.

**Phase 4 — eval & comparison**
- `eval/metrics.py` (IQM, drawdown via Rliable).
- Notebook 07 (compare models).
- README + `md-files/adding_a_model.md`.

**Phase 5 — student polish**
- `colab/` notebooks finalized.
- Quickstart README.
- Setup script for `data/`.

## 15. Dependencies

```
gymnasium, minigrid, numpy, pandas, pyarrow, duckdb, torch
stable-baselines3, sb3-contrib    # optional, for SB3 PPO path
rliable                            # IQM, stratified bootstrap
matplotlib, ipywidgets             # notebook viz
pygame                             # local replay (no moviepy)
tqdm
```

No moviepy (video pipeline replaced by pygame replay per §1).

---

## Decisions log (incorporate as user confirms)

- ✅ Real env-derived rewards, not steps_to_go proxy.
- ✅ Per-trial RTG with ITI-start window.
- ✅ No subject conditioning (one rat per experiment).
- ✅ Drop `ACTION_WEIGHTS` hack.
- ✅ Keep grid cell embeddings (port from `DatasetBuilder.ipynb`).
- ✅ Pretrained visual CNN ships; on-the-fly retrain optional.
- ✅ All encoders standardize to 60D (grid cell constraint).
- ✅ 4 hardcoded session types; PI+VC default.
- ✅ Composable encoders + tabular/egocentric switch.
- ✅ Pip-installable for Colab.
- ✅ Run saving with seeds + git SHA, mirroring current `run_config.json` pattern.
- ✅ DT, PPO, SR ship with the repo; new models plug in via `TrainableAgent` protocol.
- ✅ Env unchanged unless model needs new obs key.
- ✅ Kill switch with slope detection, WARMUP, DEAD_WINDOW, HARD_CAP.
- ✅ Eval protocol: IQM + bootstrap CIs + performance profiles + DR/LRT reliability + Wilson rule for negatives.
- ✅ Public GitHub for pip install.
- ✅ Seed counts: pilot=5 / compare=10–20 / headline=30, exposed via `--profile` flag.
- ✅ DT positional encoding kept flexible: 4 styles ship (`learned`, `sinusoidal`, `spatial`, `none`); default `learned`.
- ✅ Session-type mapping resolved (mostly) — see §9.1 table. PI+VC_f1 data deferred (not yet yoked).
- ✅ Repo name: `corner-maze-rl` (this repo); legacy archived as `corner-maze-rl-legacy`.
- ⏳ Two empty cells in mapping table: `Fixed Cue 1 Twist`, `Dark Train × VC`.
- ⏳ Per-group session sequence ordering (for runner traversal logic).

---

## 16. Legacy source manifest

Files to copy or port from `corner-maze-rl-legacy` into this repo as the build proceeds. Phase column maps to §14. "Action" is `copy` (verbatim, read-only artifact) or `port` (rewrite + adapt; legacy file is the reference).

### 16.1 Environment

| Phase | Action | Source (legacy) | Target (new) | Notes |
|-------|--------|-----------------|--------------|-------|
| P1 | port | `src/env/corner_maze_env.py` | `src/corner_maze_dt/env/corner_maze_env.py` | Trim for student readability; **no behavioral changes**. |
| P1 | port | `src/env/constants.py` | `src/corner_maze_dt/env/constants.py` | |
| P1 | port | `src/env/trial_sequence_gen.py` | `src/corner_maze_dt/env/trial_sequence_gen.py` | |
| P1 | port | `src/env/trial_sequence_validation.py` | `src/corner_maze_dt/env/trial_sequence_validation.py` | |

### 16.2 Yoked dataset (data only — pipeline stays in legacy)

| Phase | Action | Source (legacy) | Target (new) | Notes |
|-------|--------|-----------------|--------------|-------|
| P1 | copy | `data/yoked/dataset/subjects.parquet` | `data/yoked/dataset/subjects.parquet` | Read-only consumed artifact. |
| P1 | copy | `data/yoked/dataset/sessions.parquet` | `data/yoked/dataset/sessions.parquet` | |
| P1 | copy | `data/yoked/dataset/actions_synthetic_pretrial.parquet` | `data/yoked/dataset/actions_synthetic_pretrial.parquet` | Input to `compute_returns.py`. |
| — | reference | `md-files/yoked-data-guide.md` | (linked, not copied) | Schema canonical doc; cite via legacy GitHub URL. |
| — | reference | `yoking/` | (entire dir stays in legacy) | Only relevant if extending to f1 subjects later. |

### 16.3 Encoders

| Phase | Action | Source (legacy) | Target (new) | Notes |
|-------|--------|-----------------|--------------|-------|
| P1 | port | `notebooks/DatasetBuilder.ipynb` cell 1 | `src/corner_maze_dt/encoders/grid_cells.py` | Grid-cell pose-vector generator. Deterministic, no training. |
| P2 | copy | `2S2C_task/embeddings/60d/position/pose_60Dvector_dictionary.pkl` | `data/encoders/pose_60Dvector_dictionary.pkl` | Pre-built grid-cell dict; ships for fast student start. |
| P2 | copy | `2S2C_task/embeddings/60d/image/ryans_visual_embedding_dictionary.pkl` | `data/encoders/ryans_visual_embedding_dictionary.pkl` | Pretrained visual CNN dict. |
| P2 | port | `notebooks/Minigrid_Maze_Yoked_Training.ipynb` cells 11–12 | `src/corner_maze_dt/encoders/visual_cnn.py` | CNN train + extract; pretrained weights ship. |
| P1 | port | `src/custom_rl.py::generate_state_vector*` | `src/corner_maze_dt/encoders/{one_hot_pose,reward_history}.py` | Two functions split into separate encoder modules. |

### 16.4 Models

| Phase | Action | Source (legacy) | Target (new) | Notes |
|-------|--------|-----------------|--------------|-------|
| P2 | port | `2S2C_task/colab/DTtrainer.ipynb` cells 1–2 | `src/corner_maze_dt/models/decision_transformer.py` + `data/windows.py` | DT model + dataloader template; rewrite per dt-repo-plan §6.2 (real RTG, configurable pos encoding, no action-weights hack). |
| P3 | port | `src/custom_rl.py::PPOAgent` | `src/corner_maze_dt/models/ppo.py` | |
| P3 | port | `src/custom_rl.py::SRAgent` | `src/corner_maze_dt/models/sr.py` | |
| P3 | optional | `src/sb3_agents.py` | `src/corner_maze_dt/models/sb3_ppo.py` | Optional SB3 path for advanced students. |

### 16.5 Training & utilities

| Phase | Action | Source (legacy) | Target (new) | Notes |
|-------|--------|-----------------|--------------|-------|
| P1 | port | `src/seed_utils.py` | `src/corner_maze_dt/utils/run_io.py` | Add git-SHA capture + dataset hash. |
| P2 | port | `src/session_runner.py` | `src/corner_maze_dt/train/runner.py` | |
| P2 | port | `scripts/manual_control.py` | `src/corner_maze_dt/eval/replay.py` | Pygame-based replay; no moviepy. |

### 16.6 Reference / documentation

| Action | Source (legacy) | New repo location | Notes |
|--------|-----------------|-------------------|-------|
| copied | `md-files/maze-behavior-spec.md` | `md/maze-behavior-spec.md` | ✅ done, this commit. |
| copied | `md-files/reward-structure-analysis.md` | `md/reward-structure-analysis.md` | ✅ done. |
| copied | `md-files/sr-yoked-negative-results.md` | `md/sr-yoked-negative-results.md` | ✅ done. |
| extracted | legacy `MEMORY.md` Environment Architecture section | `md/environment-architecture.md` | ✅ done. |

### 16.7 Things explicitly NOT to copy

- `src/sb3_agents.py::StereoFeaturesExtractor` and stereo-eye dataset — out of scope for DT/PPO/SR student build.
- `md-files/yoked-env-integration-plan.md`, `md-files/yoking-process-log.md`, `md-files/well-exit-action-masking-plan.md` — historical legacy planning artifacts.
- `experiments/`, `model-data/` from legacy — those are legacy run outputs.
- Notebooks `corner-maze-rl.ipynb`, `colab-minigrid-rl.ipynb` — legacy scratchpads; new repo will have its own clean Colab notebooks per §3.
