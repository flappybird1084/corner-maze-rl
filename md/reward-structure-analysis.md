# Reward Structure Analysis & Literature-Backed Recommendations

## Current Reward Values

Sourced from `src/corner_maze_rl/env/constants.py` and `_compute_reward` in `src/corner_maze_rl/env/corner_maze_env.py`. Env source is authoritative — if this section drifts from the code again, fix the doc.

| Constant | Value | Role |
|----------|-------|------|
| `STEP_FORWARD_COST` | -0.0005 | Base step cost for forward (action 2), pickup (action 3), and pause (action 4) |
| `STEP_TURN_COST` | -0.001 | Base step cost for left/right turns (actions 0, 1) — 2× forward, discourages spinning |
| `WELL_REWARD_SCR` | +1.061 | Added to the step cost when a PICKUP lands on the correct well |
| *(none)* | 0 | Wrong-well penalty — `_compute_reward` adds no bonus for `"well_empty"`; only the base step cost is charged. The env returns the label as info but the magnitude is implicitly zero. |

There are no `INAPPROPRIATE_ACTION_SCR`, `TIME_OUT_SCR`, `TOO_LONG_IN_PHASE`, `REVISIT_SCR`, `SAME_PLACE_SCR`, or `TOO_LONG_SCR` constants in the env. Earlier versions of this doc referenced names from a now-replaced reward structure; they are kept out of this analysis.

---

## Issues Identified

### 1. Oscillation / pause exploit
The agent can pause (action 4) indefinitely with only the `-0.0005` per-step cost — the same as a forward step. Spinning in place is slightly more expensive at `-0.001` per turn. Neither action carries an explicit revisit penalty, so a "stall until timeout" policy is cheap. PPO / SR policies have shown this exploit in practice; consider adding an explicit pause/revisit penalty or a count-based novelty bonus (see recommendations).

### 2. Turn-to-Forward Ratio (2:1)
Turns cost 2× forward. This is the *intended* design — the asymmetry is there specifically to discourage in-place spinning. Earlier versions of this doc reported 5:1, which was a doc-internal error; the env has always been 2:1 since the current constants landed.

### 3. Extremely sparse reward signal
The agent must navigate a complex maze with barriers, cue associations, and multiple phases, but only receives +1.061 upon reaching the correct well. Everything else is near-zero noise. This is the core motivator for the count-based exploration and PBRS recommendations below.

---

## What the Literature Recommends

### 1. Potential-Based Reward Shaping (PBRS) — Considerations for Animal Behavior Modeling

The gold standard from **Ng, Harada & Russell (1999)**. The shaped reward takes the form:

```
F(s, s') = gamma * Phi(s') - Phi(s)
```

This is **proven** to preserve the optimal policy while accelerating learning. In Sokoban (a grid puzzle), PBRS reduced episodes-to-solve by **4x** and improved solve rate from <20% to 100% on hard tasks.

#### Ecological Validity Concern

**Standard PBRS with precomputed BFS distances is NOT appropriate for animal behavior modeling.** Using `Phi(s) = -shortest_path_distance(s, goal)` injects a priori spatial knowledge — the agent "knows" where the goal is before ever discovering it. Real animals must learn spatial relationships through exploration and experience. This fundamentally violates the ecological validity of the model.

#### Valid Alternative: Bootstrapped Reward Shaping (2025, arXiv:2501.00989)

Uses the agent's **own evolving value function** as the potential: `Phi(s) = V_hat(s)`. The agent starts with no spatial knowledge and the shaping signal improves organically as it learns. This preserves the PBRS policy-invariance guarantee while requiring the agent to earn its own spatial representation — analogous to how animals build cognitive maps through experience.

### 2. Step Penalty Sizing — The -1 Per Step Finding

A surprising **2024 paper** ("Revisiting Sparse Rewards for Goal-Reaching RL", arXiv:2407.00324) found that a simple **minimum-time formulation** (`-1` per step, `0` at goal) **consistently outperformed elaborately shaped dense rewards** — policies reached goals ~3x faster on easy tasks and ~2x faster on hard tasks. The caveat: the agent must reach the goal at least **~10 times per 20k timesteps** from random exploration.

**Rule of thumb from Sutton & Barto:** `|step_penalty × optimal_path_length| < 0.5 × goal_reward`. Your current values satisfy this, but are so small they provide almost no gradient signal.

### 3. Exploration Bonuses — Medium-High Impact (Ecologically Valid)

These approaches are well-suited for animal behavior modeling because they inject no goal-location knowledge. Instead, they model intrinsic motivation — a well-documented phenomenon in animal cognition where organisms are driven to explore novel stimuli and environments.

- **Count-based (MiniGrid `PositionBonus`):** Bonus of `1/sqrt(N(s))` for visiting state `s`. First visit gets +1.0, second ~0.707, etc. This is a **free win** since MiniGrid already provides this wrapper. Models the novelty-seeking behavior observed in rodents.
- **Random Network Distillation (RND, Burda et al. 2018):** Prediction error from a fixed random network as intrinsic reward. Achieved 50%+ room exploration in Montezuma's Revenge. Analogous to prediction-error-driven exploration in animal learning theory.
- **ICM (Pathak et al. 2017):** Forward prediction error in a learned feature space as curiosity signal. Models the forward-model-based exploration hypothesized in animal spatial navigation.

### 4. Hindsight Experience Replay (HER) — For Off-Policy Algorithms

From **Andrychowicz et al. (2017)**: relabel failed trajectories with the actually-achieved state as the "goal." Uses `k=4` future goals per real transition. Only works with off-policy methods (DQN, SAC, TD3). Would require restructuring to Gymnasium's `GoalEnv` interface.

### 5. Curriculum Learning — Medium Impact

A **2025 paper** (arXiv:2501.17842) inspired by Tolman's rat maze experiments showed that **sparse-to-dense reward transitions** outperformed both pure-sparse and pure-dense. Train first with sparse rewards (builds "cognitive maps"), then transition to dense rewards (guides exploitation). Your session phases (exposure → acquisition) already hint at this structure — lean into it.

### 6. Reward Normalization — High Impact, Easy Win

SB3 documentation explicitly recommends `VecNormalize` for PPO/A2C. Large reward scale differences cause training instability. Wrapping your env with reward normalization or clipping to [-1, 1] is critical.

---

## Concrete Recommendations

| Change | Priority | Rationale |
|--------|----------|-----------|
| Add count-based exploration bonus (`PositionBonus`) | **High** | Ecologically valid intrinsic motivation; no goal knowledge leaked |
| Use `VecNormalize` or reward clipping | **High** | Prevents gradient explosion from scale mismatch |
| Add explicit pause/revisit penalty (~`-0.002`) | **Medium** | Prevents the pause-until-timeout exploit (Issue 1) |
| Consider Bootstrapped Reward Shaping | **Medium** | PBRS using agent's own learned V_hat — no a priori knowledge |
| Consider RND or ICM curiosity | **Medium** | Stronger exploration signal; models novelty-seeking behavior |
| Consider minimum-time formulation (-1/step) | **Low** | Only if agent finds goal ≥10 times per 20k steps |
| Implement HER with GoalEnv | **Low** | High effort, only for off-policy algorithms |
| ~~Add PBRS with BFS-distance potential~~ | ~~N/A~~ | ~~Rejected: injects a priori goal knowledge, invalid for animal modeling~~ |

---

## Key References

- Ng, Harada & Russell (1999) — Potential-based reward shaping policy invariance theorem
- Andrychowicz et al. (2017) — Hindsight Experience Replay
- Pathak et al. (2017) — Intrinsic Curiosity Module (ICM)
- Burda et al. (2018) — Random Network Distillation (RND)
- "Revisiting Sparse Rewards for Goal-Reaching RL" (2024) — arXiv:2407.00324
- "Bootstrapped Reward Shaping" (2025) — arXiv:2501.00989
- "From Sparse to Dense: Toddler-inspired Reward Transition" (2025) — arXiv:2501.17842
- Sutton & Barto — Reinforcement Learning: An Introduction
- Stable Baselines3 Tips & Tricks — https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
