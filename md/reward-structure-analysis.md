# Reward Structure Analysis & Literature-Backed Recommendations

## Current Reward Values

| Constant | Value | Role |
|----------|-------|------|
| `FORWARD_SCR` | -0.001 | Step cost (forward) |
| `TURN_SCR` | -0.005 | Step cost (turn) |
| `WELL_REWARD_SCR` | +1.061 | Goal reward |
| `WELL_EMPTY_SCR` | -0.005 | Wrong well penalty |
| `INAPPROPRIATE_ACTION_SCR` | -0.005 | Wall bump penalty |
| `TIME_OUT_SCR` | -1 | Episode timeout |
| `TOO_LONG_IN_PHASE` | -0.001 | Phase overtime penalty |
| `REVISIT_SCR` / `SAME_PLACE_SCR` | 0 | No cost for revisiting |

---

## Issues Identified

### 1. Oscillation Exploit
`REVISIT_SCR = 0` and `SAME_PLACE_SCR = 0` mean an agent can turn back and forth in place indefinitely with only the tiny `-0.005` turn cost. A degenerate "spin in place until timeout" policy costs only `~0.005 * steps` and avoids exploration risk entirely.

### 2. Turn-to-Forward Ratio (5:1)
Turns cost 5x more than forward steps. This biases the agent toward long straight paths even when a turn would be the shortest route. Maze navigation inherently requires many turns.

### 3. Extremely Sparse Reward Signal
The agent must navigate a complex maze with barriers, cue associations, and multiple phases, but only receives +1.061 upon reaching the correct well. Everything else is near-zero noise.

### 4. `TOO_LONG_SCR = 55`
This is imported but appears unused in the step function. If ever activated, it would be ~50x the goal reward and would dominate all learning signals.

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
| Set `REVISIT_SCR = -0.002` | **Medium** | Prevents oscillation exploits |
| Reduce turn-to-forward ratio to 2:1 | **Medium** | Current 5:1 biases against necessary turns |
| Consider Bootstrapped Reward Shaping | **Medium** | PBRS using agent's own learned V_hat — no a priori knowledge |
| Consider RND or ICM curiosity | **Medium** | Stronger exploration signal; models novelty-seeking behavior |
| Remove or fix `TOO_LONG_SCR = 55` | **Medium** | Dangerous if ever activated at current magnitude |
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
