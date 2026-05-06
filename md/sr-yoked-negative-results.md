# Vanilla Tabular SR-TD: Negative Results in No-Teleport Session Task

**Date:** 2026-05-03

> **CAUTION — Design flaw identified (2026-05-04):** These experiments are **poisoned by two design errors**: (1) the yoked training data included sessions where the goal corner rotated between sessions, teaching w contradictory reward locations; (2) exposure phases were included in yoking, where all 4 wells give reward. These errors make goal discrimination impossible regardless of SR architecture. **Do NOT cite these results as evidence that SR cannot handle the task.** The closed-form control proves the representation is adequate. A clean re-run with fixed goal + trial-only data is needed before drawing conclusions about SR model capability.

## Task Description

A tabular Successor Representation agent (de Cothi et al. 2022 style) was tested on the Corner Maze environment: a 13x13 MiniGrid with pretrial, trial, and ITI phases. Barrier configurations change between phases (pretrial dead-ends, trial choice points, ITI open corridors). The agent navigates continuously without teleportation between phases. State vector is a one-hot encoding of pose (196 dims for 49 positions x 4 directions) plus decaying reward timers (4 x n_wm_units).

The experiment tested whether yoked training on real rodent action sequences (from the yoking pipeline) could bootstrap an SR agent's M matrix and reward weights w, enabling it to then perform goal-directed navigation autonomously.

## Experiment Summary

All 14+ configurations failed to reach acquisition criterion during the free navigation phase. None progressed to probe sessions. **However, these failures are primarily attributable to the design flaws noted above, not to SR model limitations.**

### Configurations Tested

| # | Config Name | gamma | lr_m | lr_w | M Strategy | State Dims | Result |
|---|-------------|-------|------|------|------------|------------|--------|
| 1 | SR_tabular (non-yoked baseline) | 0.999 | 0.001 | 1.0 | Online TD(0) | 236 | No criterion |
| 2 | SR_yoked_CM002 | 0.999 | 0.001 | 1.0 | Yoked then reset | 236 | No criterion |
| 3 | SR_yoked_CM023 lr_w=1.0 | 0.999 | 0.001 | 1.0 | Yoked then reset | 236 | No criterion |
| 4 | SR_yoked_CM023 lr_w=0.01 | 0.999 | 0.001 | 0.01 | Yoked then reset | 236 | No criterion |
| 5 | SR_yoked_CM023 lr_w=0.1 | 0.999 | 0.001 | 0.1 | Yoked then reset | 236 | No criterion |
| 6 | SR_yoked_CM023 lr_w=0.5 | 0.999 | 0.001 | 0.5 | Yoked then reset | 236 | No criterion |
| 7 | SR_yoked_CM023 lr_w=0.05 | 0.999 | 0.001 | 0.05 | Yoked then reset | 236 | No criterion |
| 8 | SR_yoked_posw (position-only w) | 0.999 | 0.01 | 0.1 | Yoked then reset, boosted goal w | 236 | No criterion |
| 9 | SR_phase keep_M gamma=0.99 | 0.99 | 0.01/0.005 | 0.1/1.0 | Keep yoked M, free lr_m=0.005 | 239 | No criterion |
| 10 | SR_phase reset_M gamma=0.99 | 0.99 | 0.01/0.005 | 0.1/1.0 | Reset M, free lr_m=0.005 | 239 | No criterion |
| 11 | SR_phase reset_M gamma=0.95 | 0.95 | 0.01/0.005 | 0.1/1.0 | Reset M, free lr_m=0.005 | 239 | No criterion |
| 12 | SR_phase frozen_M gamma=0.95 | 0.95 | 0.01/0.0 | 0.1/1.0 | Keep yoked M, freeze (lr_m=0) | 239 | No criterion |
| 13 | SR_phase frozen_M gamma=0.99 | 0.99 | 0.01/0.0 | 0.1/1.0 | Keep yoked M, freeze (lr_m=0) | 239 | No criterion |
| 14 | SR_phase frozen_M gamma=0.95 (rerun) | 0.95 | 0.01/0.0 | 0.1/1.0 | Keep yoked M, freeze (lr_m=0) | 239 | No criterion |

**State dims:** 236 = 196 pose (one-hot) + 40 reward timers (4 wells x 10 units). 239 = 236 + 3 phase indicators (pretrial/trial/ITI).

**M Strategy column:** "Yoked then reset" means M was learned from rat action sequences, then reset to identity before free navigation. "Keep yoked M" means the rat-learned M was preserved. "Freeze" means lr_m=0 during free phase.

## Key Findings

1. **M diverges under online TD(0) updates.** The successor matrix M(a) is updated per-step as M(a)[s] += lr_m * (e_s + gamma * M(a)[s'] - M(a)[s]). In a 236-dim state space with 5 actions, M has 5 x 236 x 236 = 278,480 parameters. Online TD(0) with a tabular representation produces noisy, unstable updates that prevent M from converging to the true successor representation. This is the primary failure mode.

2. **Yoked M (rat transitions) does not transfer to MiniGrid agent.** The rat's transition dynamics differ from the agent's: the rat moves continuously through zones while the agent takes discrete grid steps. Even when M is learned accurately from rat data, the resulting successor predictions do not generalize to the agent's own trajectories. Resetting M after yoking was necessary, but then the agent faces the same online learning instability.

3. **Phase-aware state vector (239-dim) did not resolve the instability.** Adding 3-dim phase encoding (pretrial/trial/ITI one-hot) to distinguish the same physical location across different barrier configurations was theoretically sound but did not address the fundamental issue of online TD(0) learning instability in high-dimensional tabular M.

4. **Freezing M after yoking eliminates learning instability but produces wrong Q-values.** With lr_m=0, M is stable but reflects the rat's transition dynamics, not the agent's. Q = M * w produces action values based on where the _rat_ would end up, not where the _agent_ would end up.

5. **Hyperparameter sweeps across gamma, lr_m, lr_w had no effect on the core failure.** The problem is structural: online tabular SR-TD with this many states cannot learn a stable M within the episode budget of this task.

## Scope of This Negative Result

This result is specific to: **vanilla online tabular SR-TD(0)** combined with **this task structure** (no teleportation, phase-changing barriers, 236-239 dim state space).

It does NOT show that SR representations cannot handle this task.

## Closed-Form Control (2026-05-04)

**Result: The SR representation CAN solve this task.** Online TD is the bottleneck, not the representation.

A control experiment (`experiments/sr_closedform_control.py`) estimated the one-step transition matrix T from 500k random-policy steps (475 unique states discovered), then computed M = (I - γT)^{-1} analytically.

| Metric | Value |
|--------|-------|
| Unique tabular states | 475 |
| Condition number of (I - γT) | 327 (well-conditioned) |
| Trials correct (manual R, ε=0.1) | 13/20 across 10 episodes |
| Best episodes | 4/4, 4/6 |
| Pure greedy (ε=0) | Trapped in pretrial (Q=0, no gradient) |

**Key findings:**
- The closed-form M produces genuine goal-seeking behavior during trial phase
- Agent requires ε-greedy exploration to traverse non-rewarded phases (pretrial/ITI have Q=0)
- Feature-based M reconstruction fails (Φ^TΦ condition number ~10^52) — the 239-dim feature space with overlapping activations is ill-conditioned for linear SR
- The task is within SR's representational capacity; online TD(0) simply cannot converge to the correct M in this state space

**Conclusion:** The failure documented above is a **learning procedure failure**, not a representational one. The SR decomposition Q(s,a) = M(s,a) · w correctly identifies goal-directed paths when M is computed analytically.

## Paths Forward

- **Batch TD / experience replay**: Stabilize M learning with replay buffers
- **Model-based M computation**: The env is fully known — compute T directly from grid structure
- **Neural SR**: Learned features to reduce effective dimensionality (de Cothi et al. 2022 style)
- **Lower-dimensional state**: Direction-collapsed (~49 states) where online TD might converge
- **Eligibility traces / TD(λ)**: Multi-step returns to propagate signal faster through long corridors

## References

- de Cothi, W., Nyberg, N., Griesbauer, E. M., et al. (2022). Predictive maps in rats and humans for spatial navigation. *Current Biology*, 32(17), 3676-3689.
- Dayan, P. (1993). Improving generalization for temporal difference learning: The successor representation. *Neural Computation*, 5(4), 613-624.

## Archived Data

All experiment model data has been moved to `archive/sr_yoked_experiments/`. The superseded training script `train_sr_tabular_yoked.py` is also archived there. The phase-aware version `experiments/train_sr_tabular_yoked_phase.py` is retained as the cleanest implementation and reference for future work.
