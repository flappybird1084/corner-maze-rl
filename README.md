# corner-maze-rl

Decision Transformer + PPO + Successor-Representation baselines on a 13×13 corner-maze MiniGrid environment, trained against yoked rodent behavior. Built for student use via Colab and locally in VS Code.

> **Status:** planning phase, pre-build. Design doc in [md/dt-repo-plan.md](md/dt-repo-plan.md). The legacy research repo with the original env, yoking pipeline, and prior experiments lives at [corner-maze-rl-legacy](https://github.com/ryangrg/corner-maze-rl-legacy) — code and data will be ported in per the manifest in §16 of the design doc.

## What this is

A teaching repo for offline / behavior-cloning-style RL on a real neuroscience task:

- **Environment** — 13×13 MiniGrid corner maze with structured trial phases (exposure → pretrial → trial → ITI), four wells, four cues, configurable session paradigms (PI, VC, PI+VC).
- **Yoked dataset** — action sequences derived from real rodent behavioral tracking (≈250 sessions across PI / VC / PI+VC training groups), with env-derived per-step rewards and per-trial return-to-go.
- **Models** — Decision Transformer (the headline model), plus PPO and SR baselines for comparison.
- **Encoders** — composable state encoders (grid-cell pose vectors, pretrained visual CNN, one-hot tabular, reward-history) all standardized to 60D.
- **Eval protocol** — IQM + stratified bootstrap CIs, drawdown reliability, performance profiles, kill-switch on flat learning curves. Grounded in the empirical-RL methodology canon ([Henderson 2018], [Agarwal 2021], [Patterson 2024]).

## Quickstart

> Build is in progress. The commands below describe the *target* student workflow; not all are wired up yet.

### Local + VS Code (recommended)
```bash
git clone https://github.com/ryangrg/corner-maze-rl.git
cd corner-maze-rl
# point VS Code at your existing ai-venv, or:
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Colab-in-browser (limited)
The interactive `notebooks/` UI is **VS Code / local Jupyter only** — Colab's iframe-sandboxed widget framework breaks rendering and keyboard input. For non-interactive components (training scripts, dataset builds, evaluation), Colab is fine:
```python
!pip install git+https://github.com/ryangrg/corner-maze-rl.git
from corner_maze_rl import ...
```
If you need cloud + the interactive UI, point VS Code at a remote Jupyter kernel (e.g. GitHub Codespaces or a managed runtime) — the IDE renders widgets locally even when compute is remote.

## Repository layout

```
corner-maze-rl/
├── README.md
├── md/                        # design + spec docs (start here)
│   ├── dt-repo-plan.md        # full design doc — start here
│   ├── environment-architecture.md
│   ├── maze-behavior-spec.md  # 2S2C task rules
│   ├── reward-structure-analysis.md
│   ├── sr-yoked-negative-results.md
│   └── yoked-data-from-legacy-data.md  # exposure-only yoked dataset converter
├── src/corner_maze_rl/        # (to be built — see plan §3)
├── notebooks/                 # interactive notebooks (VS Code recommended)
├── data/                      # (gitignored; setup script will populate)
└── LICENSE
```

## Where to start reading

1. [md/dt-repo-plan.md](md/dt-repo-plan.md) — the full design doc. Sections 1–3 give the goals and layout in five minutes; §4 (reward + RTG), §5 (encoders), §8 (kill switch), and §10 (eval protocol) are the parts that distinguish this from a textbook DT.
2. [md/maze-behavior-spec.md](md/maze-behavior-spec.md) — task rules: arm structure, trial phases, turn detection, well-visit, reward triggering.
3. [md/environment-architecture.md](md/environment-architecture.md) — env spec the code is built against.
4. [md/reward-structure-analysis.md](md/reward-structure-analysis.md) — *why* the reward shaping is what it is.
5. [md/sr-yoked-negative-results.md](md/sr-yoked-negative-results.md) — a documented negative result (SR fails on yoked data); the plan's §10.3 frames how to report findings like this scientifically.
6. [md/yoked-data-from-legacy-data.md](md/yoked-data-from-legacy-data.md) — how to build a (currently exposure-only) yoked dataset from `../rat-neural-model-2` when the canonical legacy yoking output isn't available; covers schema mismatch, conversion strategy, and limitations.

## Yoked dataset

The plan's `setup_data.sh` assumes a canonical 3-table yoked dataset has been produced by the legacy yoking pipeline; that pipeline isn't yet published. As a stopgap, `scripts/build_yoked_from_legacy_data.py` builds an **exposure-only** yoked dataset from a sibling `rat-neural-model-2` checkout. Full design notes + caveats in [md/yoked-data-from-legacy-data.md](md/yoked-data-from-legacy-data.md).

```bash
.venv/bin/python scripts/build_yoked_from_legacy_data.py \
    --legacy-data-root ../rat-neural-model-2
.venv/bin/python -m corner_maze_rl.scripts.build_returns_dataset \
    --dataset-dir data/yoked/dataset \
    --actions-variant real_pretrial
```

## Related projects

- [corner-maze-rl-legacy](https://github.com/ryangrg/corner-maze-rl-legacy) — the prior research repo. Source of the env, yoking pipeline, yoked dataset, and prior PPO/SR experiments. This repo (`corner-maze-rl`) ports/refactors content from it per the manifest in [md/dt-repo-plan.md](md/dt-repo-plan.md) §16.

## License

See [LICENSE](LICENSE).
