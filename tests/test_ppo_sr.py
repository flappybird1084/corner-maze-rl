"""Tests for PPO + SR agent ports."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from corner_maze_rl.encoders.state_vectors import (
    NUM_LOCATIONS,
    build_position_projection_matrix,
    compute_obs_dim,
    compute_obs_dim_phase,
    generate_state_vector,
    generate_state_vector_onehot,
    generate_state_vector_phase,
)
from corner_maze_rl.models.ppo import ActorCritic, PPOAgent
from corner_maze_rl.models.sr import SRAgent


# ---------------------------------------------------------------------------
# state_vectors helpers
# ---------------------------------------------------------------------------

def test_compute_obs_dim_default():
    assert compute_obs_dim() == 236  # 49*4 + 4*10
    assert compute_obs_dim_phase() == 239
    assert compute_obs_dim(n_wm_units=5) == NUM_LOCATIONS * 4 + 4 * 5


def test_generate_state_vector_shape_dtype():
    v = generate_state_vector((6, 6, 0), [0, 0, 0, 0])
    assert v.shape == (236,) and v.dtype == np.float32


def test_generate_state_vector_onehot_has_one_pose_active():
    v = generate_state_vector_onehot((6, 6, 0), [10, 10, 10, 10])
    # n_wm depleted (timer >= n_wm), so wm part should be all zeros.
    # Pose part has exactly one active cell (no adjacency).
    assert v[:NUM_LOCATIONS * 4].sum() == 1.0
    assert v[NUM_LOCATIONS * 4:].sum() == 0.0


def test_generate_state_vector_phase_includes_phase_one_hot():
    v = generate_state_vector_phase((6, 6, 0), [0, 0, 0, 0], env=None)
    assert v.shape == (239,)
    # Phase one-hot lives at indices 196..198
    assert v[196:199].sum() == 1.0


def test_position_projection_matrix_collapses_directions():
    P = build_position_projection_matrix(n_wm_units=10)
    assert P.shape == (236, 89)
    # A pose-only state with direction=0 at location 5 should project the
    # same as direction=2 at location 5.
    s0 = np.zeros(236, dtype=np.float32); s0[0 * NUM_LOCATIONS + 5] = 1
    s2 = np.zeros(236, dtype=np.float32); s2[2 * NUM_LOCATIONS + 5] = 1
    assert np.allclose(s0 @ P, s2 @ P)


# ---------------------------------------------------------------------------
# ActorCritic
# ---------------------------------------------------------------------------

def test_actor_critic_shapes():
    ac = ActorCritic(obs_dim=12, action_dim=5, n_hidden_units=8)
    x = torch.randn(3, 12)
    logits, value = ac(x)
    assert logits.shape == (3, 5)
    assert value.shape == (3, 1)


def test_actor_critic_no_hidden():
    ac = ActorCritic(obs_dim=12, action_dim=5, n_hidden_units=0)
    x = torch.randn(2, 12)
    logits, value = ac(x)
    assert logits.shape == (2, 5)
    # actor head init gain=0.01 → logits should be small
    assert logits.abs().max().item() < 1.0


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

@pytest.fixture
def ppo_agent():
    return PPOAgent(
        obs_dim=12,
        action_dim=5,
        n_hidden_units=8,
        n_steps=8,
        batch_size=4,
        n_epochs=2,
    )


def test_ppo_select_action_returns_valid(ppo_agent):
    state = np.random.randn(12).astype(np.float32)
    mask = [True, True, False, True, True]  # action 2 disallowed
    action, info = ppo_agent.select_action(state, mask)
    assert 0 <= action < 5 and action != 2
    assert "value" in info and "log_prob" in info


def test_ppo_buffer_fills_then_ready(ppo_agent):
    state = np.zeros(12, dtype=np.float32)
    for i in range(8):
        ppo_agent.add_experience(state, action=0, reward=0.0, done=False,
                                 value=0.0, log_prob=0.0)
    assert ppo_agent.is_ready_to_update() is True


def test_ppo_update_runs_and_resets(ppo_agent):
    state = np.zeros(12, dtype=np.float32)
    for _ in range(8):
        ppo_agent.add_experience(state, action=0, reward=0.1, done=False,
                                 value=0.0, log_prob=-1.6)
    metrics = ppo_agent.update(next_state=state, next_done=False)
    assert {"pg_loss", "v_loss", "entropy"} <= metrics.keys()
    assert ppo_agent._step == 0
    assert not ppo_agent.is_ready_to_update()


def test_ppo_save_load_roundtrip(ppo_agent, tmp_path):
    path = tmp_path / "ppo.pt"
    ppo_agent.save(str(path))
    new = PPOAgent(obs_dim=12, action_dim=5, n_hidden_units=8,
                   n_steps=8, batch_size=4, n_epochs=2)
    new.load(str(path))
    # Same parameters → same forward output
    x = torch.randn(2, 12)
    p1 = ppo_agent.policy(x)[0]
    p2 = new.policy(x)[0]
    assert torch.allclose(p1, p2, atol=1e-6)


# ---------------------------------------------------------------------------
# SRAgent
# ---------------------------------------------------------------------------

def test_sr_init_shapes_default():
    a = SRAgent(obs_dim=12, action_dim=5)
    assert a.M.shape == (5, 12, 12)
    assert a.w.shape == (12,)
    # M starts as stacked identities
    for k in range(5):
        assert np.allclose(a.M[k], np.eye(12))


def test_sr_position_only_w_uses_smaller_w():
    a = SRAgent(obs_dim=236, action_dim=5, position_only_w=True)
    assert a.w.shape == (89,)


def test_sr_select_action_obeys_mask():
    a = SRAgent(obs_dim=12, action_dim=5, epsilon_start=0.0)
    state = np.random.randn(12).astype(np.float32)
    action, info = a.select_action(state, [True, False, True, True, False])
    assert action in (0, 2, 3)
    assert "q_values" in info


def test_sr_update_decays_epsilon():
    a = SRAgent(obs_dim=12, action_dim=5, epsilon_start=1.0,
                epsilon_decay=0.5, epsilon_end=0.0)
    state = np.random.randn(12).astype(np.float32)
    a.add_experience(state, action=0, reward=1.0, done=False)
    next_state = np.random.randn(12).astype(np.float32)
    metrics = a.update(next_state, next_done=False)
    assert metrics["epsilon"] == pytest.approx(0.5)


def test_sr_save_load_roundtrip(tmp_path):
    a = SRAgent(obs_dim=12, action_dim=5)
    state = np.random.randn(12).astype(np.float32)
    a.add_experience(state, action=0, reward=1.0, done=False)
    a.update(state, next_done=False)
    path = tmp_path / "sr.npz"
    a.save(str(path))
    b = SRAgent(obs_dim=12, action_dim=5)
    b.load(str(path))
    assert np.array_equal(a.M, b.M)
    assert np.array_equal(a.w, b.w)
    assert a.epsilon == b.epsilon
