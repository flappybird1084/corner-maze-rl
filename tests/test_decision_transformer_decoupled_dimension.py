"""Tests for the decoupled-dim DT variants (softmax + linear-attn)."""
from __future__ import annotations

import torch

from corner_maze_rl.models.decision_transformer_decoupled_dimension import (
    DTConfigDecoupled,
    DecisionTransformerDecoupled,
)
from corner_maze_rl.models.linear_decision_transformer_decoupled_dimension import (
    LinearDTConfigDecoupled,
    LinearDecisionTransformerDecoupled,
)


def _toy_batch(B=2, K=8, state_dim=60, embed_dim=128, A=5):
    rtg = torch.randn(B, K, 1)
    state = torch.randn(B, K, state_dim)
    action = torch.zeros(B, K, A); action[..., 0] = 1.0
    pos = torch.randn(B, K, embed_dim)
    return rtg, state, action, pos


# --- Softmax decoupled ---

def test_softmax_forward_shape_state_dim_lt_embed_dim():
    cfg = DTConfigDecoupled(state_dim=60, embed_dim=128, num_actions=5,
                            context_size=8, num_heads=2, num_layers=1)
    model = DecisionTransformerDecoupled(cfg)
    rtg, state, action, _ = _toy_batch(K=cfg.context_size,
                                       state_dim=cfg.state_dim,
                                       embed_dim=cfg.embed_dim)
    logits = model(rtg, state, action)
    assert logits.shape == (rtg.shape[0], cfg.context_size, cfg.num_actions)


def test_softmax_forward_shape_state_dim_gt_embed_dim():
    cfg = DTConfigDecoupled(state_dim=200, embed_dim=64, num_actions=5,
                            context_size=8, num_heads=4, num_layers=1)
    model = DecisionTransformerDecoupled(cfg)
    rtg, state, action, _ = _toy_batch(K=cfg.context_size,
                                       state_dim=cfg.state_dim,
                                       embed_dim=cfg.embed_dim)
    logits = model(rtg, state, action)
    assert logits.shape == (rtg.shape[0], cfg.context_size, cfg.num_actions)


def test_softmax_embed_state_input_dim_matches_state_dim():
    cfg = DTConfigDecoupled(state_dim=60, embed_dim=128,
                            context_size=4, num_heads=2, num_layers=1)
    model = DecisionTransformerDecoupled(cfg)
    assert model.embed_state.in_features == cfg.state_dim
    assert model.embed_state.out_features == cfg.embed_dim


def test_softmax_save_load_roundtrip(tmp_path):
    cfg = DTConfigDecoupled(state_dim=60, embed_dim=128,
                            context_size=4, num_heads=2, num_layers=1)
    model = DecisionTransformerDecoupled(cfg)
    rtg, state, action, _ = _toy_batch(K=cfg.context_size,
                                       state_dim=cfg.state_dim,
                                       embed_dim=cfg.embed_dim)
    model.eval()
    with torch.no_grad():
        before = model(rtg, state, action)
    path = tmp_path / "dt_decoupled.pt"
    model.save(str(path))
    loaded = DecisionTransformerDecoupled.load(str(path))
    loaded.eval()
    with torch.no_grad():
        after = loaded(rtg, state, action)
    assert torch.allclose(before, after, atol=1e-6)
    assert loaded.cfg.state_dim == 60 and loaded.cfg.embed_dim == 128


def test_softmax_backward_runs():
    cfg = DTConfigDecoupled(state_dim=60, embed_dim=128,
                            context_size=4, num_heads=2, num_layers=1)
    model = DecisionTransformerDecoupled(cfg)
    rtg, state, action, _ = _toy_batch(K=cfg.context_size,
                                       state_dim=cfg.state_dim,
                                       embed_dim=cfg.embed_dim)
    logits = model(rtg, state, action)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.num_actions), action.argmax(-1).reshape(-1))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0
               for p in model.parameters())


# --- Linear decoupled ---

def test_linear_forward_shape_state_dim_lt_embed_dim():
    cfg = LinearDTConfigDecoupled(state_dim=60, embed_dim=128, num_actions=5,
                                  context_size=8, num_heads=2, num_layers=1)
    model = LinearDecisionTransformerDecoupled(cfg)
    rtg, state, action, _ = _toy_batch(K=cfg.context_size,
                                       state_dim=cfg.state_dim,
                                       embed_dim=cfg.embed_dim)
    logits = model(rtg, state, action)
    assert logits.shape == (rtg.shape[0], cfg.context_size, cfg.num_actions)


def test_linear_embed_state_input_dim_matches_state_dim():
    cfg = LinearDTConfigDecoupled(state_dim=60, embed_dim=128,
                                  context_size=4, num_heads=2, num_layers=1)
    model = LinearDecisionTransformerDecoupled(cfg)
    assert model.embed_state.in_features == cfg.state_dim
    assert model.embed_state.out_features == cfg.embed_dim


def test_linear_save_load_roundtrip(tmp_path):
    cfg = LinearDTConfigDecoupled(state_dim=60, embed_dim=128,
                                  context_size=4, num_heads=2, num_layers=1)
    model = LinearDecisionTransformerDecoupled(cfg)
    rtg, state, action, _ = _toy_batch(K=cfg.context_size,
                                       state_dim=cfg.state_dim,
                                       embed_dim=cfg.embed_dim)
    model.eval()
    with torch.no_grad():
        before = model(rtg, state, action)
    path = tmp_path / "lin_decoupled.pt"
    model.save(str(path))
    loaded = LinearDecisionTransformerDecoupled.load(str(path))
    loaded.eval()
    with torch.no_grad():
        after = loaded(rtg, state, action)
    assert torch.allclose(before, after, atol=1e-6)


def test_linear_backward_runs():
    cfg = LinearDTConfigDecoupled(state_dim=60, embed_dim=128,
                                  context_size=4, num_heads=2, num_layers=1)
    model = LinearDecisionTransformerDecoupled(cfg)
    rtg, state, action, _ = _toy_batch(K=cfg.context_size,
                                       state_dim=cfg.state_dim,
                                       embed_dim=cfg.embed_dim)
    logits = model(rtg, state, action)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.num_actions), action.argmax(-1).reshape(-1))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0
               for p in model.parameters())


def test_linear_feature_map_is_relu_plus_eps():
    cfg = LinearDTConfigDecoupled(state_dim=60, embed_dim=128,
                                  context_size=4, num_heads=2, num_layers=1,
                                  feature_eps=1e-3)
    model = LinearDecisionTransformerDecoupled(cfg)
    layer = model.layers[0].attn
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = torch.tensor([1e-3, 1e-3, 1e-3, 0.5 + 1e-3, 2.0 + 1e-3])
    assert torch.allclose(layer._phi(x), expected, atol=1e-6)


def test_softmax_and_linear_differ_at_same_seed():
    torch.manual_seed(0)
    sm = DecisionTransformerDecoupled(DTConfigDecoupled(
        state_dim=60, embed_dim=128, context_size=4,
        num_heads=2, num_layers=1, pos_encoding="none",
    ))
    torch.manual_seed(0)
    lin = LinearDecisionTransformerDecoupled(LinearDTConfigDecoupled(
        state_dim=60, embed_dim=128, context_size=4,
        num_heads=2, num_layers=1, pos_encoding="none",
    ))
    rtg, state, action, _ = _toy_batch(K=4, state_dim=60, embed_dim=128)
    sm.eval(); lin.eval()
    with torch.no_grad():
        out_sm = sm(rtg, state, action)
        out_lin = lin(rtg, state, action)
    assert not torch.allclose(out_sm, out_lin, atol=1e-3)
