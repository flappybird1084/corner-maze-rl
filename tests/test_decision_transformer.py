"""Tests for corner_maze_rl.models.decision_transformer."""
from __future__ import annotations

import torch

from corner_maze_rl.models.decision_transformer import DecisionTransformer, DTConfig


def _toy_batch(B=2, K=8, D=60, A=5):
    rtg = torch.randn(B, K, 1)
    state = torch.randn(B, K, D)
    action = torch.zeros(B, K, A)
    action[:, :, 0] = 1.0  # one-hot dummy
    pos = torch.randn(B, K, D)
    return rtg, state, action, pos


def test_forward_output_shape_default_pos():
    cfg = DTConfig(embed_dim=60, num_actions=5, context_size=8,
                   num_heads=2, num_layers=1)
    model = DecisionTransformer(cfg)
    rtg, state, action, pos = _toy_batch(K=cfg.context_size)
    logits = model(rtg, state, action)
    assert logits.shape == (rtg.shape[0], cfg.context_size, cfg.num_actions)


def test_pos_encoding_modes_produce_different_logits():
    """The four pos-encoding modes should not all collapse to the same output."""
    rtg, state, action, pos = _toy_batch(K=8)
    outs = []
    for mode in ("none", "learned", "sinusoidal", "spatial"):
        torch.manual_seed(0)  # reset RNG so init is comparable
        cfg = DTConfig(embed_dim=60, num_actions=5, context_size=8,
                       num_heads=2, num_layers=1, pos_encoding=mode)
        model = DecisionTransformer(cfg)
        outs.append(model(rtg, state, action, pos_vecs=pos).detach())
    # All four should have correct shape
    for o in outs:
        assert o.shape == (rtg.shape[0], 8, 5)
    # And not all four should be identical (pos encoding does *something*).
    # Check at least one pair differs.
    pairwise_equal = sum(torch.allclose(outs[i], outs[j]) for i in range(4) for j in range(i + 1, 4))
    assert pairwise_equal < 6  # 4-choose-2 = 6 pairs, must not all match


def test_spatial_with_no_pos_vecs_passes_through():
    """spatial mode must not crash if pos_vecs is None — it should just pass R/S/A through."""
    cfg = DTConfig(context_size=8, num_layers=1, num_heads=2, pos_encoding="spatial")
    model = DecisionTransformer(cfg)
    rtg, state, action, _ = _toy_batch(K=8)
    logits = model(rtg, state, action, pos_vecs=None)
    assert logits.shape == (rtg.shape[0], 8, 5)


def test_save_and_load_roundtrip(tmp_path):
    cfg = DTConfig(embed_dim=60, num_actions=5, context_size=4,
                   num_heads=2, num_layers=1, pos_encoding="learned")
    model = DecisionTransformer(cfg)
    rtg, state, action, _ = _toy_batch(K=cfg.context_size)
    model.eval()
    with torch.no_grad():
        before = model(rtg, state, action)

    path = tmp_path / "dt.pt"
    model.save(str(path))
    loaded = DecisionTransformer.load(str(path))
    loaded.eval()
    with torch.no_grad():
        after = loaded(rtg, state, action)
    assert torch.allclose(before, after, atol=1e-6)
    assert loaded.cfg.context_size == 4
    assert loaded.cfg.pos_encoding == "learned"


def test_causal_mask_blocks_future_attention():
    """Smoke test: with one transformer layer and only-zero state for future
    positions, predictions for early positions should not depend on later RTG."""
    cfg = DTConfig(embed_dim=60, num_actions=5, context_size=4,
                   num_heads=2, num_layers=1, pos_encoding="none")
    model = DecisionTransformer(cfg)
    model.eval()
    rtg, state, action, _ = _toy_batch(B=1, K=4)
    rtg2 = rtg.clone()
    rtg2[:, -1, :] = rtg[:, -1, :] + 100.0  # only modify the LAST timestep's RTG
    with torch.no_grad():
        out1 = model(rtg, state, action)
        out2 = model(rtg2, state, action)
    # First timestep's prediction must be unaffected by changes to the future.
    assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5)
    # But the last timestep's prediction CAN differ.


def test_unknown_pos_encoding_raises():
    import pytest
    with pytest.raises(ValueError):
        DTConfig(pos_encoding="bogus")  # constructs OK; raises in build
        # actually pos_encoding is just a Literal annotation; the model raises:
        DecisionTransformer(DTConfig(pos_encoding="bogus", context_size=4,
                                       num_layers=1, num_heads=2))
