"""Decision Transformer for the corner-maze.

Adapted from legacy ``2S2C_task/colab/DTtrainer.ipynb`` cells 1–2 with the
deviations called out in plan §6.2:

  * Real env-derived ``return_to_go`` (preprocessing pass per §4); the
    fake ``1 - steps_to_go/500`` proxy is gone.
  * ``ACTION_WEIGHTS = [1,1,1,20,1]`` cross-entropy hack is dropped.
  * Configurable positional encoding (``learned`` | ``sinusoidal`` |
    ``spatial`` | ``none``); default ``learned``. The legacy notebook
    used ``spatial`` (add pose-vector to R/S/A); kept here for ablation.
  * Variable context size (default 64).

Token layout: each timestep contributes three tokens — ``rtg``, ``state``,
``action`` — interleaved so the transformer attends causally over a
``3·K`` sequence. Action prediction is read out from the state tokens
(positions ``1, 4, 7, ...`` after interleaving).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


PosEncoding = Literal["learned", "sinusoidal", "spatial", "none"]


@dataclass
class DTConfig:
    embed_dim: int = 60
    num_actions: int = 5
    context_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    pos_encoding: PosEncoding = "learned"


# ---------------------------------------------------------------------------
# Positional encoders
# ---------------------------------------------------------------------------

class _NoPosition(nn.Module):
    def forward(self, rtg, state, action, pos_vecs=None):
        return rtg, state, action


class _LearnedPosition(nn.Module):
    """Learned per-token-position embedding shared across R/S/A streams."""
    def __init__(self, context_size: int, embed_dim: int):
        super().__init__()
        # 3 tokens per timestep
        self.position_emb = nn.Embedding(context_size, embed_dim)

    def forward(self, rtg, state, action, pos_vecs=None):
        seq_len = rtg.size(1)
        idx = torch.arange(seq_len, device=rtg.device)
        p = self.position_emb(idx).unsqueeze(0)  # (1, K, D)
        return rtg + p, state + p, action + p


class _SinusoidalPosition(nn.Module):
    def __init__(self, context_size: int, embed_dim: int):
        super().__init__()
        pe = torch.zeros(context_size, embed_dim)
        pos = torch.arange(0, context_size).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(-1)])
        else:
            pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, K, D)

    def forward(self, rtg, state, action, pos_vecs=None):
        seq_len = rtg.size(1)
        p = self.pe[:, :seq_len, :]
        return rtg + p, state + p, action + p


class _SpatialPosition(nn.Module):
    """Legacy quirk: add the pose-vector to each of R/S/A.

    Kept for ablation parity with ``DTtrainer.ipynb``. Requires
    ``pos_vecs`` of shape ``(B, K, embed_dim)``; passes through R/S/A
    unchanged if ``pos_vecs`` is None.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, rtg, state, action, pos_vecs=None):
        if pos_vecs is None:
            return rtg, state, action
        return rtg + pos_vecs, state + pos_vecs, action + pos_vecs


def _build_pos_encoder(cfg: DTConfig) -> nn.Module:
    if cfg.pos_encoding == "none":
        return _NoPosition()
    if cfg.pos_encoding == "learned":
        return _LearnedPosition(cfg.context_size, cfg.embed_dim)
    if cfg.pos_encoding == "sinusoidal":
        return _SinusoidalPosition(cfg.context_size, cfg.embed_dim)
    if cfg.pos_encoding == "spatial":
        return _SpatialPosition(cfg.embed_dim)
    raise ValueError(f"Unknown pos_encoding: {cfg.pos_encoding!r}")


# ---------------------------------------------------------------------------
# Decision Transformer
# ---------------------------------------------------------------------------

class DecisionTransformer(nn.Module):
    """Token order per timestep: (rtg, state, action), interleaved."""

    def __init__(self, cfg: DTConfig | None = None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = DTConfig(**kwargs)
        elif kwargs:
            raise TypeError("Pass DTConfig or kwargs, not both")
        self.cfg = cfg

        self.embed_rtg = nn.Linear(1, cfg.embed_dim)
        self.embed_state = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.embed_action = nn.Linear(cfg.num_actions, cfg.embed_dim)

        self.pos_encoder = _build_pos_encoder(cfg)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.predict_action = nn.Linear(cfg.embed_dim, cfg.num_actions)

    @staticmethod
    def _causal_mask(seq_len: int, device) -> torch.Tensor:
        # mask[i, j] = -inf if j > i else 0 (lower-triangular)
        m = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(
        self,
        rtg: torch.Tensor,           # (B, K, 1)
        state: torch.Tensor,         # (B, K, embed_dim)
        action: torch.Tensor,        # (B, K, num_actions)
        pos_vecs: torch.Tensor | None = None,   # (B, K, embed_dim) — only for "spatial"
    ) -> torch.Tensor:
        """Returns logits of shape ``(B, K, num_actions)``."""
        b, k, _ = state.shape

        r_emb = self.embed_rtg(rtg)
        s_emb = self.embed_state(state)
        a_emb = self.embed_action(action)

        r_emb, s_emb, a_emb = self.pos_encoder(r_emb, s_emb, a_emb, pos_vecs=pos_vecs)

        # Interleave (R, S, A) → (B, 3*K, D)
        stacked = torch.stack((r_emb, s_emb, a_emb), dim=2)
        stacked = stacked.reshape(b, 3 * k, self.cfg.embed_dim)

        mask = self._causal_mask(3 * k, rtg.device)
        out = self.transformer(stacked, mask=mask)
        # Predict action from state tokens (positions 1, 4, 7, ...)
        return self.predict_action(out[:, 1::3, :])

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {"state_dict": self.state_dict(), "cfg": self.cfg.__dict__},
            path,
        )

    @classmethod
    def load(cls, path: str, map_location=None) -> "DecisionTransformer":
        bundle = torch.load(path, map_location=map_location, weights_only=False)
        cfg = DTConfig(**bundle["cfg"])
        model = cls(cfg)
        model.load_state_dict(bundle["state_dict"])
        return model
