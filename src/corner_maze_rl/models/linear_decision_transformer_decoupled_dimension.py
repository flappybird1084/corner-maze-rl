"""Decoupled-dimension linear-attention DT.

Two changes vs. ``linear_decision_transformer.py``:

1. ``state_dim`` is decoupled from ``embed_dim`` (same rationale as
   ``decision_transformer_decoupled_dimension.py``: lets us widen the
   internal hidden dim past the encoder's output, raising
   ``d_head = embed_dim / num_heads`` past linear attention's rank-
   ``d_head`` expressivity ceiling).

2. Feature map: ``phi(x) = relu(x) + eps`` instead of ``elu(x) + 1``.
   Cheaper per element on most accelerators; same non-negativity
   property. The trade is half-gradient on negatives, which is a wash
   in practice once the larger ``d_head`` gives the model enough
   capacity to compensate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


PosEncoding = Literal["learned", "sinusoidal", "spatial", "none"]


@dataclass
class LinearDTConfigDecoupled:
    state_dim: int = 60
    embed_dim: int = 128
    num_actions: int = 5
    context_size: int = 64
    num_heads: int = 2
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    pos_encoding: PosEncoding = "learned"
    feature_eps: float = 1e-6


class _NoPosition(nn.Module):
    def forward(self, rtg, state, action, pos_vecs=None):
        return rtg, state, action


class _LearnedPosition(nn.Module):
    def __init__(self, context_size: int, embed_dim: int):
        super().__init__()
        self.position_emb = nn.Embedding(context_size, embed_dim)

    def forward(self, rtg, state, action, pos_vecs=None):
        seq_len = rtg.size(1)
        idx = torch.arange(seq_len, device=rtg.device)
        p = self.position_emb(idx).unsqueeze(0)
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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, rtg, state, action, pos_vecs=None):
        seq_len = rtg.size(1)
        p = self.pe[:, :seq_len, :]
        return rtg + p, state + p, action + p


class _SpatialPosition(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, rtg, state, action, pos_vecs=None):
        if pos_vecs is None:
            return rtg, state, action
        return rtg + pos_vecs, state + pos_vecs, action + pos_vecs


def _build_pos_encoder(cfg: LinearDTConfigDecoupled) -> nn.Module:
    if cfg.pos_encoding == "none":
        return _NoPosition()
    if cfg.pos_encoding == "learned":
        return _LearnedPosition(cfg.context_size, cfg.embed_dim)
    if cfg.pos_encoding == "sinusoidal":
        return _SinusoidalPosition(cfg.context_size, cfg.embed_dim)
    if cfg.pos_encoding == "spatial":
        return _SpatialPosition(cfg.embed_dim)
    raise ValueError(f"Unknown pos_encoding: {cfg.pos_encoding!r}")


class CausalLinearMHADecoupled(nn.Module):
    """Causal linear MHA with ``phi(x) = relu(x) + eps``."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        feature_eps: float = 1e-6,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feature_eps = feature_eps
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) + self.feature_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        qkv = self.qkv_proj(x).view(b, l, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self._phi(q)
        k = self._phi(k)
        kv = torch.einsum("bhld,bhle->bhlde", k, v).cumsum(dim=2)
        k_cum = k.cumsum(dim=2)
        num = torch.einsum("bhld,bhlde->bhle", q, kv)
        den = torch.einsum("bhld,bhld->bhl", q, k_cum).unsqueeze(-1).clamp_min(1e-6)
        out = num / den
        out = out.transpose(1, 2).contiguous().view(b, l, self.num_heads * self.head_dim)
        return self.dropout(self.out_proj(out))


class LinearAttnEncoderLayerDecoupled(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        feature_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CausalLinearMHADecoupled(
            embed_dim, num_heads, dropout=dropout, feature_eps=feature_eps,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LinearDecisionTransformerDecoupled(nn.Module):
    """Decoupled-dim linear-attn DT with ``relu+eps`` feature map."""

    def __init__(self, cfg: LinearDTConfigDecoupled | None = None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = LinearDTConfigDecoupled(**kwargs)
        elif kwargs:
            raise TypeError("Pass LinearDTConfigDecoupled or kwargs, not both")
        self.cfg = cfg

        self.embed_rtg = nn.Linear(1, cfg.embed_dim)
        self.embed_state = nn.Linear(cfg.state_dim, cfg.embed_dim)
        self.embed_action = nn.Linear(cfg.num_actions, cfg.embed_dim)

        self.pos_encoder = _build_pos_encoder(cfg)

        self.layers = nn.ModuleList([
            LinearAttnEncoderLayerDecoupled(
                cfg.embed_dim, cfg.num_heads, cfg.dim_feedforward,
                cfg.dropout, feature_eps=cfg.feature_eps,
            )
            for _ in range(cfg.num_layers)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.predict_action = nn.Linear(cfg.embed_dim, cfg.num_actions)

    def forward(
        self,
        rtg: torch.Tensor,           # (B, K, 1)
        state: torch.Tensor,         # (B, K, state_dim)
        action: torch.Tensor,        # (B, K, num_actions)
        pos_vecs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, k, _ = state.shape
        r = self.embed_rtg(rtg)
        s = self.embed_state(state)
        a = self.embed_action(action)
        r, s, a = self.pos_encoder(r, s, a, pos_vecs=pos_vecs)
        x = torch.stack((r, s, a), dim=2).reshape(b, 3 * k, self.cfg.embed_dim)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.predict_action(x[:, 1::3, :])

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg.__dict__}, path)

    @classmethod
    def load(cls, path: str, map_location=None) -> "LinearDecisionTransformerDecoupled":
        bundle = torch.load(path, map_location=map_location, weights_only=False)
        cfg = LinearDTConfigDecoupled(**bundle["cfg"])
        model = cls(cfg)
        model.load_state_dict(bundle["state_dict"])
        return model
