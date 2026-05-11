"""Linear-attention Decision Transformer.

Same I/O contract as ``DecisionTransformer``: ``(rtg, state, action)`` context
windows of length ``K`` → ``(B, K, num_actions)`` logits read at state-token
positions. Swaps softmax multi-head attention for causal **linear attention**
(Katharopoulos+ 2020, ``phi(x) = elu(x) + 1``).

Linear attention can be evaluated as a recurrence whose per-step state is a
fixed ``O(d_k * d_v)`` matrix per head per layer — independent of sequence
length. This module implements the parallel form (cumulative sums) used at
training time; the same weights support an ``O(1)``-per-step recurrent
rollout (not implemented here).
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
class LinearDTConfig:
    embed_dim: int = 60
    num_actions: int = 5
    context_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    pos_encoding: PosEncoding = "learned"


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


def _build_pos_encoder(cfg: LinearDTConfig) -> nn.Module:
    if cfg.pos_encoding == "none":
        return _NoPosition()
    if cfg.pos_encoding == "learned":
        return _LearnedPosition(cfg.context_size, cfg.embed_dim)
    if cfg.pos_encoding == "sinusoidal":
        return _SinusoidalPosition(cfg.context_size, cfg.embed_dim)
    if cfg.pos_encoding == "spatial":
        return _SpatialPosition(cfg.embed_dim)
    raise ValueError(f"Unknown pos_encoding: {cfg.pos_encoding!r}")


class CausalLinearMHA(nn.Module):
    """Multi-head causal linear attention with ``phi(x) = elu(x) + 1``.

    Builds running cumulative sums of ``phi(K) v^T`` and ``phi(K)`` along the
    sequence axis, then reads out ``phi(Q_t) S_t / (phi(Q_t) z_t)``.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        qkv = self.qkv_proj(x).view(b, l, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)            # (3, B, H, L, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.einsum("bhld,bhle->bhlde", k, v).cumsum(dim=2)   # (B,H,L,d,d)
        k_cum = k.cumsum(dim=2)                                      # (B,H,L,d)
        num = torch.einsum("bhld,bhlde->bhle", q, kv)                # (B,H,L,d)
        den = torch.einsum("bhld,bhld->bhl", q, k_cum).unsqueeze(-1).clamp_min(1e-6)
        out = num / den
        out = out.transpose(1, 2).contiguous().view(b, l, self.num_heads * self.head_dim)
        return self.dropout(self.out_proj(out))


class LinearAttnEncoderLayer(nn.Module):
    """Pre-norm encoder block: linear-attn + FFN, each with residual."""
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CausalLinearMHA(embed_dim, num_heads, dropout=dropout)
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


class LinearDecisionTransformer(nn.Module):
    """Drop-in replacement for ``DecisionTransformer`` with causal linear attention."""

    def __init__(self, cfg: LinearDTConfig | None = None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = LinearDTConfig(**kwargs)
        elif kwargs:
            raise TypeError("Pass LinearDTConfig or kwargs, not both")
        self.cfg = cfg

        self.embed_rtg = nn.Linear(1, cfg.embed_dim)
        self.embed_state = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.embed_action = nn.Linear(cfg.num_actions, cfg.embed_dim)

        self.pos_encoder = _build_pos_encoder(cfg)

        self.layers = nn.ModuleList([
            LinearAttnEncoderLayer(
                cfg.embed_dim, cfg.num_heads, cfg.dim_feedforward, cfg.dropout,
            )
            for _ in range(cfg.num_layers)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.predict_action = nn.Linear(cfg.embed_dim, cfg.num_actions)

    def forward(
        self,
        rtg: torch.Tensor,           # (B, K, 1)
        state: torch.Tensor,         # (B, K, embed_dim)
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
    def load(cls, path: str, map_location=None) -> "LinearDecisionTransformer":
        bundle = torch.load(path, map_location=map_location, weights_only=False)
        cfg = LinearDTConfig(**bundle["cfg"])
        model = cls(cfg)
        model.load_state_dict(bundle["state_dict"])
        return model
