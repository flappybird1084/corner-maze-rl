"""Encoder protocol shared by all vector state encoders.

The egocentric-image branch (CnnPolicy) is *not* a StateEncoder — it stays
as a raw 21x21x3 obs and is consumed by a CNN extractor directly. See
plan §5.4. This Protocol governs only the composable vector encoders that
feed into the DT / MLP-PPO / SR paths.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class StateEncoder(Protocol):
    """Maps a pose (and optional layout) to a fixed-dim vector.

    Attributes
    ----------
    output_dim : int
        Length of the vector returned by ``encode``.
    """

    output_dim: int

    def encode(
        self,
        x: int,
        y: int,
        direction: int,
        layout: str | None = None,
    ) -> np.ndarray:
        """Return a 1-D float32 vector of length ``output_dim``.

        Encoders that don't depend on the maze layout (grid_cells, one_hot_pose,
        reward_history) ignore ``layout``; ``visual_cnn`` uses it to disambiguate
        renderings under different barrier configurations.
        """
        ...


class CompositeEncoder:
    """Concatenate several StateEncoders into one.

    ``output_dim`` is the sum of components' dims; ``encode`` concatenates
    along axis 0 in the order encoders were registered.
    """

    def __init__(self, encoders: list[StateEncoder]):
        if not encoders:
            raise ValueError("CompositeEncoder requires at least one component")
        self._encoders = list(encoders)
        self.output_dim = sum(e.output_dim for e in self._encoders)

    def encode(
        self,
        x: int,
        y: int,
        direction: int,
        layout: str | None = None,
    ) -> np.ndarray:
        parts = [e.encode(x, y, direction, layout=layout) for e in self._encoders]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    @property
    def components(self) -> list[StateEncoder]:
        return list(self._encoders)
