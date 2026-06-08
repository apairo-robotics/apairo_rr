"""Colormap abstractions and built-in implementations for Pipeline.colormap_fn."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

# A gradient maps a 1-D scalar array to (N, 3) uint8 RGB.
# Use as building blocks inside custom Colormap subclasses.
Gradient = Callable[[np.ndarray], np.ndarray]


class Colormap(ABC):
    """Interface for ``Pipeline.colormap_fn``.

    Subclass this to define any coloring logic from a full point array::

        class RangeColormap(Colormap):
            def __call__(self, pts: np.ndarray) -> np.ndarray:
                r = np.linalg.norm(pts[:, :3], axis=1)
                return red_blue(r)

        Pipeline("Range", colormap_fn=RangeColormap())
    """

    @abstractmethod
    def __call__(self, pts: np.ndarray) -> np.ndarray:
        """Map an (N, D) point array to an (N, 3) uint8 RGB array."""


# ---------------------------------------------------------------------------
# Gradient primitives  —  Gradient = (N,) scalar → (N, 3) uint8
# ---------------------------------------------------------------------------

def red_blue(v: np.ndarray) -> np.ndarray:
    """Red (low) → blue (high) gradient, normalised over *v*."""
    lo, hi = float(v.min()), float(v.max())
    t = (v - lo) / max(hi - lo, 1e-6)
    rgb = np.zeros((len(v), 3), dtype=np.uint8)
    rgb[:, 0] = (255 * (1 - t)).astype(np.uint8)
    rgb[:, 2] = (255 * t).astype(np.uint8)
    return rgb


def green_red(v: np.ndarray) -> np.ndarray:
    """Green (low) → red (high) gradient, normalised over *v*."""
    lo, hi = float(v.min()), float(v.max())
    t = (v - lo) / max(hi - lo, 1e-6)
    rgb = np.zeros((len(v), 3), dtype=np.uint8)
    rgb[:, 1] = (255 * (1 - t)).astype(np.uint8)
    rgb[:, 0] = (255 * t).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# Built-in Colormap implementations
# ---------------------------------------------------------------------------

class KeyColormap(Colormap):
    """Colour by a per-point scalar stored in a dedicated sample key.

    Unlike :class:`ColumnColormap`, which reads a column from the point array,
    ``KeyColormap`` reads a separate channel from the sample directly — no
    embedding into the point array is needed.

    Args:
        key:      Sample key containing the per-point scalar array
                  (e.g. ``"ground_height_csf"``).
        gradient: :data:`Gradient` applied to the scalar.  Defaults to
                  :func:`red_blue`.

    Example::

        Pipeline("Height CSF", point_key="voxelised", label_key=None,
                 colormap_fn=KeyColormap("ground_height_csf"))
    """

    def __init__(self, key: str, gradient: Gradient = red_blue):
        self.key = key
        self.gradient = gradient

    def __call__(self, pts: np.ndarray, sample=None) -> np.ndarray:
        if sample is not None and self.key in sample.data:
            v = np.asarray(sample.data[self.key], dtype=np.float32).ravel()
        else:
            v = pts[:, 0]
        return self.gradient(v)


class ColumnColormap(Colormap):
    """Colour each point by a single scalar column, via a :data:`Gradient`.

    Args:
        col:      Column index to extract from the (N, D) pts array.
        gradient: A :data:`Gradient` function applied to the extracted column.
                  Defaults to :func:`red_blue`.

    Examples::

        Pipeline("Height Z",      colormap_fn=ColumnColormap(2))
        Pipeline("Intensity",     colormap_fn=ColumnColormap(3, gradient=green_red))
        Pipeline("Ground height", colormap_fn=ColumnColormap(4))
    """

    def __init__(self, col: int, gradient: Gradient = red_blue):
        self.col = col
        self.gradient = gradient

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        return self.gradient(pts[:, self.col])
