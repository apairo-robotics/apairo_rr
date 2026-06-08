"""Compare CSF and RANSAC ground height estimates on RELLIS-3D.

Both channels must be precomputed on disk before running this script.
Each pipeline colours points by height above ground (red = ground-level,
blue = elevated).  Use ``--priors`` to also show the derived traversability
prior  π = ½(1 − tanh(h − τ))  for each method.

Requires: apairo-rr, apairo_preprocess
Requires precomputed channels: ground_height_csf, ground_height_ransac

Usage::

    python examples/view_ground_height.py ~/data/rellis
    python examples/view_ground_height.py ~/data/rellis --sequence 00000 --every 5
    python examples/view_ground_height.py ~/data/rellis --priors --tau 1.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import apairo
import apairo_rr
from apairo.core.sample import Sample
from apairo_rr import Pipeline

_DEFAULT_ROOT = Path.home() / "data" / "rellis"


# ---------------------------------------------------------------------------
# Coloring
# ---------------------------------------------------------------------------

def _red_blue(v: np.ndarray) -> np.ndarray:
    """Normalise *v* to [0, 1] and map to a red (low) → blue (high) gradient."""
    lo, hi = float(v.min()), float(v.max())
    t = (v - lo) / max(hi - lo, 1e-6)
    rgb = np.zeros((len(v), 3), dtype=np.uint8)
    rgb[:, 0] = (255 * (1 - t)).astype(np.uint8)
    rgb[:, 2] = (255 * t).astype(np.uint8)
    return rgb


def _height_colormap(col: int):
    """colormap_fn: colour by raw height value (metres) in column *col*."""
    def fn(pts: np.ndarray) -> np.ndarray:
        return _red_blue(pts[:, col])
    return fn


def _prior_colormap(col: int, tau: float):
    """colormap_fn: colour by height prior  π = ½(1 − tanh(h − τ)).

    Red = high prior (likely traversable), blue = low prior.
    """
    def fn(pts: np.ndarray) -> np.ndarray:
        pi = 0.5 * (1.0 - np.tanh(pts[:, col] - tau))
        return _red_blue(pi)
    return fn


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def _embed_scalars(keys: list[str]):
    """sample_transform: append scalar channels as extra columns in voxelised."""
    def _fn(sample: Sample) -> Sample:
        pts = np.asarray(sample.data["voxelised"], dtype=np.float32)
        extras = [
            np.asarray(sample.data[k], dtype=np.float32)[:, None]
            for k in keys
        ]
        sample.data["voxelised"] = np.hstack([pts, *extras])
        return sample
    return _fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("root",       nargs="?", default=str(_DEFAULT_ROOT))
    p.add_argument("--sequence", default=None, help="Restrict to one sequence ID.")
    p.add_argument("--every",    type=int,   default=1,   help="Log every Nth frame.")
    p.add_argument("--idx",      type=int,   default=0,   help="First frame index.")
    p.add_argument("--priors",   action="store_true",
                   help="Add pipelines showing π = ½(1−tanh(h−τ)) for each method.")
    p.add_argument("--tau",      type=float, default=2.0,
                   help="Height threshold τ (metres) for the prior (default: 2.0).")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    # Discover which height channels are available
    scalar_keys = []
    for key in ("ground_height_csf", "ground_height_ransac"):
        try:
            apairo.Rellis3DDataset(root, keys=[key])
            scalar_keys.append(key)
        except (KeyError, FileNotFoundError):
            print(f"  [skip] {key} not found — run preprocess first.")

    if not scalar_keys:
        raise SystemExit(
            "No ground height channels found.\n"
            "Run: python -m scripts.preprocess.preprocess_wipunce "
            "--config config/rellis_preprocess.yaml"
        )

    # Load voxelised + available height channels.
    # sample_transform embeds the scalars as extra columns in voxelised at
    # __getitem__ time — no disk writes (apairo synchronous transform API).
    ds = apairo.Rellis3DDataset(root, keys=["voxelised"] + scalar_keys)
    ds.sample_transform(_embed_scalars(scalar_keys))

    n = len(ds)
    print(f"  {n} scans — sequences: {ds.sequence_ids}")
    print(f"  channels: {scalar_keys}")

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        frames = ds.sequence(args.sequence)._indices[args.idx::args.every]
    else:
        frames = range(args.idx, n, args.every)

    # One pipeline per channel (raw height), plus prior pipelines if requested.
    # colormap_fn reads the scalar column directly — XYZ positions are not distorted.
    pipelines = []
    label_cfgs = []

    labels = {"ground_height_csf": "CSF", "ground_height_ransac": "RANSAC"}

    for i, key in enumerate(scalar_keys):
        col = 4 + i
        pipelines.append(Pipeline(
            f"Height {labels[key]} (m)",
            point_key="voxelised",
            label_key=None,
            colormap_fn=_height_colormap(col),
        ))
        label_cfgs.append(None)

    if args.priors:
        print(f"  prior: π = ½(1 − tanh(h − {args.tau}))")
        for i, key in enumerate(scalar_keys):
            col = 4 + i
            pipelines.append(Pipeline(
                f"Prior π — {labels[key]}",
                point_key="voxelised",
                label_key=None,
                colormap_fn=_prior_colormap(col, args.tau),
            ))
            label_cfgs.append(None)

    apairo_rr.view(
        ds,
        label_cfgs=label_cfgs,
        pipelines=pipelines,
        frames=frames,
        application_id="ground_height",
    )


if __name__ == "__main__":
    main()
