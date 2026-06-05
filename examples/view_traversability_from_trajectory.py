"""Visualise TraversabilityFromTrajectory on RELLIS-3D with Rerun (in-place, no disk write).

Loads GT poses from RELLIS and computes per-point traversability on-the-fly using
TraversabilityFromTrajectory — no data written to disk.

Usage::

    python examples/view_traversability_from_trajectory.py
    python examples/view_traversability_from_trajectory.py --root ~/data/rellis --sequence 00000
    python examples/view_traversability_from_trajectory.py --sequence 00000 \\
        --robot-radius 1.5 --every 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import apairo
import apairo_rr
from apairo_rr import Pipeline
from apairo_preprocess import TraversabilityFromTrajectory

_DEFAULT_ROOT = Path.home() / "data" / "rellis"


def range_filter(pts, labels, max_r: float = 50.0):
    mask = np.linalg.norm(pts[:, :3], axis=1) < max_r
    return pts[mask], labels[mask] if labels is not None else None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root",          default=str(_DEFAULT_ROOT))
    p.add_argument("--sequence",      default=None,  help="Restrict to one sequence ID.")
    p.add_argument("--every",         type=int,   default=1,    help="Log every Nth frame.")
    p.add_argument("--idx",           type=int,   default=0,    help="First frame index.")
    p.add_argument("--robot-radius",  type=float, default=1.0,  help="Robot XY half-width (m).")
    p.add_argument("--height-min",    type=float, default=-5.0, help="Min point height (m).")
    p.add_argument("--height-max",    type=float, default=0.5,  help="Max point height (m).")
    p.add_argument("--forward-window", type=int,  default=None, help="Look-ahead pose count.")
    p.add_argument("--sequence-gap",  type=float, default=5.0,  help="Boundary gap (m).")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["lidar", "labels", "poses"])
    n = len(ds)
    print(f"  {n} scans — sequences: {ds.sequence_ids}")

    # Build the full (N, 4, 4) pose array required by TraversabilityFromTrajectory
    poses_4x4 = np.eye(4)[None].repeat(n, axis=0)
    poses_4x4[:, :3, :] = np.stack([ds[i].data["poses"] for i in range(n)])

    trav_traj_proc = TraversabilityFromTrajectory(
        poses_4x4,
        robot_radius=args.robot_radius,
        height_min=args.height_min,
        height_max=args.height_max,
        forward_window=args.forward_window,
        sequence_gap=args.sequence_gap,
    )

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        seq = ds.sequence(args.sequence)
        frames = seq._indices[args.idx::args.every]
    else:
        frames = range(args.idx, n, args.every)

    cfg_trav = {
        "color_map":    {0: [200, 50, 50], 1: [50, 200, 80]},
        "semantic_map": {0: "non-traversable", 1: "traversable"},
    }

    print(f"[params] robot_radius={args.robot_radius}, height=[{args.height_min}, {args.height_max}]")

    apairo_rr.view(
        ds,
        label_cfgs=[apairo_rr.load_label_config("rellis"), cfg_trav],
        poses=list(poses_4x4),
        frames=frames,
        pipelines=[
            Pipeline("Semantic GT",                  [range_filter]),
            Pipeline("Traversability — trajectory",  [range_filter, trav_traj_proc]),
        ],
    )


if __name__ == "__main__":
    main()
