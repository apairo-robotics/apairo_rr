"""RELLIS-3D — three-way traversability comparison with Rerun.

  Pipeline 1 — Semantic GT        : RELLIS semantic labels (20 classes)
  Pipeline 2 — Trav — trajectory  : TraversabilityFromTrajectory on-the-fly
  Pipeline 3 — Trav — labels      : TraversabilityFromLabels on-the-fly

Usage:
    python examples/view_rellis_traversability.py
    python examples/view_rellis_traversability.py --root ~/data/rellis --every 5
    python examples/view_rellis_traversability.py --sequence 00000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import apairo
import apairo_rr
from apairo_rr import Pipeline
from apairo_preprocess import TraversabilityFromLabels, TraversabilityFromTrajectory

_DEFAULT_ROOT = Path.home() / "data" / "rellis"


TRAVERSABILITY_TRAJECTORY_CFG = {
    "robot_radius": 1.0,
    "height_min":  -5.0,
    "height_max":  0.5,
    "forward_window": None,
    "sequence_gap":  5.0,
}


def range_filter(pts, labels, max_r=50.0):
    mask = np.linalg.norm(pts[:, :3], axis=1) < max_r
    return pts[mask], labels[mask] if labels is not None else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root",     default=str(_DEFAULT_ROOT))
    p.add_argument("--sequence", default=None, help="Sequence ID to visualise (default: all)")
    p.add_argument("--radius",   type=float, default=None)
    p.add_argument("--every",    type=int,   default=1, help="Log every Nth frame")
    p.add_argument("--idx",      type=int,   default=0, help="First frame index")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["lidar", "labels", "poses"])
    print(f"  {len(ds)} scans — sequences: {ds.sequence_ids}")

    n = len(ds)
    poses_4x4 = np.eye(4)[None].repeat(n, axis=0)
    poses_4x4[:, :3, :] = np.stack([ds[i].data["poses"] for i in range(n)])

    cfg_trav = {
        "color_map":    {0: [200, 50, 50], 1: [50, 200, 80]},
        "semantic_map": {0: "non-traversable", 1: "traversable"},
    }

    cfg_trav_tj = TRAVERSABILITY_TRAJECTORY_CFG
    cfg_trav_tj["robot_radius"] = cfg_trav_tj["robot_radius"] if args.radius is None else args.radius

    trav_traj_proc = TraversabilityFromTrajectory(poses_4x4, **cfg_trav_tj)
    trav_labs_proc = TraversabilityFromLabels()

    # Resolve frame indices — optionally restricted to a single sequence.
    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. "
                f"Available: {ds.sequence_ids}"
            )
        seq = ds.sequence(args.sequence)
        frames = seq._indices[args.idx::args.every]
        print(f"[params] sequence={args.sequence} ({len(seq)} frames, showing {len(frames)})")
    else:
        frames = range(args.idx, n, args.every)
        print(f"[params] all sequences ({len(frames)} frames)")

    print(f"[params] root={root}, every={args.every}, idx={args.idx}, radius={args.radius}")
    print(f"[params] TraversabilityFromTrajectory cfg: {cfg_trav_tj}")

    apairo_rr.view(
        ds,
        label_cfgs = [apairo_rr.load_label_config("rellis"), cfg_trav, cfg_trav],
        poses      = list(poses_4x4),
        frames     = frames,
        pipelines  = [
            Pipeline("Semantic GT",       [range_filter]),
            Pipeline("Trav — trajectory", [range_filter, trav_traj_proc]),
            Pipeline("Trav — labels",     [range_filter, trav_labs_proc]),
        ],
    )


if __name__ == "__main__":
    main()
