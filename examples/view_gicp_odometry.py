"""Visualise GICP odometry on RELLIS-3D with Rerun (in-place, no disk write).

Computes the GICP (Generalised ICP via Open3D) trajectory in memory and streams
the estimated poses to Rerun alongside the semantic labels — no data written to disk.

Compare the estimated trajectory (orange marker + blue line) with the map
structure to verify that scan-to-scan registration is geometrically consistent.

Requires: apairo-rr, open3d  (pip install apairo-rr open3d)

Usage::

    python examples/view_gicp_odometry.py ~/data/rellis --sequence 00000
    python examples/view_gicp_odometry.py ~/data/rellis --sequence 00000 --voxel-size 0.3
    python examples/view_gicp_odometry.py ~/data/rellis --sequence 00000 --every 5
"""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import numpy as np

import apairo
import apairo_rr
from apairo_rr import Pipeline
from apairo_preprocess import GICPOdometry

_DEFAULT_ROOT = Path.home() / "data" / "rellis"


def range_filter(pts, labels, max_r: float = 50.0):
    mask = np.linalg.norm(pts[:, :3], axis=1) < max_r
    return pts[mask], labels[mask] if labels is not None else None


def compute_gicp_poses(ds, frame_list, voxel_size, max_range, max_corr):
    """Run GICPOdometry on the given frames and return poses without writing to disk."""
    print(f"Computing GICP odometry for {len(frame_list)} frames (in memory) …")
    proc = GICPOdometry(
        voxel_size=voxel_size,
        max_range=max_range,
        max_corr=max_corr,
    )
    poses = []
    for k, fi in enumerate(frame_list):
        s = ds[fi]
        ns = types.SimpleNamespace(data={"lidar": np.asarray(s.data["lidar"])})
        poses.append(proc.process(ns))   # (4, 4) float64
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(frame_list)}")
    print("  Done.")
    return poses


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("root",        nargs="?", default=str(_DEFAULT_ROOT))
    p.add_argument("--sequence",  default=None, help="Restrict to one sequence ID (recommended).")
    p.add_argument("--voxel-size", type=float, default=0.3,
                   help="Down-sampling voxel size for ICP (default: 0.3 m).")
    p.add_argument("--max-range",  type=float, default=50.0,
                   help="Maximum point range kept before registration (default: 50.0 m).")
    p.add_argument("--max-corr",   type=float, default=1.0,
                   help="Maximum ICP correspondence distance (default: 1.0 m).")
    p.add_argument("--every",  type=int, default=1, help="Log every Nth frame.")
    p.add_argument("--idx",    type=int, default=0, help="First frame index.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["lidar", "labels"])
    n = len(ds)
    print(f"  {n} scans — sequences: {ds.sequence_ids}")
    print(f"  voxel_size={args.voxel_size} m, max_range={args.max_range} m, max_corr={args.max_corr} m")

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        seq_frames = list(ds.sequence(args.sequence)._indices)
    else:
        seq_frames = list(range(n))

    # Compute poses for the full (sub)sequence for a complete trajectory
    poses = compute_gicp_poses(
        ds, seq_frames,
        voxel_size=args.voxel_size,
        max_range=args.max_range,
        max_corr=args.max_corr,
    )

    # Build full-length poses list (identity for frames outside this sequence)
    all_poses = [np.eye(4)] * n
    for fi, pose in zip(seq_frames, poses):
        all_poses[fi] = pose

    view_frames = seq_frames[args.idx::args.every]

    apairo_rr.view(
        ds,
        label_cfgs=[apairo_rr.load_label_config("rellis")],
        poses=all_poses,
        frames=view_frames,
        pipelines=[
            Pipeline("GICP — semantic labels", [range_filter]),
        ],
    )


if __name__ == "__main__":
    main()
