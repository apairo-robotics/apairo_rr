"""Visualise RemoveRobotPoints on RELLIS-3D with Rerun (in-place, no disk write).

Compares the original point cloud with the version where robot-body returns have
been removed using RemoveRobotPoints — no data written to disk.

The default box dimensions are a rough approximation of the Ouster OS1 sensor
mounting on the RELLIS robot.

Requires: apairo-rr  (pip install apairo-rr)

Usage::

    python examples/view_robot_filter.py ~/data/rellis
    python examples/view_robot_filter.py ~/data/rellis --shape cylinder --radius 0.5
    python examples/view_robot_filter.py ~/data/rellis --sequence 00000 --every 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import apairo
import apairo_rr
from apairo_rr import Pipeline
from apairo_preprocess import RemoveRobotPoints

_DEFAULT_ROOT = Path.home() / "data" / "rellis"

# Default box extents (metres, sensor frame)
_BOX_X = (-0.5, 0.5)
_BOX_Y = (-0.5, 0.5)
_BOX_Z = (-0.5, 0.2)


def range_filter(pts, labels, max_r: float = 50.0):
    mask = np.linalg.norm(pts[:, :3], axis=1) < max_r
    return pts[mask], labels[mask] if labels is not None else None


def make_robot_filter_step(shape: str, x_range, y_range, z_range, radius: float):
    """Create a pipeline callable backed by RemoveRobotPoints.

    RemoveRobotPoints.process() returns only filtered points; this wrapper also
    applies the same mask to labels so both arrays stay aligned.
    """
    # Instantiate to validate parameters
    proc = RemoveRobotPoints(
        shape=shape,
        x_range=x_range, y_range=y_range, z_range=z_range,
        radius=radius,
    )

    def step(pts, labels):
        xyz = pts[:, :3]
        if proc._shape == "box":
            inside = (
                (xyz[:, 0] >= proc._x_range[0]) & (xyz[:, 0] <= proc._x_range[1])
                & (xyz[:, 1] >= proc._y_range[0]) & (xyz[:, 1] <= proc._y_range[1])
                & (xyz[:, 2] >= proc._z_range[0]) & (xyz[:, 2] <= proc._z_range[1])
            )
        else:
            dist_xy = np.linalg.norm(xyz[:, :2], axis=1)
            inside = (
                (dist_xy <= proc._radius)
                & (xyz[:, 2] >= proc._z_range[0])
                & (xyz[:, 2] <= proc._z_range[1])
            )
        mask = ~inside
        return pts[mask], labels[mask] if labels is not None else None

    return step


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("root",     nargs="?", default=str(_DEFAULT_ROOT))
    p.add_argument("--shape",  choices=["box", "cylinder"], default="box")
    p.add_argument("--radius", type=float, default=0.5,
                   help="Cylinder radius in XY (metres, cylinder only).")
    p.add_argument("--sequence", default=None, help="Restrict to one sequence ID.")
    p.add_argument("--every",    type=int, default=1,  help="Log every Nth frame.")
    p.add_argument("--idx",      type=int, default=0,  help="First frame index.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["lidar", "labels"])
    n = len(ds)
    print(f"  {n} scans — sequences: {ds.sequence_ids}")
    if args.shape == "box":
        print(f"  shape=box  x={_BOX_X} y={_BOX_Y} z={_BOX_Z}")
    else:
        print(f"  shape=cylinder  radius={args.radius} z={_BOX_Z}")

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        frames = ds.sequence(args.sequence)._indices[args.idx::args.every]
    else:
        frames = range(args.idx, n, args.every)

    robot_filter_step = make_robot_filter_step(
        shape=args.shape,
        x_range=_BOX_X, y_range=_BOX_Y, z_range=_BOX_Z,
        radius=args.radius,
    )
    rellis_cfg = apairo_rr.load_label_config("rellis")

    apairo_rr.view(
        ds,
        label_cfgs=[rellis_cfg, rellis_cfg],
        frames=frames,
        pipelines=[
            Pipeline("Original",
                     [range_filter]),
            Pipeline(f"Robot points removed  ({args.shape})",
                     [range_filter, robot_filter_step]),
        ],
    )


if __name__ == "__main__":
    main()
