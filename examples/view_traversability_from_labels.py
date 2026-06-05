"""Visualise TraversabilityFromLabels on RELLIS-3D with Rerun (in-place, no disk write).

Compares the original semantic labels with the binary traversable/non-traversable
mapping produced by TraversabilityFromLabels — no data written to disk.

Requires: apairo-rr  (pip install apairo-rr)

Usage::

    python examples/view_traversability_from_labels.py ~/data/rellis
    python examples/view_traversability_from_labels.py ~/data/rellis --sequence 00000 --every 5
    python examples/view_traversability_from_labels.py ~/data/rellis --ids 1 3 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import apairo
import apairo_rr
from apairo_rr import Pipeline
from apairo_preprocess import TraversabilityFromLabels

_DEFAULT_ROOT = Path.home() / "data" / "rellis"


def range_filter(pts, labels, max_r: float = 50.0):
    mask = np.linalg.norm(pts[:, :3], axis=1) < max_r
    return pts[mask], labels[mask] if labels is not None else None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("root", nargs="?", default=str(_DEFAULT_ROOT))
    p.add_argument("--sequence", default=None, help="Restrict to one sequence ID.")
    p.add_argument("--every",    type=int, default=1,  help="Log every Nth frame.")
    p.add_argument("--idx",      type=int, default=0,  help="First frame index.")
    p.add_argument("--ids", nargs="+", type=int, default=None,
                   help="Traversable class IDs. Defaults to RELLIS built-in set.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["lidar", "labels"])
    n = len(ds)
    print(f"  {n} scans — sequences: {ds.sequence_ids}")

    traversable_ids = frozenset(args.ids) if args.ids else None
    trav_proc = TraversabilityFromLabels(traversable_ids=traversable_ids)
    print(f"  traversable IDs: {trav_proc._trav_ids}")

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        frames = ds.sequence(args.sequence)._indices[args.idx::args.every]
    else:
        frames = range(args.idx, n, args.every)

    cfg_trav = {
        "color_map":    {0: [200, 50, 50], 1: [50, 200, 80]},
        "semantic_map": {0: "non-traversable", 1: "traversable"},
    }

    apairo_rr.view(
        ds,
        label_cfgs=[apairo_rr.load_label_config("rellis"), cfg_trav],
        frames=frames,
        pipelines=[
            Pipeline("Semantic GT",              [range_filter]),
            Pipeline("Traversability — labels",  [range_filter, trav_proc]),
        ],
    )


if __name__ == "__main__":
    main()
