"""Compare CSF and RANSAC ground height estimates on RELLIS-3D (in-place, no disk write).

Runs ground segmentation (CSF and/or RANSAC) and height estimation in memory,
then visualises the per-point height above ground side by side — no data written
to disk.

Use ``--priors`` to also show the traversability prior π = ½(1 − tanh(h − τ)).

Requires: apairo-rr, CSF  (pip install apairo-rr CSF)

Usage::

    python examples/view_ground_height.py ~/data/rellis
    python examples/view_ground_height.py ~/data/rellis --sequence 00000 --every 5
    python examples/view_ground_height.py ~/data/rellis --priors --tau 1.5
    python examples/view_ground_height.py ~/data/rellis --method ransac
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import apairo
import apairo_rr
from apairo_rr import Pipeline, KeyColormap, red_blue
from apairo_rr import Preprocess
from apairo_preprocess import (
    GroundSegmentationCSF,
    GroundSegmentationRANSAC,
    GroundHeightFromLabels,
)


from utils import get_generic_argparser_rellis

def main() -> None:
    p = get_generic_argparser_rellis()
    p.add_argument("--method",     choices=["csf", "ransac", "both"], default="both",
                   help="Ground segmentation method (default: both).")
    p.add_argument("--priors",     action="store_true",
                   help="Add pipelines showing π = ½(1−tanh(h−τ)).")
    p.add_argument("--tau",        type=float, default=2.0,
                   help="Height threshold τ (metres) for the prior (default: 2.0).")
    # CSF params
    p.add_argument("--cloth-resolution", type=float, default=0.5)
    p.add_argument("--class-threshold",  type=float, default=0.5)
    # RANSAC params
    p.add_argument("--ransac-iters",     type=int,   default=200)
    p.add_argument("--inlier-threshold", type=float, default=0.15)
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = apairo.Rellis3DDataset(root, keys=["voxelised"])
    n = len(ds)
    print(f"  {n} scans — sequences: {ds.sequence_ids}")

    if args.sequence is not None:
        if args.sequence not in ds.sequence_ids:
            raise SystemExit(
                f"Sequence '{args.sequence}' not found. Available: {ds.sequence_ids}"
            )
        seq_frames = list(ds.sequence(args.sequence)._indices)
    else:
        seq_frames = list(range(n))

    frames = seq_frames[args.idx::args.every]

    # Chain Preprocess layers — each is lazy, computed on first access per frame.
    methods = []
    if args.method in ("csf", "both"):
        ds = Preprocess(GroundSegmentationCSF(
            cloth_resolution=args.cloth_resolution,
            class_threshold=args.class_threshold,
        )).run(ds, seq_frames, key="ground_csf")
        ds = Preprocess(GroundHeightFromLabels("ground_csf")).run(
            ds, seq_frames, key="ground_height_csf"
        )
        methods.append(("CSF", "ground_height_csf"))

    if args.method in ("ransac", "both"):
        ds = Preprocess(GroundSegmentationRANSAC(
            n_iter=args.ransac_iters,
            inlier_threshold=args.inlier_threshold,
        )).run(ds, seq_frames, key="ground_ransac")
        ds = Preprocess(GroundHeightFromLabels("ground_ransac")).run(
            ds, seq_frames, key="ground_height_ransac"
        )
        methods.append(("RANSAC", "ground_height_ransac"))

    pipelines = []
    label_cfgs = []

    for label, key in methods:
        pipelines.append(Pipeline(
            f"Height {label} (m)",
            point_key="voxelised",
            label_key=None,
            colormap_fn=KeyColormap(key),
        ))
        label_cfgs.append(None)

    if args.priors:
        tau = args.tau
        print(f"  prior: π = ½(1 − tanh(h − {tau}))")
        for label, key in methods:
            pipelines.append(Pipeline(
                f"Prior π — {label}",
                point_key="voxelised",
                label_key=None,
                colormap_fn=KeyColormap(
                    key, gradient=lambda v: red_blue(0.5 * (1 - np.tanh(v - tau)))
                ),
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
