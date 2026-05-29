"""Rerun-based LiDAR viewer for apairo datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .pipeline import Pipeline

_CONFIGS_DIR = Path(__file__).parent / "configs"


def load_label_config(name: str) -> dict:
    """Load a built-in label config by name (``"rellis"``, ``"goose"``, ``"semantic_kitti"``)."""
    import yaml
    path = _CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No built-in config '{name}'. Available: {[p.stem for p in _CONFIGS_DIR.glob('*.yaml')]}")
    with path.open() as f:
        return yaml.safe_load(f)


def _normalize_color_map(raw: dict) -> dict[int, list[int]]:
    return {int(k): [int(c) for c in v] for k, v in raw.items()}


def _height_colors(z: np.ndarray) -> np.ndarray:
    lo, hi = z.min(), z.max()
    t = (z - lo) / max(hi - lo, 1e-6)
    colors = np.zeros((len(z), 3), dtype=np.uint8)
    colors[:, 0] = (255 * (1 - t)).astype(np.uint8)
    colors[:, 2] = (255 * t).astype(np.uint8)
    return colors


def _annotation_context(cfg: dict) -> list[rr.ClassDescription]:
    color_map   = _normalize_color_map(cfg["color_map"])
    semantic_map = {int(k): v for k, v in cfg.get("semantic_map", {}).items()}
    return [
        rr.ClassDescription(info=rr.AnnotationInfo(
            id    = cls_id,
            color = tuple(color_map.get(cls_id, [128, 128, 128])),
            label = semantic_map.get(cls_id, str(cls_id)),
        ))
        for cls_id in sorted(semantic_map.keys())
    ]


def view(
    dataset,
    *,
    label_cfg:  dict | None = None,
    label_cfgs: list[dict | None] | None = None,
    poses:      list[np.ndarray] | None = None,
    pipelines:  list[Pipeline] | None = None,
    point_key:  str = "lidar",
    label_key:  str | None = "labels",
    frames:     Iterable[int] | None = None,
    application_id: str = "apairo_rr",
    spawn: bool = True,
) -> None:
    """Log an apairo dataset to the Rerun viewer.

    Args:
        dataset:        Any apairo dataset supporting ``dataset[idx]`` → ``Sample``.
        label_cfg:      Single label config dict applied to all pipelines.
        label_cfgs:     Per-pipeline label configs (overrides ``label_cfg``).
        poses:          List of 4×4 pose matrices for the trajectory overlay.
        pipelines:      List of :class:`Pipeline` objects.  Defaults to one raw pipeline.
        point_key:      Key for the point cloud in each sample.
        label_key:      Key for semantic labels in each sample.
        frames:         Subset of frame indices to log.  Defaults to all frames.
        application_id: Rerun application name.
        spawn:          Whether to spawn the Rerun viewer automatically.
    """
    if pipelines is None:
        pipelines = [Pipeline("Raw")]

    n_pipe = len(pipelines)

    # Resolve per-pipeline label configs
    if label_cfgs is not None:
        resolved: list[dict | None] = list(label_cfgs)
        while len(resolved) < n_pipe:
            resolved.append(label_cfg)
    else:
        resolved = [label_cfg] * n_pipe

    # ------------------------------------------------------------------ init
    rr.init(application_id, spawn=spawn)

    # Blueprint: one Spatial3DView per pipeline, side-by-side
    views = [
        rrb.Spatial3DView(origin=f"/{pipe.name}", name=pipe.name)
        for pipe in pipelines
    ]
    layout = rrb.Horizontal(*views) if len(views) > 1 else views[0]
    rr.send_blueprint(rrb.Blueprint(layout, auto_layout=False, auto_views=False))

    # Annotation contexts (static — logged once, apply to all frames)
    for pipe, cfg in zip(pipelines, resolved):
        if cfg is not None:
            rr.log(
                f"/{pipe.name}",
                rr.AnnotationContext(_annotation_context(cfg)),
                static=True,
            )

    # Full trajectory (static)
    if poses is not None:
        positions = np.array([p[:3, 3] for p in poses])
        rr.log(
            "/world/trajectory",
            rr.LineStrips3D([positions], colors=[[80, 140, 255]]),
            static=True,
        )

    # ----------------------------------------------------------------- frames
    frame_indices = list(frames) if frames is not None else range(len(dataset))
    n = len(frame_indices)
    print(f"Logging {n} frames × {n_pipe} pipeline(s) …")

    for count, frame_idx in enumerate(frame_indices):
        rr.set_time("frame", sequence=frame_idx)

        sample  = dataset[frame_idx]
        pts_raw = np.asarray(sample.data[point_key], dtype=np.float32)
        lbl_raw: np.ndarray | None = None
        if label_key and label_key in sample.data:
            lbl_raw = np.asarray(sample.data[label_key], dtype=np.int64)

        # Robot position along trajectory
        if poses is not None and frame_idx < len(poses):
            rr.log("/world/robot", rr.Points3D(
                [poses[frame_idx][:3, 3]],
                radii=0.4,
                colors=[[255, 200, 0]],
            ))

        # Per-pipeline point clouds
        for pipe, cfg in zip(pipelines, resolved):
            pts, labels = pipe.run(
                pts_raw.copy(),
                lbl_raw.copy() if lbl_raw is not None else None,
                frame_idx=frame_idx,
            )
            xyz = pts[:, :3].astype(np.float64)

            if labels is not None and cfg is not None:
                rr.log(f"/{pipe.name}/lidar", rr.Points3D(
                    xyz,
                    class_ids=labels.astype(np.uint16),
                    radii=rr.Radius.ui_points(2.0),
                ))
            else:
                colors = _height_colors(pts[:, 2])
                rr.log(f"/{pipe.name}/lidar", rr.Points3D(xyz, colors=colors, radii=0.02))

        if (count + 1) % 100 == 0:
            print(f"  {count + 1}/{n}")

    print("Done — open the Rerun viewer to explore.")
