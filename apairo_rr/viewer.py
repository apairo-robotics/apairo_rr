"""Rerun-based LiDAR viewer for apairo datasets."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Iterable

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .colormaps import ColumnColormap
from .pipeline import Pipeline

_CONFIGS_DIR = Path(__file__).parent / "configs"


def load_label_config(name: str) -> dict:
    """Load a built-in label config by name.

    Args:
        name: One of ``"rellis"``, ``"semantic_kitti"``, ``"goose"``.

    Returns:
        Dict with ``color_map`` (``{class_id: [R, G, B]}``) and
        ``semantic_map`` (``{class_id: label_str}``) keys, ready to pass
        to :func:`view` as ``label_cfg`` or ``label_cfgs``.

    Raises:
        FileNotFoundError: If *name* does not match any built-in config.
    """
    import yaml
    path = _CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No built-in config '{name}'. Available: {[p.stem for p in _CONFIGS_DIR.glob('*.yaml')]}")
    with path.open() as f:
        return yaml.safe_load(f)


def _normalize_color_map(raw: dict) -> dict[int, list[int]]:
    return {int(k): [int(c) for c in v] for k, v in raw.items()}


_default_colormap = ColumnColormap(2)


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
    pose_key:   str | None = None,
    pipelines:  list[Pipeline] | None = None,
    point_key:  str = "lidar",
    label_key:  str | None = "labels",
    frames:     Iterable[int] | None = None,
    application_id: str = "apairo_rr",
    spawn: bool = True,
    web: bool = False,
    web_port: int = 9090,
    grpc_port: int = 9876,
    point_radius: float | None = None,
    max_points: int | None = None,
) -> None:
    """Log an apairo dataset to the Rerun viewer.

    Opens the Rerun viewer with one :class:`~rerun.blueprint.Spatial3DView` per
    pipeline, arranged side-by-side.  If the dataset exposes ``sequence_ids``
    (all :class:`~apairo.core.ProfiledDataset` subclasses), a **Sequence**
    text panel is added above the 3D views and updates as you scrub the timeline.

    Args:
        dataset:        Any apairo dataset supporting ``dataset[idx]`` -> ``Sample``.
        label_cfg:      Single label config applied to all pipelines.
        label_cfgs:     Per-pipeline label configs; overrides ``label_cfg``.
                        Use :func:`load_label_config` to obtain built-in configs.
        poses:          List of 4×4 pose matrices (one per dataset frame).
                        Logs a static trajectory and a per-frame robot marker.
                        Mutually exclusive with *pose_key*.
        pose_key:       Sample key containing the per-frame 4×4 pose matrix.
                        Use this when the dataset was enriched with
                        :class:`~apairo_rr.Preprocess` (e.g. ``key="pose"``).
                        Mutually exclusive with *poses*.
        pipelines:      Ordered list of :class:`Pipeline` objects.
                        Defaults to a single ``Pipeline("Raw")``.
        point_key:      Sample key for the point cloud array (default ``"lidar"``).
        label_key:      Sample key for per-point semantic labels (default ``"labels"``).
                        Set to ``None`` to disable label colouring.
        frames:         Iterable of global frame indices to log.
                        Defaults to all frames in *dataset*.
        application_id: Rerun recording name shown in the viewer.
        spawn:          If ``True`` (default), launch the Rerun viewer process.
                        Set to ``False`` when saving to a ``.rrd`` file instead.
                        Ignored when ``web=True``.
        web:            If ``True``, serve a web viewer instead of spawning the
                        desktop app.  Any browser on the same local network can
                        open the URL that is printed to the console.
        web_port:       HTTP port for the web viewer (default 9090).
        grpc_port:      gRPC port for the data stream (default 9876).
        point_radius:   World-space radius for all logged point clouds.
                        Overrides the per-case defaults (``ui_points(2.0)`` for
                        labelled clouds, ``0.02`` m otherwise).
        max_points:     If set, randomly subsample each frame to at most this
                        many points before logging.  Reduces Rerun memory usage
                        at the cost of visual density (e.g. ``max_points=5000``).
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
    if web:
        import socket
        rr.init(application_id, spawn=False)
        rr.serve_grpc(grpc_port=grpc_port, cors_allow_origin=["*"])
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _s:
            try:
                _s.connect(("10.255.255.255", 1))
                _host_ip = _s.getsockname()[0]
            except OSError:
                _host_ip = "127.0.0.1"
        _grpc_url = f"rerun+http://{_host_ip}:{grpc_port}/proxy"
        rr.serve_web_viewer(connect_to=_grpc_url, web_port=web_port, open_browser=False)
        print(f"Web viewer -> http://{_host_ip}:{web_port}?url={_grpc_url}")
    else:
        rr.init(application_id, spawn=spawn)

    # Build frame->sequence mapping if the dataset supports it.
    seq_map: dict[int, str] = {}
    if hasattr(dataset, "sequence_ids"):
        for sid in dataset.sequence_ids:
            for i in dataset.sequence(sid)._indices:
                seq_map[i] = sid

    # Blueprint: one Spatial3DView per pipeline, side-by-side
    views = [
        rrb.Spatial3DView(origin=f"/{pipe.name}", name=pipe.name)
        for pipe in pipelines
    ]
    spatial = rrb.Horizontal(*views) if len(views) > 1 else views[0]
    if seq_map:
        seq_view = rrb.TextDocumentView(origin="/info/sequence", name="Sequence")
        layout = rrb.Vertical(seq_view, spatial, row_shares=[1, 12])
    else:
        layout = spatial
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

    # Apply view()-level defaults to pipelines that still carry the generic values.
    # Pipelines that set their own point_key / label_key explicitly are not affected.
    for pipe in pipelines:
        if pipe.point_key == "lidar":
            pipe.point_key = point_key
        if pipe.label_key == "labels":
            pipe.label_key = label_key

    # Pre-compute whether each colormap_fn accepts a sample argument.
    _cm_takes_sample = [
        len(inspect.signature(pipe.colormap_fn).parameters) >= 2
        if pipe.colormap_fn is not None else False
        for pipe in pipelines
    ]

    # ----------------------------------------------------------------- frames
    frame_indices = list(frames) if frames is not None else range(len(dataset))
    n = len(frame_indices)
    print(f"Logging {n} frames × {n_pipe} pipeline(s) …")
    _traj_positions: list[np.ndarray] = []  # incremental trajectory for pose_key

    for count, frame_idx in enumerate(frame_indices):
        rr.set_time("frame", sequence=frame_idx)

        sample = dataset[frame_idx]

        # Sequence label
        if seq_map:
            rr.log("/info/sequence", rr.TextDocument(seq_map.get(frame_idx, "?")))

        # Robot position along trajectory
        if poses is not None and frame_idx < len(poses):
            rr.log("/world/robot", rr.Points3D(
                [poses[frame_idx][:3, 3]],
                radii=0.4,
                colors=[[255, 200, 0]],
            ))
        elif pose_key is not None:
            p = sample.data.get(pose_key)
            if p is not None:
                pos = p[:3, 3]
                _traj_positions.append(pos)
                if len(_traj_positions) > 1:
                    rr.log("/world/trajectory", rr.LineStrips3D(
                        [np.array(_traj_positions)], colors=[[80, 140, 255]]
                    ))
                rr.log("/world/robot", rr.Points3D(
                    [pos], radii=0.4, colors=[[255, 200, 0]]
                ))

        # Per-pipeline point clouds — each pipeline resolves its own keys from sample
        for pipe, cfg, cm_takes_sample in zip(pipelines, resolved, _cm_takes_sample):
            pts, labels = pipe.run(sample, frame_idx=frame_idx)

            if max_points is not None and len(pts) > max_points:
                idx = np.random.choice(len(pts), max_points, replace=False)
                pts = pts[idx]
                if labels is not None:
                    labels = labels[idx]

            xyz = pts[:, :3].astype(np.float64)

            if labels is not None and cfg is not None:
                radius = point_radius if point_radius is not None else rr.Radius.ui_points(2.0)
                rr.log(f"/{pipe.name}/lidar", rr.Points3D(
                    xyz,
                    class_ids=labels.astype(np.uint16),
                    radii=radius,
                ))
            elif pipe.colormap_fn is not None:
                colors = pipe.colormap_fn(pts, sample) if cm_takes_sample else pipe.colormap_fn(pts)
                radius = point_radius if point_radius is not None else 0.02
                rr.log(f"/{pipe.name}/lidar", rr.Points3D(xyz, colors=colors, radii=radius))
            else:
                colors = _default_colormap(pts)
                radius = point_radius if point_radius is not None else 0.02
                rr.log(f"/{pipe.name}/lidar", rr.Points3D(xyz, colors=colors, radii=radius))

        if (count + 1) % 100 == 0:
            print(f"  {count + 1}/{n}")

    if web:
        print("Done — press Ctrl-C to stop the server.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        print("Done — open the Rerun viewer to explore.")
