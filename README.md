# apairo-rr

[Rerun](https://rerun.io) visualisation layer for [apairo](../apairo) datasets.

Logs LiDAR point clouds, semantic labels, and robot trajectories to the Rerun viewer.  Supports multiple side-by-side pipelines, sequence-aware navigation, and any `apairo` dataset (RELLIS-3D, SemanticKITTI, GOOSE, TartanKitti…).

---

## Installation

```bash
pip install -e .
```

Requires `apairo` and `apairo_preprocess` (resolved from local paths in `pyproject.toml`).

---

## Quickstart

```python
import apairo
import apairo_rr
from apairo_rr import Pipeline

ds = apairo.Rellis3DDataset("/data/RELLIS", keys=["lidar", "labels"])

apairo_rr.view(
    ds,
    label_cfgs=[apairo_rr.load_label_config("rellis")],
    pipelines=[Pipeline("Semantic GT")],
)
```

---

## API

### `view(dataset, *, ...)`

Main entry point.  Logs a dataset to the Rerun viewer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset` | apairo dataset | Any dataset supporting `dataset[idx]` -> `Sample` |
| `label_cfg` | `dict \| None` | Label config applied to all pipelines |
| `label_cfgs` | `list[dict \| None]` | Per-pipeline label configs (overrides `label_cfg`) |
| `poses` | `list[np.ndarray]` | 4×4 pose matrices — logs a trajectory overlay |
| `pipelines` | `list[Pipeline]` | Processing pipelines, one 3D view each |
| `point_key` | `str` | Sample key for the point cloud (default `"lidar"`) |
| `label_key` | `str \| None` | Sample key for semantic labels (default `"labels"`) |
| `frames` | `Iterable[int] \| None` | Frame indices to log (default: all) |
| `spawn` | `bool` | Spawn the Rerun viewer automatically (default `True`) |

If the dataset exposes `sequence_ids` (all `ProfiledDataset` subclasses), a **Sequence** panel appears in the viewer and updates as you scrub the timeline.

### `Pipeline(name, steps=[])`

Named sequence of per-frame transforms.

```python
Pipeline("Raw")
Pipeline("Range filter", [range_filter])
Pipeline("Trav — labels", [range_filter, TraversabilityFromLabels()])
```

Each step must have signature `(pts, labels) -> (pts, labels)` or `(pts, labels, frame_idx=...) -> (pts, labels)`.  `apairo_preprocess` `FramePreprocessor` objects are accepted directly.

### `load_label_config(name)`

Load a built-in label config by name.

```python
cfg = apairo_rr.load_label_config("rellis")          # RELLIS-3D (20 classes)
cfg = apairo_rr.load_label_config("semantic_kitti")  # SemanticKITTI (28 classes)
cfg = apairo_rr.load_label_config("goose")           # GOOSE (64 classes)
```

Returns a dict with `color_map` and `semantic_map` keys, compatible with `view(label_cfgs=...)`.

---

## Examples

### RELLIS-3D — three-way traversability comparison

```bash
# All sequences
python examples/view_rellis_traversability.py --root ~/data/rellis

# Single sequence, every 5th frame
python examples/view_rellis_traversability.py --sequence 00000 --every 5

# Custom robot radius for trajectory-based traversability
python examples/view_rellis_traversability.py --sequence 00001 --radius 0.8
```

**CLI options**

| Flag | Default | Description |
|------|---------|-------------|
| `--root` | `~/data/rellis` | Dataset root directory |
| `--sequence` | *(all)* | Sequence ID to visualise (`00000`, `00001`, …) |
| `--every N` | `1` | Log every Nth frame |
| `--idx N` | `0` | Start at frame N (within the selected sequence) |
| `--radius R` | `1.0` | Robot radius for `TraversabilityFromTrajectory` |

---

## Sequence navigation

When `--sequence` is omitted, all sequences are concatenated on the same timeline.  Scrub the Rerun timeline to move between frames — the **Sequence** panel at the top of the viewer shows the current sequence ID.

To jump directly to a sequence, relaunch with `--sequence <id>`.  Available IDs are printed at startup:

```
  1832 scans — sequences: ['00000', '00001', '00002', '00003', '00004']
```

---

## Adding a custom dataset

Any object supporting `__getitem__(int) -> Sample` and `__len__` works with `view()`.  For sequence-aware navigation, the dataset must expose:

```python
dataset.sequence_ids          # list[str]
dataset.sequence(seq_id)      # SequenceView with ._indices (global frame indices)
```

All `apairo.ProfiledDataset` subclasses (RELLIS, SemanticKITTI, GOOSE) implement this automatically.
