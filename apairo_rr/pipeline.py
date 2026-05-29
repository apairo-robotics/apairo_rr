from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Pipeline:
    """Named sequence of per-frame transforms applied before logging.

    Each step is a callable with signature::

        step(pts: np.ndarray, labels: np.ndarray | None)
            -> tuple[np.ndarray, np.ndarray | None]

    Steps are applied in order.  apairo_preprocess ``FramePreprocessor``
    objects are also accepted directly as steps — their ``process(sample)``
    method is called automatically with the right keys.

    Examples::

        Pipeline("Raw")
        Pipeline("Range filter", [range_filter])
        Pipeline("Trav — labels", [range_filter, TraversabilityFromLabels()])
    """

    name: str
    steps: list[Callable] = field(default_factory=list)

    def run(
        self,
        pts: np.ndarray,
        labels: np.ndarray | None,
        frame_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        for step in self.steps:
            if hasattr(step, "process") and hasattr(step, "input_keys"):
                if hasattr(step, "_idx"):
                    step._idx = frame_idx
                data = {}
                for key in step.input_keys:
                    if key == "lidar":
                        data["lidar"] = pts
                    elif key == "labels" and labels is not None:
                        data["labels"] = labels
                sample = types.SimpleNamespace(data=data)
                result = step.process(sample)
                if result is not None:
                    labels = np.asarray(result, dtype=np.int64)
            else:
                try:
                    pts, labels = step(pts, labels, frame_idx=frame_idx)
                except TypeError:
                    pts, labels = step(pts, labels)
        return pts, labels
