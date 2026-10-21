"""Shared closed-track geometry. Distances refer to the sampled polyline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PlotPath


DEFAULT_TRACK = (
    Path(__file__).resolve().parent
    / "data/tracks/2024_reinvent_champ_ccw.npy"
)


def signed_area(points):
    return float(0.5 * np.sum(
        points[:-1, 0] * points[1:, 1]
        - points[1:, 0] * points[:-1, 1]
    ))


@dataclass
class Track:
    center: np.ndarray  # Includes the repeated closing point.
    outer: np.ndarray
    inner: np.ndarray
    original_length: float
    width: float

    def __post_init__(self):
        self.vectors = np.diff(self.center, axis=0)
        self.ds = np.linalg.norm(self.vectors, axis=1)
        if np.any(self.ds < 1e-9):
            raise ValueError("Track contains zero-length segments.")
        self.s = np.concatenate(([0.0], np.cumsum(self.ds)))
        self.length = float(self.s[-1])
        self.outer_path = PlotPath(self.outer, closed=True)
        self.inner_path = PlotPath(self.inner, closed=True)

    def position(self, distance):
        distance = distance % self.length
        return np.array([
            np.interp(distance, self.s, self.center[:, 0]),
            np.interp(distance, self.s, self.center[:, 1]),
        ])

    def project(self, x, y):
        point = np.array([x, y])
        offset = point - self.center[:-1]
        fraction = np.clip(
            np.sum(offset * self.vectors, axis=1) / self.ds**2,
            0.0, 1.0,
        )
        projected = self.center[:-1] + fraction[:, None] * self.vectors
        distances = np.linalg.norm(projected - point, axis=1)
        index = int(np.argmin(distances))
        distance = self.s[index] + fraction[index] * self.ds[index]
        return float(distance % self.length), float(distances[index])

    def contains(self, x, y):
        # Rear-axle midpoint only; not a vehicle-footprint test.
        return bool(
            self.outer_path.contains_point((x, y))
            and not self.inner_path.contains_point((x, y))
        )


def load_track(path=DEFAULT_TRACK, spacing=0.05):
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Sample spacing must be positive and finite.")
    raw = np.load(path, allow_pickle=False)
    if raw.ndim != 2 or raw.shape[1] != 6 or len(raw) < 4:
        raise ValueError("Expected at least four rows with six columns.")
    if not np.isfinite(raw).all():
        raise ValueError("Track contains invalid coordinates.")
    if np.allclose(raw[0, :2], raw[-1, :2]):
        raw = raw.copy()
        raw[-1] = raw[0]
    else:
        raw = np.vstack((raw, raw[0]))
    raw_ds = np.linalg.norm(np.diff(raw[:, :2], axis=0), axis=1)
    if np.any(raw_ds < 1e-9):
        raise ValueError("Source track contains zero-length segments.")
    if signed_area(raw[:, :2]) <= 0:
        raise ValueError("Expected a counterclockwise track.")
    raw_s = np.concatenate(([0.0], np.cumsum(raw_ds)))
    count = max(8, int(np.ceil(raw_s[-1] / spacing)))
    sample_s = np.linspace(0.0, raw_s[-1], count, endpoint=False)
    center = np.column_stack([
        np.interp(sample_s, raw_s, raw[:, column])
        for column in range(2)
    ])
    center = np.vstack((center, center[0]))
    # Preserve original boundary polygons instead of resampling their corners.
    boundary_a, boundary_b = raw[:, 2:4], raw[:, 4:6]
    if abs(signed_area(boundary_a)) > abs(signed_area(boundary_b)):
        outer, inner = boundary_a, boundary_b
    else:
        outer, inner = boundary_b, boundary_a
    return Track(
        center=center, outer=outer, inner=inner,
        original_length=float(raw_s[-1]),
        width=float(np.median(np.linalg.norm(boundary_a - boundary_b, axis=1))),
    )
