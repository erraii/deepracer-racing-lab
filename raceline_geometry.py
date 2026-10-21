"""Signed corridor clearance and exact checks on the exported polyline."""

import numpy as np


def nearest_segments(points, starts, vectors):
    offset = points[:, None, :] - starts[None, :, :]
    squared = np.sum(vectors**2, axis=1)
    fraction = np.clip(np.sum(offset * vectors[None], axis=2) / squared, 0, 1)
    projected = starts[None] + fraction[:, :, None] * vectors[None]
    difference = points[:, None] - projected
    distances_squared = np.sum(difference**2, axis=2)
    nearest = np.argmin(distances_squared, axis=1)
    rows = np.arange(len(points))
    return np.sqrt(distances_squared[rows, nearest]), difference[rows, nearest]


class Corridor:
    def __init__(self, road):
        self.road = road
        self.starts = np.vstack((road.outer[:-1], road.inner[:-1]))
        self.vectors = np.vstack((np.diff(road.outer, axis=0), np.diff(road.inner, axis=0)))
        nonzero = np.linalg.norm(self.vectors, axis=1) > 1e-10
        self.starts = self.starts[nonzero]
        self.vectors = self.vectors[nonzero]

    def signed_clearance(self, points):
        distances, differences = nearest_segments(points, self.starts, self.vectors)
        inside = (self.road.outer_path.contains_points(points)
                  & ~self.road.inner_path.contains_points(points))
        sign = np.where(inside, 1.0, -1.0)
        gradient = sign[:, None] * differences / np.maximum(distances[:, None], 1e-12)
        return sign * distances, gradient


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def intersection_matrix(starts_a, vectors_a, starts_b, vectors_b):
    a, r = starts_a[:, None], vectors_a[:, None]
    b, s = starts_b[None], vectors_b[None]
    delta = b - a
    denominator = cross(r, s)
    parallel = np.abs(denominator) <= 1e-12
    safe = np.where(parallel, 1.0, denominator)
    t, u = cross(delta, s) / safe, cross(delta, r) / safe
    crossing = (~parallel & (t >= -1e-10) & (t <= 1 + 1e-10)
                & (u >= -1e-10) & (u <= 1 + 1e-10))
    collinear = parallel & (np.abs(cross(delta, r)) <= 1e-12)
    a_end, b_end = a + r, b + s
    overlaps = np.all(
        np.maximum(np.minimum(a, a_end), np.minimum(b, b_end))
        <= np.minimum(np.maximum(a, a_end), np.maximum(b, b_end)) + 1e-10,
        axis=2,
    )
    return crossing | (collinear & overlaps)


def validate_reference(road, center, margin):
    """Validate full straight segments, not just sampled point membership."""
    center = np.asarray(center, dtype=float)
    if (center.ndim != 2 or center.shape[1] != 2 or len(center) < 9
            or not np.isfinite(center).all()):
        raise ValueError("Invalid reference coordinates.")
    if not np.allclose(center[0], center[-1], atol=1e-8, rtol=0):
        raise ValueError("Reference must be explicitly closed.")
    vectors = np.diff(center, axis=0)
    if np.any(np.linalg.norm(vectors, axis=1) < 1e-9):
        raise ValueError("Reference contains zero-length segments.")
    area = 0.5 * np.sum(cross(center[:-1], center[1:]))
    if area <= 0:
        raise ValueError("Reference must be counterclockwise.")
    corridor = Corridor(road)
    clearance, _ = corridor.signed_clearance(center[:-1])
    if clearance.min() < margin - 1e-7:
        raise ValueError("Reference vertices violate the boundary margin.")
    count = len(vectors)
    for begin in range(0, count, 128):
        end = min(begin + 128, count)
        if intersection_matrix(center[begin:end], vectors[begin:end],
                               corridor.starts, corridor.vectors).any():
            raise ValueError("A reference segment crosses a track boundary.")
        intersects = intersection_matrix(center[begin:end], vectors[begin:end],
                                         center[:-1], vectors)
        i, j = np.arange(begin, end)[:, None], np.arange(count)[None]
        adjacent = ((i == j) | ((i - j) % count == 1) | ((j - i) % count == 1))
        if np.any(intersects & ~adjacent):
            raise ValueError("Reference has a self-intersection.")
    # For nonintersecting planar segments, the closest pair includes an endpoint.
    # Check boundary endpoints against reference segments as well as vice versa.
    reverse_distance, _ = nearest_segments(corridor.starts, center[:-1], vectors)
    minimum = float(min(clearance.min(), reverse_distance.min()))
    if minimum < margin - 1e-7:
        raise ValueError("A reference segment violates the boundary margin.")
    return minimum
