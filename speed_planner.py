"""Periodic reference speed plan with independent acceleration limits."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Limits:
    max_speed: float = 3.0
    lateral_acceleration: float = 3.0
    acceleration: float = 1.0
    braking: float = 2.0

    def __post_init__(self):
        values = (
            self.max_speed, self.lateral_acceleration,
            self.acceleration, self.braking,
        )
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("All planning limits must be positive and finite.")


@dataclass
class SpeedPlan:
    curvature: np.ndarray
    local_limit: np.ndarray
    speed: np.ndarray
    iterations: int
    lap_time: float
    curvature_offset: float

    def target_at(self, track, distance):
        # Linear interpolation of v^2 preserves the spatial acceleration bounds.
        squared = np.concatenate((self.speed**2, [self.speed[0]**2]))
        return float(np.sqrt(np.interp(
            distance % track.length, track.s, squared,
        )))


def estimate_curvature(track, window=0.25):
    if not np.isfinite(window) or window <= 0:
        raise ValueError("Curvature window must be positive and finite.")
    points = track.center[:-1]
    offset = min(
        max(1, round(window / float(np.mean(track.ds)))),
        (len(points) - 1) // 2,
    )
    previous = np.roll(points, offset, axis=0)
    following = np.roll(points, -offset, axis=0)
    a, b, c = points - previous, following - points, following - previous
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    denominator = (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        * np.linalg.norm(c, axis=1)
    )
    curvature = np.divide(
        2 * cross, denominator, out=np.zeros_like(cross),
        where=denominator > 1e-12,
    )
    return curvature, offset * float(np.mean(track.ds))


def build_speed_plan(track, limits=None, window=0.25):
    limits = Limits() if limits is None else limits
    curvature, actual_offset = estimate_curvature(track, window)
    local_limit = np.minimum(
        limits.max_speed,
        np.sqrt(limits.lateral_acceleration / np.maximum(abs(curvature), 1e-6)),
    )
    return _plan_from_limits(track, limits, curvature, local_limit, actual_offset)


def apply_speed_caps(track, plan, limits, speed_caps):
    """Re-run periodic acceleration/braking passes after lowering local caps."""
    caps = np.asarray(speed_caps, dtype=float)
    if caps.shape != plan.speed.shape or not np.all(np.isfinite(caps) & (caps > 0)):
        raise ValueError("Speed caps must be positive finite values at every reference sample.")
    caps = np.minimum(caps, plan.local_limit)
    return _plan_from_limits(track, limits, plan.curvature, caps, plan.curvature_offset)


def _plan_from_limits(track, limits, curvature, local_limit, actual_offset):
    squared = local_limit**2
    count = len(squared)
    for iteration in range(count + 1):
        previous = squared.copy()
        for i in range(count):
            j = (i + 1) % count
            squared[j] = min(squared[j], squared[i] + 2 * limits.acceleration * track.ds[i])
        for i in range(count - 1, -1, -1):
            j = (i + 1) % count
            squared[i] = min(squared[i], squared[j] + 2 * limits.braking * track.ds[i])
        if np.max(abs(squared - previous)) < 1e-10:
            break
    else:
        raise RuntimeError("Speed planner did not converge.")

    next_squared = np.roll(squared, -1)
    checks = (
        squared <= local_limit**2 + 1e-8,
        next_squared <= squared + 2 * limits.acceleration * track.ds + 1e-8,
        squared <= next_squared + 2 * limits.braking * track.ds + 1e-8,
    )
    if not all(np.all(check) for check in checks):
        raise RuntimeError("Speed profile validation failed.")
    speed = np.sqrt(squared)
    lap_time = float(np.sum(2 * track.ds / (speed + np.roll(speed, -1))))
    return SpeedPlan(curvature, local_limit, speed, iteration + 1, lap_time, actual_offset)
