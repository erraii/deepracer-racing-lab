"""Conservative speed scaling verified with the selected controller and road."""

from dataclasses import replace
import math

import numpy as np

from simulation import Simulation
from speed_planner import apply_speed_caps


def calibrate_speed_plan(track, plan, limits, *, road_track=None,
                         lookahead=0.8, dt=0.02, max_time=60.0,
                         reserve=0.02, max_attempts=10):
    """Return a plan verified over one rolling lap, or raise without accepting it.

    Uniform scaling preserves the existing spatial acceleration/braking bounds.
    Steering is never modified. Every new scale is simulated because changing
    speed can also change the discrete controller's driven trajectory.
    The reserve is relative to the planning lateral limit, not a tire model.
    """
    if not math.isfinite(reserve) or not 0 <= reserve < 1:
        raise ValueError("Reserve must be finite and in [0, 1).")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer.")
    target = limits.lateral_acceleration * (1 - reserve)
    scale = 1.0
    history = []
    for attempt in range(1, max_attempts + 1):
        candidate = replace(plan, speed=plan.speed * scale,
                            lap_time=plan.lap_time / scale)
        sim = Simulation(track, candidate, limits, lookahead=lookahead,
                         dt=dt, max_time=max_time, road_track=road_track)
        while sim.status == "RUNNING":
            sim.step()
        history.append((scale, sim.max_lateral_demand, sim.elapsed, sim.status))
        print(f"Speed validation {attempt}: scale={scale:.4f}; "
              f"peak lateral={sim.max_lateral_demand:.4f} m/s^2; "
              f"lap={sim.elapsed:.2f} s; {sim.status}")
        if sim.status != "COMPLETED":
            raise RuntimeError(f"Speed validation failed: {sim.status}. "
                               "No validated speed plan was accepted.")
        if sim.max_lateral_demand <= target:
            return candidate, history
        # v^2 * curvature predicts the next scale; the next full lap verifies it.
        scale *= min(0.99, 0.995 * math.sqrt(target / sim.max_lateral_demand))
    raise RuntimeError("Speed validation did not converge. "
                       "No validated speed plan was accepted.")


def refine_speed_plan_locally(track, plan, limits, *, road_track=None,
                             lookahead=0.8, dt=0.02, max_time=60.0,
                             reserve=0.02, max_attempts=12):
    """Lower caps near measured violations, then propagate braking backwards.

    Caps only decrease. A full simulation must pass before acceptance. If local
    refinement exhausts its attempts, global calibration is used as a fallback.
    This is controller-specific empirical refinement, not a time optimum.
    """
    if not math.isfinite(reserve) or not 0 <= reserve < 1:
        raise ValueError("Reserve must be finite and in [0, 1).")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer.")
    target = limits.lateral_acceleration * (1 - reserve)
    candidate = plan
    history = []
    for attempt in range(1, max_attempts + 1):
        sim = Simulation(track, candidate, limits, lookahead=lookahead,
                         dt=dt, max_time=max_time, road_track=road_track)
        violations = []
        while sim.status == "RUNNING":
            before = sim.previous_s
            sim.step()
            if sim.lateral_demand > target:
                # Include both endpoints and nearby samples of this control step.
                advance = (sim.previous_s - before + track.length / 2) % track.length - track.length / 2
                middle = (before + advance / 2) % track.length
                violations.append((middle, abs(advance) / 2 + 0.10,
                                   0.985 * math.sqrt(target / sim.lateral_demand)))
        history.append((sim.max_lateral_demand, sim.elapsed, sim.status))
        print(f"Local speed validation {attempt}: "
              f"peak lateral={sim.max_lateral_demand:.4f} m/s^2; "
              f"lap={sim.elapsed:.2f} s; {sim.status}")
        if sim.status != "COMPLETED":
            raise RuntimeError(f"Local speed validation failed: {sim.status}. "
                               "No validated speed plan was accepted.")
        if sim.max_lateral_demand <= target:
            return candidate, history
        caps = candidate.local_limit.copy()
        for middle, radius, factor in violations:
            distance = abs((track.s[:-1] - middle + track.length / 2)
                           % track.length - track.length / 2)
            mask = distance <= max(radius, float(np.max(track.ds)))
            caps[mask] = np.minimum(caps[mask], candidate.speed[mask] * factor)
        candidate = apply_speed_caps(track, candidate, limits, caps)
    print("Local attempts exhausted; applying validated global fallback.")
    candidate, global_history = calibrate_speed_plan(
        track, candidate, limits, road_track=road_track, lookahead=lookahead,
        dt=dt, max_time=max_time, reserve=reserve,
    )
    history.extend((peak, elapsed, status) for _, peak, elapsed, status in global_history)
    return candidate, history
