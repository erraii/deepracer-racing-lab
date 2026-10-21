"""Inspect exactly the same track and speed plan used by main.py."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from speed_planner import Limits, build_speed_plan
from track import DEFAULT_TRACK, load_track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default=DEFAULT_TRACK)
    parser.add_argument("--no-show", action="store_true", help="Check and print without plotting.")
    args = parser.parse_args()
    track = load_track(args.track)
    limits = Limits()
    plan = build_speed_plan(track, limits)
    print(f"Original centerline length: {track.original_length:.3f} m")
    print(f"Shared sampled centerline length: {track.length:.3f} m")
    print(f"Samples: {len(track.ds)}")
    print(f"Curvature offset: {plan.curvature_offset:.3f} m")
    print(f"Maximum absolute curvature: {np.max(abs(plan.curvature)):.3f} 1/m")
    print(f"Local speed limit range: {plan.local_limit.min():.2f}-{plan.local_limit.max():.2f} m/s")
    print("Speed profile checks passed.")
    print(f"Planner iterations: {plan.iterations}")
    print(f"Planned speed range: {plan.speed.min():.2f}-{plan.speed.max():.2f} m/s")
    print(f"Estimated reference lap time: {plan.lap_time:.2f} s")
    print("Rolling lap; independent acceleration limits; uncalibrated vehicle parameters.")
    if args.no_show:
        return
    fig, (ax_track, ax_curve, ax_speed) = plt.subplots(
        3, 1, figsize=(12, 10), constrained_layout=True,
    )
    ax_track.plot(*track.outer.T, color="gray", linewidth=1)
    ax_track.plot(*track.inner.T, color="gray", linewidth=1)
    points = track.center[:-1]
    colored = ax_track.scatter(*points.T, c=abs(plan.curvature), cmap="viridis", s=18)
    ax_track.plot(*points[0], "rx", markersize=10, label="Start")
    ax_track.set(aspect="equal", xlabel="x [m]", ylabel="y [m]",
                 title="Forever Raceway CCW - Curvature Magnitude")
    ax_track.legend()
    fig.colorbar(colored, ax=ax_track, label="Absolute curvature [1/m]")
    # Repeat the first sample at lap end to show the periodic connection.
    ax_curve.plot(track.s, np.r_[plan.curvature, plan.curvature[0]], color="teal")
    ax_curve.axhline(0, color="gray", linewidth=1)
    ax_curve.set(xlabel="Distance from start [m]", ylabel="Curvature [1/m]",
                 title="Signed Curvature Along the Centerline")
    ax_curve.grid(alpha=0.3)
    ax_speed.plot(track.s, np.r_[plan.local_limit, plan.local_limit[0]], "--",
                  color="darkorange", alpha=0.7, label="Local speed limit")
    ax_speed.plot(track.s, np.r_[plan.speed, plan.speed[0]], color="royalblue",
                  linewidth=2, label="Planned speed")
    ax_speed.set(xlabel="Distance from start [m]", ylabel="Speed [m/s]",
                 ylim=(0, limits.max_speed * 1.1),
                 title="Speed Profile with Acceleration and Braking Limits")
    ax_speed.grid(alpha=0.3)
    ax_speed.legend()
    plt.show()


if __name__ == "__main__":
    main()
