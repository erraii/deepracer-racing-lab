"""Animate a lap, or run the same simulation without a GUI."""

import argparse
import math
from itertools import count

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from simulation import Simulation
from speed_planner import Limits, build_speed_plan
from track import DEFAULT_TRACK, load_track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default=DEFAULT_TRACK)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mode", choices=("planned", "constant"), default="planned")
    parser.add_argument("--speed", type=float, default=2.0, help="Constant-mode speed [m/s].")
    parser.add_argument("--lookahead", type=float, default=0.8)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--max-time", type=float, default=60.0)
    args = parser.parse_args()
    track = load_track(args.track)
    limits = Limits()
    plan = build_speed_plan(track, limits)
    sim = Simulation(track, plan, limits, args.lookahead, args.dt,
                     args.mode, args.speed, args.max_time)
    print(f"Original centerline length: {track.original_length:.3f} m")
    print(f"Shared sampled centerline length: {track.length:.3f} m")
    print(f"Median track width: {track.width:.3f} m")
    print(f"Reference lap estimate: {plan.lap_time:.2f} s")

    if args.headless:
        while sim.status == "RUNNING":
            sim.step()
        sim.report()
        return

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.set_facecolor("#b8d8a5")
    ax.fill(*track.outer.T, color="#555555")
    ax.fill(*track.inner.T, color="#b8d8a5")
    ax.plot(*track.outer.T, color="white", linewidth=2)
    ax.plot(*track.inner.T, color="white", linewidth=2)
    ax.plot(*track.center.T, "--", color="gold", label="Center line")
    ax.plot([track.inner[0, 0], track.outer[0, 0]],
            [track.inner[0, 1], track.outer[0, 1]], color="white", linewidth=4)
    car, = ax.plot([], [], "ro", markersize=9, label="Car", zorder=5)
    target, = ax.plot([], [], "bx", markersize=10, label="Steering target", zorder=6)
    trail, = ax.plot([], [], color="cyan", linewidth=1, alpha=0.8, label="Driven path")
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                   bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.set(aspect="equal", xlabel="x [m]", ylabel="y [m]",
           title=f"Forever Raceway CCW - {args.mode} speed")
    ax.legend(loc="upper right")
    history_x, history_y = [sim.vehicle.x], [sim.vehicle.y]

    def draw():
        car.set_data([sim.vehicle.x], [sim.vehicle.y])
        target.set_data([sim.target[0]], [sim.target[1]])
        trail.set_data(history_x, history_y)
        info.set_text(
            f"Status: {sim.status}\n"
            f"Time: {sim.elapsed:.2f} s\n"
            f"Speed / target: {sim.vehicle.speed:.2f} / {sim.target_speed:.2f} m/s\n"
            f"Steering: {math.degrees(sim.steering):.1f} deg\n"
            f"Acceleration: {sim.acceleration:.2f} m/s^2\n"
            f"Max center error: {sim.max_error:.3f} m\n"
            f"Lateral demand: {sim.lateral_demand:.2f} m/s^2"
        )
        return car, target, trail, info

    def update(_frame):
        if sim.status == "RUNNING":
            sim.step()
            history_x.append(sim.vehicle.x)
            history_y.append(sim.vehicle.y)
            if sim.status != "RUNNING":
                animation.event_source.stop()
                sim.report()
        return draw()

    animation = FuncAnimation(
        fig, update, init_func=draw, frames=count(), interval=sim.dt * 1000,
        blit=True, repeat=False, cache_frame_data=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
