from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    track_file = (
        Path(__file__).resolve().parent
        / "data/tracks/2024_reinvent_champ_ccw.npy"
    )
    track = np.load(track_file, allow_pickle=False)

    center = track[:, :2]

    if not np.allclose(center[0], center[-1]):
        center = np.vstack((center, center[0]))

    segment_lengths = np.linalg.norm(
        np.diff(center, axis=0),
        axis=1,
    )

    if np.any(segment_lengths < 1e-9):
        raise ValueError("Track contains zero-length segments.")

    raw_s = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    track_length = float(raw_s[-1])

    # Tur boyunca yaklaşık 5 cm aralıklı örnekler.
    # Son noktayı tekrarlamıyoruz; bağlantıyı np.roll kuracak.
    count = int(np.ceil(track_length / 0.05))
    s = np.linspace(0.0, track_length, count, endpoint=False)

    points = np.column_stack((
        np.interp(s, raw_s, center[:, 0]),
        np.interp(s, raw_s, center[:, 1]),
    ))

    # Eğriliği hesaplamak için yaklaşık 25 cm geriye
    # ve 25 cm ileriye bakıyoruz.
    sample_spacing = track_length / count
    offset = max(1, round(0.25 / sample_spacing))

    previous = np.roll(points, offset, axis=0)
    following = np.roll(points, -offset, axis=0)

    a = points - previous
    b = following - points
    c = following - previous

    # Üç noktadan geçen çemberin işaretli eğriliği.
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    denominator = (
        np.linalg.norm(a, axis=1)
        * np.linalg.norm(b, axis=1)
        * np.linalg.norm(c, axis=1)
    )

    curvature = np.divide(
        2.0 * cross,
        denominator,
        out=np.zeros_like(cross),
        where=denominator > 1e-12,
    )

    max_speed = 3.0
    max_lateral_acceleration = 3.0

    local_speed_limit = np.minimum(
        max_speed,
        np.sqrt(
            max_lateral_acceleration
            / np.maximum(np.abs(curvature), 1e-6)
        ),
    )

    print(
        f"Local speed limit range: "
        f"{local_speed_limit.min():.2f}–"
        f"{local_speed_limit.max():.2f} m/s"
    )

    max_speed = 3.0
    max_lateral_acceleration = 3.0

    local_speed_limit = np.minimum(
        max_speed,
        np.sqrt(
            max_lateral_acceleration
            / np.maximum(np.abs(curvature), 1e-6)
        ),
    )

    max_acceleration = 1.0
    max_braking = 2.0

    # s, orijinal merkez çizgisi boyunca mesafeyi temsil ediyor.
    # Son segment, son örnekten başlangıca kadar devam ediyor.
    ds = np.diff(np.concatenate((s, [track_length])))

    # Hesapları hızın karesi üzerinden yapıyoruz.
    speed_squared = local_speed_limit**2

    for iteration in range(count):
        previous = speed_squared.copy()

        # İleri geçiş: hızlanma sınırı.
        for i in range(count):
            j = (i + 1) % count

            speed_squared[j] = min(
                speed_squared[j],
                speed_squared[i] + 2 * max_acceleration * ds[i],
            )

        # Geri geçiş: yaklaşan viraj için frenleme sınırı.
        for i in range(count - 1, -1, -1):
            j = (i + 1) % count

            speed_squared[i] = min(
                speed_squared[i],
                speed_squared[j] + 2 * max_braking * ds[i],
            )

        # Kapalı pistte başlangıç ve bitiş de birbirini etkiler.
        if np.max(np.abs(speed_squared - previous)) < 1e-10:
            break
    else:
        raise RuntimeError("Speed planner did not converge.")

    planned_speed = np.sqrt(speed_squared)

    tolerance = 1e-8
    next_squared = np.roll(speed_squared, -1)

    assert np.all(
        planned_speed <= local_speed_limit + tolerance
    ), "Local speed limit exceeded."

    assert np.all(
        next_squared
        <= speed_squared + 2 * max_acceleration * ds + tolerance
    ), "Acceleration limit exceeded."

    assert np.all(
        speed_squared
        <= next_squared + 2 * max_braking * ds + tolerance
    ), "Braking limit exceeded."

    print("Speed profile checks passed.")

    next_speed = np.roll(planned_speed, -1)
    estimated_lap_time = np.sum(
        2 * ds / (planned_speed + next_speed)
    )

    print(f"Planner iterations: {iteration + 1}")
    print(
        f"Planned speed range: "
        f"{planned_speed.min():.2f}–"
        f"{planned_speed.max():.2f} m/s"
    )
    print(
        f"Estimated reference lap time: "
        f"{estimated_lap_time:.2f} s"
    )

    print(
        f"Local speed limit range: "
        f"{local_speed_limit.min():.2f}–"
        f"{local_speed_limit.max():.2f} m/s"
    )

    print(f"Track length: {track_length:.3f} m")
    print(f"Samples: {count}")
    print(f"Sample spacing: {sample_spacing:.3f} m")
    print(f"Curvature offset: {offset * sample_spacing:.3f} m")
    print(
        f"Maximum absolute curvature: "
        f"{np.max(np.abs(curvature)):.3f} 1/m"
    )

    fig, (ax_track, ax_curve, ax_speed) = plt.subplots(
        3, 1,
        figsize=(12, 10),
        constrained_layout=True,
    )

    # Orijinal pist sınırları.
    ax_track.plot(
        track[:, 2], track[:, 3],
        color="gray", linewidth=1,
    )
    ax_track.plot(
        track[:, 4], track[:, 5],
        color="gray", linewidth=1,
    )

    colored_points = ax_track.scatter(
        points[:, 0],
        points[:, 1],
        c=np.abs(curvature),
        cmap="viridis",
        s=18,
    )

    ax_track.plot(
        points[0, 0], points[0, 1],
        "rx", markersize=10,
        label="Start",
    )

    ax_track.set_aspect("equal")
    ax_track.set_title("Forever Raceway CCW — Curvature Magnitude")
    ax_track.set_xlabel("x [m]")
    ax_track.set_ylabel("y [m]")
    ax_track.legend()

    fig.colorbar(
        colored_points,
        ax=ax_track,
        label="Absolute curvature [1/m]",
    )

    ax_curve.plot(s, curvature, color="teal")
    ax_curve.axhline(0.0, color="gray", linewidth=1)
    ax_curve.set_title("Signed Curvature Along the Centerline")
    ax_curve.set_xlabel("Distance from start [m]")
    ax_curve.set_ylabel("Curvature [1/m]")
    ax_curve.grid(alpha=0.3)

    ax_speed.plot(
        s,
        local_speed_limit,
        "--",
        color="darkorange",
        alpha=0.7,
        label="Local speed limit",
    )

    ax_speed.plot(
        s,
        planned_speed,
        color="royalblue",
        linewidth=2,
        label="Planned speed",
    )

    ax_speed.set_title("Speed Profile with Acceleration and Braking Limits")
    ax_speed.set_xlabel("Distance from start [m]")
    ax_speed.set_ylabel("Speed [m/s]")
    ax_speed.set_ylim(0, max_speed * 1.1)
    ax_speed.grid(alpha=0.3)
    ax_speed.legend()

    plt.show()


if __name__ == "__main__":
    main()
