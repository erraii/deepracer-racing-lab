import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from matplotlib.animation import FuncAnimation
from vehicle import Vehicle
from pathlib import Path
from matplotlib.path import Path as PlotPath

def main():
    track_file = (
        Path(__file__).resolve().parent
        / "data/tracks/2024_reinvent_champ_ccw.npy"
    )

    track = np.load(track_file, allow_pickle=False)

    if track.ndim != 2 or track.shape[1] != 6:
        raise ValueError("Expected track data with six columns.")

    if not np.isfinite(track).all():
        raise ValueError("Track contains invalid coordinates.")

    if not np.allclose(track[0], track[-1]):
        track = np.vstack((track, track[0]))

    # Orijinal merkez çizgisi boyunca mesafeler.
    raw_center = track[:, :2]
    raw_lengths = np.linalg.norm(
        np.diff(raw_center, axis=0),
        axis=1,
    )

    if np.any(raw_lengths < 1e-9):
        raise ValueError("Track contains zero-length segments.")

    raw_s = np.concatenate(([0.0], np.cumsum(raw_lengths)))

    # Kontrol algoritması için yaklaşık 5 cm aralıklı noktalar.
    # Orijinal dosyayı değiştirmiyoruz.
    sample_count = int(np.ceil(raw_s[-1] / 0.05)) + 1
    sample_s = np.linspace(0.0, raw_s[-1], sample_count)

    sampled = np.column_stack([
        np.interp(sample_s, raw_s, track[:, column])
        for column in range(6)
    ])

    x = sampled[:, 0]
    y = sampled[:, 1]

    boundary_a = sampled[:, 2:4]
    boundary_b = sampled[:, 4:6]

    def signed_area(points):
        return 0.5 * np.sum(
            points[:-1, 0] * points[1:, 1]
            - points[1:, 0] * points[:-1, 1]
        )

    # Standart x-y eksenlerinde pozitif alan CCW demektir.
    if signed_area(sampled[:, :2]) <= 0:
        raise ValueError("Expected counterclockwise track order.")

    if abs(signed_area(boundary_a)) > abs(signed_area(boundary_b)):
        outer, inner = boundary_a, boundary_b
    else:
        outer, inner = boundary_b, boundary_a

    outer_x, outer_y = outer.T
    inner_x, inner_y = inner.T

    track_width = float(np.median(
        np.linalg.norm(boundary_a - boundary_b, axis=1)
    ))

    outer_path = PlotPath(outer, closed=True)
    inner_path = PlotPath(inner, closed=True)

    print(f"Loaded track: {track_file.name}")
    print(f"Original centerline length: {raw_s[-1]:.3f} m")
    print(f"Median track width: {track_width:.3f} m")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#b8d8a5")

    ax.fill(outer_x, outer_y, color="#555555")
    ax.fill(inner_x, inner_y, color="#b8d8a5")

    ax.plot(outer_x, outer_y, color="white", linewidth=2)
    ax.plot(inner_x, inner_y, color="white", linewidth=2)
    ax.plot(x, y, "--", color="gold", label="Center line")

    # Başlangıç çizgisi ve arabanın başlangıç konumu.
    ax.plot(
        [inner_x[0], outer_x[0]],
        [inner_y[0], outer_y[0]],
        color="white",
        linewidth=4,
    )

    car, = ax.plot(
        [x[0]], [y[0]],
        "ro",
        markersize=9,
        label="Car",
        zorder=5,
    )

    target, = ax.plot(
            [], [],
            "bx",
            markersize=10,
            label="Target",
            zorder=6,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("DeepRacer Racing Lab — Forever Raceway CCW")
    ax.legend(loc="upper right")

    # Merkez çizgisi boyunca biriken mesafe.
    segment_lengths = np.hypot(np.diff(x), np.diff(y))
    distance_along_track = np.concatenate(
        ([0.0], np.cumsum(segment_lengths))
    )
    track_length = distance_along_track[-1]

    points = np.column_stack((x, y))
    segment_starts = points[:-1]
    segment_vectors = np.diff(points, axis=0)

    def project_to_track(px, py):
        offset = np.array([px, py]) - segment_starts

        fraction = np.clip(
            np.sum(offset * segment_vectors, axis=1)
            / segment_lengths**2,
            0.0,
            1.0,
        )

        projections = (
            segment_starts
            + fraction[:, None] * segment_vectors
        )

        distances = np.linalg.norm(
            projections - np.array([px, py]),
            axis=1,
        )

        index = int(np.argmin(distances))

        position_s = (
            distance_along_track[index]
            + fraction[index] * segment_lengths[index]
        )

        return float(position_s), float(distances[index])

    speed = 2.0  # m/s
    dt = 0.02    # Her karede geçen simülasyon süresi.

    info = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top",
    )

    initial_yaw = np.arctan2(
        y[1] - y[0],
        x[1] - x[0],
    )

    vehicle = Vehicle(
        x=float(x[0]),
        y=float(y[0]),
        yaw=float(initial_yaw),
    )

    lookahead = 0.8  # Metre: çizgi üzerinde ne kadar ileri bakıyoruz?

    initial_s, _ = project_to_track(vehicle.x, vehicle.y)

    results = {
        "previous_s": initial_s,
        "progress": 0.0,
        "max_error": 0.0,
        "finished": False,
    }

    print(f"Track length: {track_length:.2f} m")
    print(f"Speed: {speed:.2f} m/s")
    print(f"Lookahead: {lookahead:.2f} m")

    def initialize():
        car.set_data([vehicle.x], [vehicle.y])
        target.set_data([], [])
        info.set_text("")
        return car, target, info

    def update(frame):
        if results["finished"]:
            return car, target, info
        # 1. Araca en yakın merkez çizgisi noktasını bul.
        distances = np.hypot(
            x[:-1] - vehicle.x,
            y[:-1] - vehicle.y,
        )
        nearest = int(np.argmin(distances))

        # 2. Merkez çizgisi üzerinde biraz ileride hedef seç.
        target_distance = (
            distance_along_track[nearest] + lookahead
        ) % track_length

        target_x = np.interp(
            target_distance, distance_along_track, x
        )
        target_y = np.interp(
            target_distance, distance_along_track, y
        )

        # 3. Hedefin aracın yönüne göre açısını hesapla.
        dx = target_x - vehicle.x
        dy = target_y - vehicle.y

        alpha = np.arctan2(dy, dx) - vehicle.yaw
        target_range = max(np.hypot(dx, dy), 1e-6)

        # 4. Pure Pursuit direksiyon açısını hesapla.
        steering = np.arctan2(
            2 * vehicle.wheelbase * np.sin(alpha),
            target_range,
        )

        # 5. Direksiyonu uygula; araç modeli konumu güncellesin.
        applied_steering = vehicle.step(speed, steering, dt)

        # 6. Yeni konumu ve bilgileri göster.
        car.set_data([vehicle.x], [vehicle.y])
        target.set_data([target_x], [target_y])

        center_error = np.min(
            np.hypot(x[:-1] - vehicle.x, y[:-1] - vehicle.y)
        )

        position_s, center_error = project_to_track(
            vehicle.x, vehicle.y
        )

        # Başlangıç çizgisini geçerken mesafenin sıfıra
        # dönmesini hesaba katarak ilerlemeyi biriktir.
        progress_step = (
            position_s
            - results["previous_s"]
            + track_length / 2
        ) % track_length - track_length / 2

        results["progress"] += progress_step
        results["previous_s"] = position_s

        results["max_error"] = max(
            results["max_error"], center_error
        )

        elapsed_time = (frame + 1) * dt

        # Şimdilik yalnızca arka aksın orta noktasını kontrol et.
        position = (vehicle.x, vehicle.y)

        offtrack = (
            not outer_path.contains_point(position)
            or inner_path.contains_point(position)
        )
        completed = results["progress"] >= track_length

        status = "Running"

        if offtrack or completed:
            results["finished"] = True
            status = "OFF TRACK" if offtrack else "COMPLETED"

            animation.event_source.stop()

            print("\n--- Run results ---")
            print(f"Status: {status}")
            print(f"Lookahead: {lookahead:.2f} m")
            print(f"Speed: {speed:.2f} m/s")
            print(f"Simulation time: {elapsed_time:.2f} s")
            print(
                f"Maximum center error: "
                f"{results['max_error']:.3f} m"
            )

        info.set_text(
            f"Status: {status}\n"
            f"Speed: {speed:.1f} m/s\n"
            f"Simulation time: {elapsed_time:.2f} s\n"
            f"Steering: {np.degrees(applied_steering):.1f} deg\n"
            f"Center error: {center_error:.3f} m\n"
            f"Max error: {results['max_error']:.3f} m"
        )

        return car, target, info

    fig.tight_layout()

    animation = FuncAnimation(
        fig,
        update,
        init_func=initialize,
        frames=count(),
        interval=dt * 1000,
        blit=True,
        repeat=False,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
