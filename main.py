import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from matplotlib.animation import FuncAnimation
from vehicle import Vehicle

def main():
    # Oval merkez çizgisi: uzunluklar metre cinsinden.
    a = 5.0
    b = 3.0
    track_width = 1.0

    angle = np.linspace(0, 2 * np.pi, 401)

    x = a * np.cos(angle)
    y = b * np.sin(angle)

    # Merkez çizgisine dik, dışa bakan birim vektörler.
    normal_x = np.cos(angle) / a
    normal_y = np.sin(angle) / b
    normal_length = np.hypot(normal_x, normal_y)

    normal_x /= normal_length
    normal_y /= normal_length

    half_width = track_width / 2

    outer_x = x + half_width * normal_x
    outer_y = y + half_width * normal_y
    inner_x = x - half_width * normal_x
    inner_y = y - half_width * normal_y

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
    ax.set_title("DeepRacer Racing Lab — First Track")
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

    vehicle = Vehicle(
        x=float(x[0]),
        y=float(y[0]),
        yaw=np.pi / 2,
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
        offtrack = center_error > track_width / 2
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
