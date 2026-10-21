import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from matplotlib.animation import FuncAnimation

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

    speed = 2.0  # m/s
    dt = 0.02    # Her karede geçen simülasyon süresi.

    info = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top",
    )

    print(f"Track length: {track_length:.2f} m")
    print(f"Expected lap time: {track_length / speed:.2f} s")

    def update(frame):
        elapsed_time = frame * dt
        total_distance = speed * elapsed_time

        # Tur bitince yeniden pistin başlangıcına geç.
        lap_distance = total_distance % track_length
        completed_laps = int(total_distance / track_length)

        # Gidilen mesafenin karşılık geldiği koordinatları bul.
        car_x = np.interp(lap_distance, distance_along_track, x)
        car_y = np.interp(lap_distance, distance_along_track, y)

        car.set_data([car_x], [car_y])
        info.set_text(
            f"Speed: {speed:.1f} m/s\n"
            f"Simulation time: {elapsed_time:.1f} s\n"
            f"Completed laps: {completed_laps}"
        )

        return car, info

    fig.tight_layout()

    # Pencere açıkken animasyon nesnesini bellekte tut.
    animation = FuncAnimation(
        fig,
        update,
        frames=count(),
        interval=dt * 1000,
        blit=True,
        repeat=False,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
