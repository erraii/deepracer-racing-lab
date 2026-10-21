import matplotlib.pyplot as plt
import numpy as np


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
    ax.plot(x[0], y[0], "ro", markersize=9, label="Car")

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("DeepRacer Racing Lab — First Track")
    ax.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
