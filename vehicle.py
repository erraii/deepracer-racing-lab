from dataclasses import dataclass
import math


@dataclass
class Vehicle:
    # x ve y, arka aksın orta noktasını temsil ediyor.
    x: float
    y: float
    yaw: float

    wheelbase: float = 0.33
    max_steering: float = math.radians(30)

    def step(self, speed: float, steering: float, dt: float):
        steering = max(
            -self.max_steering,
            min(self.max_steering, steering),
        )

        self.x += speed * math.cos(self.yaw) * dt
        self.y += speed * math.sin(self.yaw) * dt

        self.yaw += (
            speed / self.wheelbase
            * math.tan(steering)
            * dt
        )

        # Açıyı [-pi, pi] aralığında tut.
        self.yaw = math.atan2(
            math.sin(self.yaw),
            math.cos(self.yaw),
        )

        return steering
