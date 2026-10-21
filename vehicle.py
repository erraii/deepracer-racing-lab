"""Kinematic bicycle with bounded speed changes; no tire-slip dynamics."""

from dataclasses import dataclass
import math


@dataclass
class Vehicle:
    x: float
    y: float
    yaw: float
    speed: float = 0.0
    wheelbase: float = 0.33
    max_steering: float = math.radians(30)
    max_acceleration: float = 1.0
    max_braking: float = 2.0

    def __post_init__(self):
        values = (self.wheelbase, self.max_steering, self.max_acceleration, self.max_braking)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Vehicle limits must be positive and finite.")
        if self.max_steering >= math.pi / 2:
            raise ValueError("Maximum steering must be below 90 degrees.")
        if not all(math.isfinite(v) for v in (self.x, self.y, self.yaw, self.speed)) or self.speed < 0:
            raise ValueError("Vehicle state must be finite with nonnegative speed.")

    def step(self, target_speed, steering, dt):
        if not all(math.isfinite(v) for v in (target_speed, steering, dt)) or dt <= 0 or target_speed < 0:
            raise ValueError("Invalid command or time step.")
        steering = max(-self.max_steering, min(self.max_steering, steering))
        old_speed = self.speed
        change = max(-self.max_braking * dt, min(self.max_acceleration * dt, target_speed - old_speed))
        self.speed = max(0.0, old_speed + change)
        distance = 0.5 * (old_speed + self.speed) * dt
        curvature = math.tan(steering) / self.wheelbase
        yaw_change = distance * curvature
        if abs(curvature) < 1e-10:
            self.x += distance * math.cos(self.yaw)
            self.y += distance * math.sin(self.yaw)
        else:
            self.x += (math.sin(self.yaw + yaw_change) - math.sin(self.yaw)) / curvature
            self.y += (math.cos(self.yaw) - math.cos(self.yaw + yaw_change)) / curvature
        self.yaw = math.atan2(math.sin(self.yaw + yaw_change), math.cos(self.yaw + yaw_change))
        # Maximum kinematic lateral demand during this step, not a slip model.
        lateral_demand = max(old_speed, self.speed)**2 * abs(curvature)
        return steering, distance, change / dt, lateral_demand
