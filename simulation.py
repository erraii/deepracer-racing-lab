"""One simulation step shared by the GUI and headless runs."""

import math

from vehicle import Vehicle


class Simulation:
    def __init__(self, track, plan, limits, lookahead=0.8, dt=0.02,
                 mode="planned", constant_speed=2.0, max_time=60.0):
        values = (lookahead, dt, constant_speed, max_time)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Simulation parameters must be positive and finite.")
        if mode not in ("planned", "constant"):
            raise ValueError("Unknown speed mode.")
        self.track, self.plan, self.limits = track, plan, limits
        self.lookahead, self.dt = lookahead, dt
        self.mode, self.constant_speed, self.max_time = mode, constant_speed, max_time
        self.elapsed = self.progress = self.distance = self.max_error = 0.0
        self.max_lateral_demand = self.max_speed_error = self.over_limit_time = 0.0
        self.previous_s = 0.0
        self.status = "RUNNING"
        start, following = track.center[:2]
        initial_speed = plan.target_at(track, 0.0) if mode == "planned" else constant_speed
        self.vehicle = Vehicle(
            x=float(start[0]), y=float(start[1]),
            yaw=math.atan2(following[1] - start[1], following[0] - start[0]),
            speed=initial_speed,
            max_acceleration=limits.acceleration, max_braking=limits.braking,
        )
        self.target = track.position(lookahead)
        self.target_speed = initial_speed
        self.steering = self.acceleration = self.lateral_demand = self.error = 0.0
        self.min_speed = self.max_speed = initial_speed

    def step(self):
        if self.status != "RUNNING":
            return
        car, track = self.vehicle, self.track
        position_s, _ = track.project(car.x, car.y)
        self.target = track.position(position_s + self.lookahead)
        dx, dy = self.target[0] - car.x, self.target[1] - car.y
        alpha = math.atan2(dy, dx) - car.yaw
        steering = math.atan2(2 * car.wheelbase * math.sin(alpha), max(math.hypot(dx, dy), 1e-6))
        self.target_speed = (
            self.plan.target_at(track, position_s)
            if self.mode == "planned" else self.constant_speed
        )
        self.steering, distance, self.acceleration, self.lateral_demand = car.step(
            self.target_speed, steering, self.dt,
        )
        self.elapsed += self.dt
        self.distance += distance
        new_s, self.error = track.project(car.x, car.y)
        self.progress += (new_s - self.previous_s + track.length / 2) % track.length - track.length / 2
        self.previous_s = new_s
        self.max_error = max(self.max_error, self.error)
        self.min_speed = min(self.min_speed, car.speed)
        self.max_speed = max(self.max_speed, car.speed)
        self.max_speed_error = max(self.max_speed_error, abs(car.speed - self.target_speed))
        self.max_lateral_demand = max(self.max_lateral_demand, self.lateral_demand)
        if self.lateral_demand > self.limits.lateral_acceleration + 1e-6:
            self.over_limit_time += self.dt
        if not track.contains(car.x, car.y):
            self.status = "OFF TRACK"
        elif self.progress >= track.length:
            self.status = "COMPLETED"
        elif self.elapsed >= self.max_time:
            self.status = "TIMEOUT"

    def report(self):
        print("\n--- Run results ---")
        print(f"Status: {self.status}")
        print(f"Mode: {self.mode}; rolling start")
        print(f"Lookahead: {self.lookahead:.2f} m")
        print(f"Simulation time: {self.elapsed:.2f} s")
        print(f"Distance driven: {self.distance:.3f} m")
        print(f"Observed speed range: {self.min_speed:.2f}-{self.max_speed:.2f} m/s")
        print(f"Maximum center error: {self.max_error:.3f} m")
        print(f"Maximum target speed error: {self.max_speed_error:.3f} m/s")
        print(f"Reference lap estimate: {self.plan.lap_time:.2f} s")
        print(f"Maximum lateral demand: {self.max_lateral_demand:.3f} m/s^2")
        print(f"Planning lateral limit: {self.limits.lateral_acceleration:.3f} m/s^2")
        print(f"Time above planning lateral limit: {self.over_limit_time:.2f} s")
        print("Point-only boundary check. No tire slip or combined grip constraint.")
