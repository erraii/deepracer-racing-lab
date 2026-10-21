"""Checks for periodic planning, bounded motion and simulation termination."""

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from simulation import Simulation
from speed_planner import Limits, build_speed_plan, estimate_curvature
from track import Track, load_track
from vehicle import Vehicle


def circle(radius=3.0, count=240):
    angle = np.linspace(0.0, 2 * np.pi, count + 1)
    unit = np.column_stack((np.cos(angle), np.sin(angle)))
    unit[-1] = unit[0]
    center = radius * unit
    return Track(center, (radius + 0.5) * unit,
                 (radius - 0.5) * unit,
                 float(np.sum(np.linalg.norm(np.diff(center, axis=0), axis=1))), 1.0)


class ModelTests(unittest.TestCase):
    def test_circle_curvature_and_projection(self):
        track = circle()
        curvature, _ = estimate_curvature(track)
        np.testing.assert_allclose(curvature, 1 / 3, atol=1e-10)
        for distance in (0, 0.3, track.length - 0.01):
            point = track.position(distance)
            recovered, error = track.project(*point)
            self.assertAlmostEqual(recovered, distance, places=9)
            self.assertLess(error, 1e-10)
        self.assertTrue(track.contains(3, 0))
        self.assertFalse(track.contains(0, 0))
        self.assertFalse(track.contains(4, 0))

    def test_periodic_constraints_and_interpolation(self):
        track = circle()
        # Deliberately asymmetric path, including turns close to the lap seam.
        points = track.center.copy()
        points[:, 0] *= 1.8
        track = Track(points, track.outer, track.inner, track.original_length, track.width)
        limits = Limits(max_speed=5, acceleration=0.5, braking=0.8)
        plan = build_speed_plan(track, limits)
        squared = plan.speed**2
        following = np.roll(squared, -1)
        self.assertTrue(np.all(plan.speed <= plan.local_limit + 1e-9))
        self.assertTrue(np.all(following <= squared + 2 * limits.acceleration * track.ds + 1e-8))
        self.assertTrue(np.all(squared <= following + 2 * limits.braking * track.ds + 1e-8))
        for index in (0, len(squared) - 1):
            middle = track.s[index] + track.ds[index] / 2
            expected = (squared[index] + following[index]) / 2
            self.assertAlmostEqual(plan.target_at(track, middle)**2, expected, places=9)
        self.assertAlmostEqual(plan.target_at(track, 0), plan.target_at(track, track.length))

    def test_bounded_speed_changes_and_distance(self):
        car = Vehicle(0, 0, 0)
        _, distance, acceleration, _ = car.step(3, 0, 0.1)
        self.assertAlmostEqual(car.speed, 0.1)
        self.assertAlmostEqual(distance, 0.005)
        self.assertAlmostEqual(acceleration, 1)
        car.speed = 2
        car.step(0, 0, 0.1)
        self.assertAlmostEqual(car.speed, 1.8)
        car.step(0, 0, 10)
        self.assertEqual(car.speed, 0)

    def test_constant_steering_follows_circle(self):
        car = Vehicle(3, 0, math.pi / 2, speed=2)
        steering = math.atan(car.wheelbase / 3)
        for _ in range(1000):
            car.step(2, steering, 0.02)
        self.assertAlmostEqual(math.hypot(car.x, car.y), 3, places=10)

    def test_loader_closure_and_invalid_source(self):
        track = circle()
        raw = np.column_stack((track.center, track.inner, track.outer))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "circle.npy"
            np.save(path, raw[:-1])
            loaded = load_track(path)
            self.assertAlmostEqual(loaded.original_length, track.original_length)
            self.assertTrue(np.array_equal(loaded.center[0], loaded.center[-1]))
            np.testing.assert_allclose(loaded.outer, track.outer)
            np.save(path, raw[::-1])
            with self.assertRaises(ValueError):
                load_track(path)

    def test_complete_timeout_and_offtrack(self):
        track = circle()
        limits = Limits()
        plan = build_speed_plan(track, limits)
        sim = Simulation(track, plan, limits)
        while sim.status == "RUNNING":
            sim.step()
        self.assertEqual(sim.status, "COMPLETED")
        self.assertLess(sim.max_error, 0.1)
        elapsed = sim.elapsed
        sim.step()
        self.assertEqual(sim.elapsed, elapsed)
        timeout = Simulation(track, plan, limits, max_time=0.1)
        while timeout.status == "RUNNING":
            timeout.step()
        self.assertEqual(timeout.status, "TIMEOUT")
        offtrack = Simulation(track, plan, limits)
        offtrack.vehicle.x = 5
        offtrack.step()
        self.assertEqual(offtrack.status, "OFF TRACK")

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            Limits(acceleration=0)
        with self.assertRaises(ValueError):
            Vehicle(0, 0, 0, max_steering=math.pi / 2)


if __name__ == "__main__":
    unittest.main()
