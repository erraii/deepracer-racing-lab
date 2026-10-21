"""Controller validation must actually reduce demand and reject failed laps."""

import unittest

import numpy as np

from simulation import Simulation
from speed_calibration import calibrate_speed_plan, refine_speed_plan_locally
from speed_planner import Limits, build_speed_plan
from test_models import circle


class SpeedCalibrationTests(unittest.TestCase):
    def test_local_caps_reduce_demand_and_preserve_periodic_bounds(self):
        track = circle()
        limits = Limits(max_speed=5.0)
        original = build_speed_plan(track, limits)
        plan, history = refine_speed_plan_locally(track, original, limits, lookahead=0.6)
        self.assertGreater(len(history), 1)
        sim = Simulation(track, plan, limits, lookahead=0.6)
        while sim.status == "RUNNING":
            sim.step()
        self.assertEqual(sim.status, "COMPLETED")
        self.assertLessEqual(sim.max_lateral_demand, 2.94)
        self.assertEqual(sim.over_limit_time, 0)
        self.assertTrue(np.all(plan.speed <= original.speed + 1e-8))
        squared = plan.speed**2
        following = np.roll(squared, -1)
        self.assertTrue(np.all(following <= squared + 2 * limits.acceleration * track.ds + 1e-8))
        self.assertTrue(np.all(squared <= following + 2 * limits.braking * track.ds + 1e-8))

    def test_local_failed_lap_is_not_accepted(self):
        track = circle()
        limits = Limits()
        with self.assertRaisesRegex(RuntimeError, "TIMEOUT"):
            refine_speed_plan_locally(track, build_speed_plan(track, limits), limits,
                                     max_time=0.02)

    def test_scaled_plan_meets_demand_and_periodic_acceleration_bounds(self):
        track = circle(radius=3.0)
        limits = Limits(max_speed=5.0)
        original = build_speed_plan(track, limits)
        plan, history = calibrate_speed_plan(track, original, limits, lookahead=0.6)
        self.assertGreater(len(history), 1)
        self.assertLess(history[-1][0], 1)
        sim = Simulation(track, plan, limits, lookahead=0.6)
        while sim.status == "RUNNING":
            sim.step()
        self.assertEqual(sim.status, "COMPLETED")
        self.assertLessEqual(sim.max_lateral_demand, 0.98 * limits.lateral_acceleration)
        self.assertEqual(sim.over_limit_time, 0)
        squared = plan.speed**2
        following = np.roll(squared, -1)
        self.assertTrue(np.all(plan.speed <= original.local_limit))
        self.assertTrue(np.all(following <= squared + 2 * limits.acceleration * track.ds + 1e-8))
        self.assertTrue(np.all(squared <= following + 2 * limits.braking * track.ds + 1e-8))
        np.testing.assert_array_equal(original.speed, build_speed_plan(track, limits).speed)

    def test_failed_lap_is_not_accepted(self):
        track = circle()
        limits = Limits()
        with self.assertRaisesRegex(RuntimeError, "TIMEOUT"):
            calibrate_speed_plan(track, build_speed_plan(track, limits), limits,
                                 max_time=0.02)

    def test_exhausted_validation_is_not_accepted(self):
        track = circle()
        limits = Limits(max_speed=5.0)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            calibrate_speed_plan(track, build_speed_plan(track, limits), limits,
                                 max_attempts=1)
