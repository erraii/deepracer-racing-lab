import tempfile
from pathlib import Path
import unittest

import numpy as np

from test_models import circle
from optimize_raceline import SplineProblem, load_reference, optimize, save_reference
from raceline_geometry import validate_reference


def approximate_jacobian(function, values, step=1e-6):
    columns = []
    for index in range(len(values)):
        delta = np.zeros_like(values)
        delta[index] = step
        columns.append((np.atleast_1d(function(values + delta))
                        - np.atleast_1d(function(values - delta))) / (2 * step))
    return np.column_stack(columns)


class RacelineTests(unittest.TestCase):
    def test_analytic_objective_and_constraint_gradients(self):
        problem = SplineProblem(circle(count=80), control_spacing=1.0, sample_spacing=0.1)
        offsets = np.random.default_rng(42).uniform(-0.12, -0.05, problem.count)
        _, analytical = problem.objective(offsets)
        numerical = approximate_jacobian(lambda values: problem.objective(values)[0], offsets).ravel()
        np.testing.assert_allclose(analytical, numerical, atol=1e-5, rtol=1e-4)
        numerical_constraints = approximate_jacobian(problem.constraints, offsets)
        np.testing.assert_allclose(problem.constraints(offsets, jacobian=True), numerical_constraints,
                                   atol=2e-5, rtol=1e-4)

    def test_full_segments_not_just_vertices(self):
        road = circle(count=240)
        angle = np.linspace(0, 2 * np.pi, 9)
        candidate = 2.6 * np.column_stack((np.cos(angle), np.sin(angle)))
        candidate[-1] = candidate[0]
        with self.assertRaises(ValueError):
            validate_reference(road, candidate, 0.05)

    def test_optimized_candidate_and_source_binding(self):
        road = circle(count=80)
        reference, result, problem, clearance = optimize(road, quiet=True)
        self.assertTrue(result.success)
        self.assertLess(problem.raw_objective(result.x)[0], problem.scale)
        self.assertGreaterEqual(clearance, 0.15 - 1e-7)
        self.assertTrue(np.allclose(reference.center[0], road.center[0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"
            save_reference(path, road, reference, 0.15, float(result.fun))
            loaded = load_reference(path, road)
            np.testing.assert_allclose(loaded.center, reference.center)
            with self.assertRaises(ValueError):
                load_reference(path, circle(radius=4))

    def test_margin_rejected(self):
        with self.assertRaises(ValueError):
            SplineProblem(circle(), margin=0.5)


if __name__ == "__main__":
    unittest.main()
