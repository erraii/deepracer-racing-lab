"""Constrained periodic-spline curvature optimization; a local candidate search."""

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from raceline_geometry import Corridor, validate_reference
from speed_planner import Limits, build_speed_plan
from track import DEFAULT_TRACK, Track, load_track


DEFAULT_RACELINE = Path(__file__).resolve().parent / "data/racelines/forever_mincurv.npz"


def track_digest(road):
    h = hashlib.sha256()
    for array in (road.center, road.outer, road.inner):
        h.update(np.asarray(array, dtype="<f8").tobytes())
    return h.hexdigest()


class SplineProblem:
    def __init__(self, road, margin=0.15, control_spacing=0.45, sample_spacing=0.04):
        if not np.isfinite([margin, control_spacing, sample_spacing]).all():
            raise ValueError("Optimizer settings must be finite.")
        if margin <= 0 or control_spacing <= 0 or sample_spacing <= 0:
            raise ValueError("Optimizer settings must be positive.")
        if margin + 0.004 >= road.width / 2:
            raise ValueError("Boundary margin leaves no usable track width.")
        self.road, self.margin = road, margin
        self.corridor = Corridor(road)
        self.count = max(12, int(np.ceil(road.length / control_spacing)))
        knots = np.linspace(0, road.length, self.count + 1)
        nodes = np.array([road.position(s) for s in knots[:-1]])
        self.knots, self.nodes = knots, nodes
        base = CubicSpline(knots, np.vstack((nodes, nodes[0])), bc_type="periodic")
        tangent = base(knots[:-1], 1)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        self.normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        eye = np.eye(self.count)
        self.basis = CubicSpline(knots, np.vstack((eye, eye[0])), bc_type="periodic")
        self.sample_count = max(300, int(np.ceil(road.length / sample_spacing)))
        self.u = np.linspace(0, road.length, self.sample_count, endpoint=False)
        self.matrices = [self.basis(self.u, order) for order in range(3)]
        self.base = [matrix @ nodes for matrix in self.matrices]
        self.mx = [matrix * self.normals[:, 0] for matrix in self.matrices]
        self.my = [matrix * self.normals[:, 1] for matrix in self.matrices]
        reference_tangent = self.base[1]
        self.reference_tangent = reference_tangent / np.linalg.norm(reference_tangent, axis=1)[:, None]
        self.scale = max(self.raw_objective(np.zeros(self.count))[0], 1e-6)

    def evaluate(self, offsets):
        return [base + np.column_stack((mx @ offsets, my @ offsets))
                for base, mx, my in zip(self.base, self.mx, self.my)]

    def raw_objective(self, offsets):
        _, first, second = self.evaluate(offsets)
        dx, dy = first.T
        ddx, ddy = second.T
        q = np.maximum(dx**2 + dy**2, 1e-10)
        numerator = dx * ddy - dy * ddx
        dn = (self.mx[1] * ddy[:, None] + self.my[2] * dx[:, None]
              - self.my[1] * ddx[:, None] - self.mx[2] * dy[:, None])
        dq = 2 * (self.mx[1] * dx[:, None] + self.my[1] * dy[:, None])
        # Integral kappa^2 ds = integral numerator^2 / q^(5/2) du.
        weight = self.road.length / self.sample_count
        value = float(np.sum(numerator**2 / q**2.5) * weight)
        gradient = np.sum(
            2 * numerator[:, None] * dn / q[:, None]**2.5
            - 2.5 * numerator[:, None]**2 * dq / q[:, None]**3.5,
            axis=0,
        ) * weight
        return value, gradient

    def objective(self, offsets):
        value, gradient = self.raw_objective(offsets)
        # Tiny regularizer resolves otherwise nearly equivalent lateral shifts.
        return (value / self.scale + 1e-5 * float(np.mean(offsets**2)),
                gradient / self.scale + 2e-5 * offsets / self.count)

    def constraints(self, offsets, jacobian=False):
        position, first, _ = self.evaluate(offsets)
        clearance, gradient = self.corridor.signed_clearance(position)
        # A small collocation reserve is followed by full-segment validation.
        values = clearance - self.margin - 0.004
        progress = np.sum(first * self.reference_tangent, axis=1) - 0.1
        if not jacobian:
            return np.concatenate((values, progress))
        boundary_jac = gradient[:, 0, None] * self.mx[0] + gradient[:, 1, None] * self.my[0]
        progress_jac = (self.reference_tangent[:, 0, None] * self.mx[1]
                        + self.reference_tangent[:, 1, None] * self.my[1])
        return np.vstack((boundary_jac, progress_jac))

    def dense_curve(self, offsets, spacing=0.015):
        sample_count = max(400, int(np.ceil(self.road.length / spacing)))
        u = np.linspace(0, self.road.length, sample_count + 1)
        nodes = self.nodes + self.normals * offsets[:, None]
        spline = CubicSpline(self.knots, np.vstack((nodes, nodes[0])), bc_type="periodic")
        center = spline(u)
        center[-1] = center[0]
        return center


def optimize(road, margin=0.15, max_iterations=200, quiet=False):
    problem = SplineProblem(road, margin)
    initial = np.zeros(problem.count)
    bounds = [(-road.width / 2, road.width / 2)] * problem.count
    bounds[0] = (0.0, 0.0)  # Same start-line position for comparisons.
    counter = [0]

    def progress(offsets):
        counter[0] += 1
        if not quiet and counter[0] % 10 == 0:
            print(f"Iteration {counter[0]}: normalized objective {problem.objective(offsets)[0]:.5f}", flush=True)

    result = minimize(
        problem.objective, initial, method="SLSQP", jac=True, bounds=bounds,
        constraints={"type": "ineq", "fun": problem.constraints,
                     "jac": lambda offsets: problem.constraints(offsets, jacobian=True)},
        options={"maxiter": max_iterations, "ftol": 1e-7, "disp": False},
        callback=progress,
    )
    if not result.success:
        raise RuntimeError(f"Optimizer did not converge; no file saved: {result.message}")
    if problem.constraints(result.x).min() < -1e-6:
        raise RuntimeError("Optimizer returned an infeasible candidate; no file saved.")
    if problem.raw_objective(result.x)[0] > problem.scale + 1e-7:
        raise RuntimeError("Candidate did not improve the spline baseline; no file saved.")
    center = problem.dense_curve(result.x)
    clearance = validate_reference(road, center, margin)
    reference = Track(center, road.outer, road.inner, road.original_length, road.width)
    return reference, result, problem, clearance


def save_reference(path, road, reference, margin, objective):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Reference output must have the .npz extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, center=reference.center,
                        road_digest=np.array(track_digest(road)), margin=np.array(margin),
                        objective=np.array(objective), format_version=np.array(1))


def load_reference(path, road):
    with np.load(path, allow_pickle=False) as data:
        if int(data["format_version"]) != 1:
            raise ValueError("Unsupported reference format.")
        if str(data["road_digest"].item()) != track_digest(road):
            raise ValueError("Reference belongs to a different road or sampling. Regenerate it.")
        center = data["center"].copy()
        margin = float(data["margin"])
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("Invalid saved boundary margin.")
    validate_reference(road, center, margin)
    return Track(center, road.outer, road.inner, road.original_length, road.width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default=DEFAULT_TRACK)
    parser.add_argument("--output", default=DEFAULT_RACELINE)
    parser.add_argument("--margin", type=float, default=0.15,
                        help="Reference-point distance from track boundaries [m].")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    road = load_track(args.track)
    print("Searching a locally minimum-curvature periodic spline...", flush=True)
    print(f"Reference-point boundary margin: {args.margin:.3f} m", flush=True)
    reference, result, problem, clearance = optimize(road, args.margin, args.max_iterations)
    limits = Limits()
    original_plan, candidate_plan = build_speed_plan(road, limits), build_speed_plan(reference, limits)
    objective = problem.raw_objective(result.x)[0]
    save_reference(args.output, road, reference, args.margin, objective)
    print(f"Optimizer iterations: {result.nit}")
    print(f"Spline curvature cost: {problem.scale:.4f} -> {objective:.4f}")
    print(f"Centerline length: {road.length:.3f} m")
    print(f"Candidate length: {reference.length:.3f} m")
    print(f"Minimum exported-polyline boundary clearance: {clearance:.3f} m")
    print(f"Centerline reference estimate: {original_plan.lap_time:.2f} s")
    print(f"Candidate reference estimate: {candidate_plan.lap_time:.2f} s")
    print(f"Saved: {args.output}")
    print("Local curvature objective, not a guaranteed minimum-time/global optimum.")
    print("Margin applies to the reference point; vehicle footprint and tracking error still need testing.")
    if args.no_show:
        return
    fig, (ax_path, ax_speed) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    ax_path.fill(*road.outer.T, color="#555555")
    ax_path.fill(*road.inner.T, color="#b8d8a5")
    ax_path.set_facecolor("#b8d8a5")
    ax_path.plot(*road.center.T, "--", color="gold", label="Centerline")
    ax_path.plot(*reference.center.T, color="black", linewidth=2, label="Optimized reference")
    ax_path.plot(*road.center[0], "rx", label="Fixed start")
    ax_path.set(aspect="equal", xlabel="x [m]", ylabel="y [m]", title="Minimum-curvature candidate")
    ax_path.legend()
    for path, plan, label, color in ((road, original_plan, "Centerline", "goldenrod"),
                                      (reference, candidate_plan, "Candidate", "black")):
        ax_speed.plot(path.s / path.length * 100, np.r_[plan.speed, plan.speed[0]], label=label, color=color)
    ax_speed.set(xlabel="Progress along each reference [%]", ylabel="Planned speed [m/s]",
                 title="Recomputed speed plans (progress is normalized separately)")
    ax_speed.grid(alpha=0.3)
    ax_speed.legend()
    plt.show()


if __name__ == "__main__":
    main()
