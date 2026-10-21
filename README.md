# DeepRacer Racing Lab

A Python workspace for studying racing lines, speed planning, and path tracking
before testing reward functions in the AWS DeepRacer environment.

The current simulation follows the **Forever Raceway** centerline in the
**counterclockwise (CCW)** direction using a kinematic bicycle model and a
Pure Pursuit steering controller. A reference speed profile accounts for
curvature and independent acceleration and braking limits.

## Project goal

Investigate where a car should drive, how fast it should travel, and whether a
controller can follow the resulting reference. Local experiments help identify
useful training targets and avoid spending DeepRacer training time on basic
geometry or reward-code errors.

The current implementation includes:

- Track loading, boundary checks, and centerline projection.
- Curvature estimation and periodic forward/backward speed planning.
- Pure Pursuit steering and bounded speed changes.
- Animated and headless simulation runs.
- Lap, tracking-error, and lateral-demand measurements.
- Automated checks for the geometry, planner, vehicle, and simulation.

Racing-line optimization and camera-based reinforcement learning remain future
work. The reference path currently comes from the track centerline.

## Requirements

- Python 3.11 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency management.
- NumPy and Matplotlib, installed through the project dependencies.
- A desktop graphical environment for interactive plots and animation.

The automated tests use Python's standard-library `unittest` module.

## Setup

```bash
git clone https://github.com/erraii/deepracer-racing-lab.git
cd deepracer-racing-lab
```

On the 42 Linux machines, load the project environment configuration before
installing dependencies or running commands:

```bash
source scripts/env.sh
uv sync
```

`scripts/env.sh` places the virtual environment, uv cache, temporary files, and
Matplotlib configuration/cache under `/goinfre/$USER/deepracer-racing-lab`.
Source it again when opening a new terminal. If temporary storage is cleared,
run `source scripts/env.sh` followed by `uv sync` to recreate the environment.

On another computer, `uv sync` can create a standard project-local environment
without sourcing the 42-specific script.

Keep `pyproject.toml` and `uv.lock` in Git. Virtual environments and cache
directories should remain excluded.

## Track data

The project uses:

```text
data/tracks/2024_reinvent_champ_ccw.npy
```

Source: [AWS DeepRacer Community Race Data Repository](https://github.com/aws-deepracer-community/deepracer-race-data/blob/main/raw_data/tracks/npy/2024_reinvent_champ_ccw.npy).

If the file is missing, download it from the project root:

```bash
mkdir -p data/tracks
curl -fL --retry 3 \
  "https://raw.githubusercontent.com/aws-deepracer-community/deepracer-race-data/main/raw_data/tracks/npy/2024_reinvent_champ_ccw.npy" \
  -o data/tracks/2024_reinvent_champ_ccw.npy
```

Each row contains three coordinate pairs: the centerline and two track
boundaries. The loader identifies the outer and inner boundary polygons by
their enclosed areas and checks the centerline's CCW ordering.

| Property | Value |
| --- | --- |
| Source array shape | 86 rows, 6 columns |
| Source centerline length | Approximately 25.107 m |
| Shared sampled centerline length | Approximately 25.086 m |
| Median track width | Approximately 0.762 m |
| Direction | Counterclockwise |

The centerline is resampled at approximately 5 cm intervals. Connecting the
resampled points can shorten the polyline slightly at source corners. Both
inspection and driving use the same sampled geometry and distance coordinate.
The original boundary polygons and source file are preserved.

## Run the project

Run these commands from the project root.

### Inspect the track and reference speed profile

```bash
uv run python inspect_track.py
```

The figure shows curvature magnitude on the track, signed curvature along the
lap, and the local speed limits together with the planned speed profile.

Print the calculations without opening a window:

```bash
uv run python inspect_track.py --no-show
```

### Run an animated lap

Use the current working lookahead setting explicitly:

```bash
uv run python main.py --lookahead 0.6
```

The display shows the car, steering target, driven path, actual and target
speeds, steering angle, acceleration, and tracking error. The simulation stops
after a completed lap, a track exit, or a timeout. The window remains open.

### Run without an animation

```bash
uv run python main.py --headless --lookahead 0.6
```

On a machine without a graphical display, select a noninteractive Matplotlib
backend as well:

```bash
MPLBACKEND=Agg uv run python main.py --headless --lookahead 0.6
```

### Compare with a constant-speed command

```bash
uv run python main.py --headless --mode constant --speed 2 --lookahead 0.6
```

Both modes use the same vehicle and steering implementation. Planned mode
starts at the reference speed at the start line; constant mode starts at the
requested constant speed. These are rolling-start experiments.

### Available simulation options

| Option | Default | Meaning |
| --- | --- | --- |
| `--track` | Forever Raceway data path | Alternative six-column CCW track file |
| `--headless` | Disabled | Run without opening an animation |
| `--mode` | `planned` | `planned` or `constant` speed command |
| `--speed` | `2.0` | Constant-mode speed in m/s |
| `--lookahead` | `0.8` | Steering lookahead in meters |
| `--dt` | `0.02` | Simulation time step in seconds |
| `--max-time` | `60.0` | Timeout in simulation seconds |

The code default remains **0.8 m**. The commands above explicitly select
**0.6 m**, the current working setting based on the comparison below.

## How the simulation works

1. Load and validate the track, then sample its centerline.
2. Estimate signed curvature from three points, looking approximately 0.25 m
   backward and forward along the sampled path.
3. Calculate a local speed cap using lateral acceleration and maximum speed.
4. Apply forward acceleration and backward braking passes until the closed-lap
   profile converges.
5. Project the vehicle onto the centerline to obtain its current path position.
6. Use Pure Pursuit to steer toward a point ahead on that line.
7. Move the vehicle toward the reference speed with bounded speed changes.
8. Update lap progress, track-exit status, and run measurements.

For a path with curvature magnitude `|kappa|`, the lateral planning relationship
is `a_y = v^2 * |kappa|`. The corresponding local speed cap is
`min(v_max, sqrt(a_y_max / |kappa|))`, with numerical protection near zero
curvature.

Longitudinal planning uses `v_next^2 <= v^2 + 2 * a_acc * ds` and
`v^2 <= v_next^2 + 2 * a_brake * ds`. The end of the lap connects back to the
start. Squared speed is interpolated between reference samples to preserve
these spatial acceleration bounds.

The vehicle integrates distance using average speed over the time step and
follows an analytic arc for the applied constant steering angle.

## Parameters

| Parameter | Current value | Location |
| --- | --- | --- |
| Maximum reference speed | 3.0 m/s | `speed_planner.py` |
| Lateral planning acceleration | 3.0 m/s^2 | `speed_planner.py` |
| Maximum acceleration | 1.0 m/s^2 | `speed_planner.py` |
| Maximum braking magnitude | 2.0 m/s^2 | `speed_planner.py` |

These four parameters are defined together in `Limits` in `speed_planner.py`.

| Vehicle / geometry parameter | Current value | Location |
| --- | --- | --- |
| Wheelbase | 0.33 m | `vehicle.py` |
| Maximum steering angle | 30 degrees | `vehicle.py` |
| Requested sampling interval | 0.05 m | `track.py` |
| Curvature estimation offset | 0.25 m | `speed_planner.py` |

All vehicle and acceleration values are illustrative and uncalibrated. They
should not be interpreted as measured DeepRacer specifications.

## Current results

The shared reference calculation gives an estimated centerline lap time of
approximately **14.07 s**, with reference speeds from **1.22 to 2.92 m/s**.

The following runs used the same reference profile, a rolling start, and a
0.02 s simulation time step:

| Lookahead | Status | Lap time | Distance driven | Maximum center error | Maximum lateral demand | Time above 3.0 m/s^2 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.4 m | Completed | 13.92 s | 24.907 m | 0.049 m | 3.216 m/s^2 | 0.44 s |
| 0.6 m | Completed | 13.76 s | 24.687 m | 0.095 m | 2.992 m/s^2 | 0.00 s |
| 0.8 m | Completed | 13.60 s | 24.462 m | 0.145 m | 3.436 m/s^2 | 0.36 s |

**0.6 m is the current working setting:** it completed the tested lap without
exceeding the chosen lateral planning limit. Its peak demand is close to that
limit, so this result does not establish a robust operating margin or physical
feasibility.

Shorter lookahead follows the centerline more closely but can demand stronger
steering corrections. Longer lookahead cuts corners more noticeably. A shorter
recorded lap therefore does not automatically mean a better feasible racing
line. The centerline reference time and driven lap time also refer to different
paths.

These values describe the current simplified simulation. Changes to sampling,
integration, control, or planning parameters require a fresh comparison. The
earlier 12.28 s constant-speed result used a previous implementation and is not
an exact regression target for this version.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

The current suite has seven tests covering:

- Circle curvature, path projection, and boundary membership.
- Periodic speed caps, acceleration/braking constraints, and interpolation.
- Bounded speed changes and integrated distance.
- Constant-steering circular motion.
- Track closure and invalid source ordering.
- Completed, off-track, and timeout simulation termination.
- Invalid vehicle and planning limits.

These checks passed in the current version. They verify behavior under the
implemented assumptions; they do not validate the model against a real car.

## Model limitations

- **Tire grip:** the vehicle has no tire-slip dynamics or combined friction
  ellipse. Longitudinal and lateral planning limits are independent.
- **Actual lateral demand:** the controller follows a different curve from the
  reference and speed tracking can lag. Actual demand can exceed the planning
  limit; the run report measures the peak and time above it.
- **Vehicle footprint:** track-exit checks cover only the rear-axle midpoint.
  Wheels and body corners are not checked.
- **Steering:** the angle is limited, but steering rate and actuator delay are
  not modeled.
- **Geometry:** curvature depends on source sampling and the estimation window.
  The sampled reference is a polyline, rather than a smooth optimized spline.
- **Timing:** runs start already moving, and stop times have the resolution of
  the selected time step. OFF TRACK and TIMEOUT durations are not completed-lap
  times.
- **Learning and transfer:** this controller uses known coordinates directly.
  There is no camera perception or learned policy in this local simulator, and
  no demonstrated equivalence to DeepRacer dynamics or competition timing.

## Project files

| Path | Purpose |
| --- | --- |
| `main.py` | Animation and headless simulation command line |
| `track.py` | Shared geometry, sampling, projection, and boundaries |
| `speed_planner.py` | Curvature and periodic speed planning |
| `vehicle.py` | Kinematic bicycle and bounded speed changes |
| `simulation.py` | Steering, reference-speed tracking, and run metrics |
| `inspect_track.py` | Shared track and speed-profile inspection |
| `tests/test_models.py` | Automated checks |
| `scripts/env.sh` | Environment paths for 42 Linux |
| `data/tracks/2024_reinvent_champ_ccw.npy` | Source track geometry |
| `pyproject.toml` / `uv.lock` | Dependencies and reproducible environment |

This README consolidates the setup and model notes previously included in
`UPDATE_README.md`; that temporary document can be removed after this README
is installed.

## Next steps

- Measure or calibrate vehicle limits against DeepRacer simulation logs.
- Add vehicle-footprint and combined-grip checks.
- Compare feasible racing lines and their associated speed profiles.
- Use promising references to design a tolerant DeepRacer reward function.
- Validate the resulting training design against a simple baseline under the
  same DeepRacer training budget.

## References

- [Project repository](https://github.com/erraii/deepracer-racing-lab)
- [DeepRacer track data](https://github.com/aws-deepracer-community/deepracer-race-data)
- [K1999 racing-line experiments](https://github.com/cdthompson/deepracer-k1999-race-lines)
- [TUM global racing trajectory optimization](https://github.com/TUMFTM/global_racetrajectory_optimization)
- [DeepRacer-for-Cloud](https://github.com/aws-deepracer-community/deepracer-for-cloud)
- [Pure Pursuit example in PythonRobotics](https://atsushisakai.github.io/PythonRobotics/modules/6_path_tracking/pure_pursuit_tracking/pure_pursuit_tracking.html)