# Shared track and speed planning update

Copy these Python files into the existing `deepracer-racing-lab` project root.
Replace `main.py`, `inspect_track.py`, and `vehicle.py`; add `track.py`,
`speed_planner.py`, `simulation.py`, and `tests/test_models.py`.
Keep your existing README, pyproject.toml, uv.lock, scripts/env.sh, and track data.
The archive intentionally does not contain the original track binary.
Dependencies remain NumPy and Matplotlib. Tests use standard-library unittest.

## Run from the project root

```bash
source scripts/env.sh
uv run python -m unittest discover -s tests -v
uv run python inspect_track.py --no-show
uv run python main.py --headless
uv run python main.py
```

`--headless` runs the same simulation without a graphical window. On a machine
without a display, use `MPLBACKEND=Agg uv run python main.py --headless`.
Inspect the plots with `uv run python inspect_track.py`.

Compare planned and constant commands with the same updated vehicle/controller:

```bash
uv run python main.py --headless --mode planned
uv run python main.py --headless --mode constant --speed 2
```

Both modes start already moving. The planned mode starts at the reference speed
at the start line. End time is quantized to the time step (default 0.02 s).
Never interpret an OFF TRACK or TIMEOUT duration as a completed lap time.

## Design

- `track.py`: one sampled centerline, projection, interpolation and original
  boundary polygons for both plotting and driving. Sampled chord length is used
  consistently in planning, projection and lap counting. Original and sampled
  lengths are printed separately, so reference estimates may change slightly.
- `speed_planner.py`: three-point curvature estimate, independent lateral/
  longitudinal limits, periodic forward/backward passes and numeric checks.
  Speed squared is interpolated between profile samples.
- `vehicle.py`: rear-axle kinematic bicycle, bounded acceleration/braking,
  distance integration using average speed, analytic constant-steering arc.
- `simulation.py`: continuous projection for Pure Pursuit targeting, speed
  commands, lap progress, tracking error and lateral-demand diagnostics.
- `main.py`: animation (including driven trail) and a headless CLI.
- `inspect_track.py`: inspection plots using the same geometry and planner.

The new integrator, continuous projection and original boundary polygons are
small intentional changes. Re-run the constant-speed baseline with this version;
the old 12.28 s result is not an exact regression target.

## Limits and interpretation

Defaults: 3 m/s maximum speed, 3 m/s^2 lateral planning limit,
1 m/s^2 acceleration, 2 m/s^2 braking, 0.8 m steering lookahead.
These are illustrative, uncalibrated parameters, not DeepRacer specifications.
Edit the single `Limits` class in speed_planner.py to change planning defaults.

The vehicle does not enforce a tire friction limit or simulate sliding.
Speed changes are bounded, but speed tracking can lag and Pure Pursuit follows
a different curve from the centerline. Consequently, actual lateral demand can
exceed the planning limit. The run reports its maximum and time above the limit;
a completed point-in-track lap is not proof of physical feasibility.
There is no combined friction ellipse, steering-rate limit, camera policy or RL.
Boundary checks cover only the rear-axle midpoint, not wheels or the vehicle body.
Curvature still depends on the 0.25 m estimation window and source sampling.

Track source (existing file):
https://github.com/aws-deepracer-community/deepracer-race-data/blob/main/raw_data/tracks/npy/2024_reinvent_champ_ccw.npy

## Commit after checking the local run

```bash
git add main.py inspect_track.py vehicle.py track.py speed_planner.py simulation.py tests/test_models.py UPDATE_README.md
git commit -m "Share track planning and track speed with bounded acceleration"
git push
```

This package changes local project files only; it does not push to GitHub.
