## Current simulation

A 2D kinematic bicycle model follows the Forever Raceway
centerline counterclockwise using Pure Pursuit steering.

Baseline:
- Speed: 2.0 m/s
- Lookahead: 0.8 m
- Simulation lap time: 12.28 s
- Maximum centerline deviation: 0.146 m

Limitations:
- No tire slip or acceleration/braking limits.
- Track-exit detection checks only the rear-axle midpoint.
- Results are not validated against DeepRacer vehicle dynamics.

## Track data

Source:
https://github.com/aws-deepracer-community/deepracer-race-data

File:
raw_data/tracks/npy/2024_reinvent_champ_ccw.npy

The original data is preserved. The simulation resamples it
to approximately 5 cm spacing.