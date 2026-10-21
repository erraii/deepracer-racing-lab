# First 10-minute training trial

Use the console link and credentials supplied by the competition organizer.
The original managed AWS DeepRacer console service ended on December 15, 2025;
this guide is for the organizer's training environment.

Proposed settings for a short first experiment (not competition requirements):

| Setting | Value |
| --- | --- |
| Model name | `forever-ccw-bootstrap-v01` |
| Race | Time trial |
| Track | Forever Raceway |
| Direction | Counterclockwise, fixed |
| Sensor | Camera |
| Algorithm | PPO |
| Hyperparameters | Leave the organizer's defaults unchanged |
| Action space | Continuous |
| Speed range | 0.8 to 2.0 m/s |
| Steering range | -30 to +30 degrees |
| Minimum evaluation trials | 1 initially |
| Training stop condition | Maximum time: 10 minutes |
| Reward | Entire contents of `training/reward_bootstrap.py` |

If the console restricts these values or requires another configuration, check
the displayed limits before starting. Do not change race eligibility settings.
The initial action space is deliberately conservative; increasing its speed
range later may require a new model rather than continuing an existing one.

## Start

1. Open the organizer's console and choose Create a Model.
2. Set the model name, time trial, Forever Raceway, and counterclockwise direction.
3. Select Camera, PPO, default hyperparameters, and the action ranges above.
4. Pick any cosmetic car body/color allowed by the event.
5. Paste the entire Python reward file into the reward editor; click Validate.
6. Set the maximum training duration to 10 minutes and review all settings.
7. Start training. Provisioning and evaluation can add wall-clock time;
   do not assume that ten minutes means a ten-minute total browser session.
8. After training, run a small evaluation if it is not already automatic.
   Inspect the console's quota accounting; evaluation overhead is environment-specific.

The organizer's supplied PDF recommends initial tests of 10-15 minutes and
states that each driver has 10 sponsored training hours. Verify your remaining
quota on the live profile. No separate personal AWS deployment is needed.

## What this reward does

It embeds a roughly 0.10 m sampled version of the uploaded optimized reference,
then rewards proximity and alignment smoothly. It penalizes off-track/crash
states and adds a small speed preference within the initial action range.
It uses only Python's math library and is stateless. No external `.npz`, NumPy,
or project files need to be uploaded alongside it.

This is a bootstrap reward, not a minimum-time objective. It neither enforces
the local simulator's speed profile nor guarantees tire grip or lap completion.
It should not be used on a different track, direction, or coordinate system.
If the optimized reference changes, the embedded coordinates must be rebuilt.

The simulator supplies position and heading to calculate the reward during
training. With the Camera configuration, the policy learns from camera images;
it does not receive the local Pure Pursuit controller or reference coordinates
as direct steering commands.

## Record results

Record model name, settings, reward file/commit, actual training duration and
quota used, reward/completion graphs, and any evaluation off-track counts,
completion percentages and lap times. If no lap completes, retain the partial
progress measurements. A ten-minute trial can expose configuration problems
and early learning, but does not establish final model quality.

Do not compare its timing directly to the 10.72 s kinematic-controller result:
the trained camera policy, action range, and simulator dynamics are different.

Reference documentation:

- https://docs.aws.amazon.com/solutions/latest/deepracer-on-aws/create-a-model.html
- https://docs.aws.amazon.com/solutions/latest/deepracer-on-aws/train-and-evaluate-models.html
- https://aws.amazon.com/deepracer/faqs/
