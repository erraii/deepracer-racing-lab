"""Bootstrap reward semantics, using the embedded reference geometry."""

import math
import unittest

from training.reward_bootstrap import RACING_LINE, reward_function


class BootstrapRewardTests(unittest.TestCase):
    def params(self):
        a, b = RACING_LINE[:2]
        return dict(x=(a[0] + b[0]) / 2, y=(a[1] + b[1]) / 2,
                    heading=math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])),
                    track_width=0.762, speed=1.0, all_wheels_on_track=True)

    def test_offtrack_and_crash_receive_minimum(self):
        for flag in ('all_wheels_on_track', 'is_offtrack', 'is_crashed'):
            params = self.params()
            params[flag] = flag != 'all_wheels_on_track'
            self.assertEqual(reward_function(params), 0.001)

    def test_direction_distance_and_speed_preferences(self):
        params = self.params()
        reference_reward = reward_function(params)
        reversed_params = dict(params, heading=params['heading'] + 180)
        self.assertEqual(reward_function(reversed_params), 0.001)
        far_params = dict(params, x=params['x'] + 10, y=params['y'] + 10)
        self.assertLess(reward_function(far_params), reference_reward)
        self.assertGreater(reward_function(dict(params, speed=2)), reference_reward)

    def test_reference_closure_has_finite_rewards(self):
        params = self.params()
        for i in (0, len(RACING_LINE) - 1):
            a, b = RACING_LINE[i], RACING_LINE[(i + 1) % len(RACING_LINE)]
            params.update(x=(a[0] + b[0]) / 2, y=(a[1] + b[1]) / 2,
                          heading=math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
            value = reward_function(params)
            self.assertTrue(math.isfinite(value))
            self.assertGreater(value, 0.8)
