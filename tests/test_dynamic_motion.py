"""Moving cylinders retain their geometry and integrate m/s at 480 Hz."""

import math
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'BaseClass'):
    sys.path.insert(0, str(directory))

from Obstacles.building import building


class DynamicMotionTests(unittest.TestCase):
    def make_obstacle(self, x=250.0, y=250.0, radius=20.0):
        environment = types.SimpleNamespace(len=500, width=500)
        parameters = {'position': {'x': x, 'y': y, 'z': 0.0},
                      '_R': radius, '_H': 50.0}
        return building(parameters, environment)

    def test_static_run_does_not_move_or_change_geometry(self):
        obstacle = self.make_obstacle()
        obstacle.run(1.0 / 480.0)
        self.assertFalse(obstacle.is_dynamic)
        self.assertEqual((obstacle.position.x, obstacle.position.y), (250.0, 250.0))
        self.assertEqual((obstacle._R, obstacle._H), (20.0, 50.0))
        self.assertEqual(obstacle.history, [])

    def test_axis_and_diagonal_speed_are_four_meters_per_second(self):
        for direction in ((1, 0), (0, 1), (1, 1), (-1, -1)):
            with self.subTest(direction=direction):
                obstacle = self.make_obstacle()
                obstacle.configure_motion(direction, speed_m_s=4.0)
                obstacle.run()
                self.assertAlmostEqual(math.hypot(obstacle.position.x - 250,
                                                 obstacle.position.y - 250), 4.0 / 480, places=11)
                for _ in range(479):
                    obstacle.run()
                self.assertAlmostEqual(math.hypot(obstacle.position.x - 250,
                                                 obstacle.position.y - 250), 4.0, places=10)
                self.assertAlmostEqual(math.hypot(obstacle.velocity.x, obstacle.velocity.y), 4.0)
                self.assertEqual(len(obstacle.history), 21)
                self.assertEqual(len(obstacle.motion_path), 21)
                self.assertAlmostEqual(obstacle.history[-1][0], 1.0)
                self.assertEqual((obstacle._R, obstacle._H), (20.0, 50.0))

    def test_oblique_motion_reverses_both_components_on_same_line(self):
        obstacle = self.make_obstacle(x=250, y=200)
        obstacle.configure_motion((2, 1), speed_m_s=4.0)
        obstacle.run(100.0)
        self.assertLess(obstacle.velocity.x, 0)
        self.assertLess(obstacle.velocity.y, 0)
        self.assertAlmostEqual(obstacle.position.y - 200,
                               0.5 * (obstacle.position.x - 250), places=10)
        self.assertTrue(0 <= obstacle.position.x <= 500)
        self.assertTrue(0 <= obstacle.position.y <= 500)
        self.assertAlmostEqual(obstacle.motion_endpoints[0][0], 0.0)
        self.assertAlmostEqual(obstacle.motion_endpoints[1][0], 500.0)
        self.assertAlmostEqual(obstacle.motion_endpoints[0][1], 75.0)
        self.assertAlmostEqual(obstacle.motion_endpoints[1][1], 325.0)

    def test_reflection_retains_overshoot_across_multiple_endpoints(self):
        obstacle = self.make_obstacle(x=5, y=5)
        obstacle.configure_motion((1, 0), speed_m_s=4.0, bounds=(0, 10, 0, 10))
        obstacle.run(7.25)  # Travel 29 m through a 10 m segment.
        self.assertAlmostEqual(obstacle.position.x, 6.0)
        self.assertAlmostEqual(obstacle.position.y, 5.0)
        self.assertEqual(obstacle.velocity.x, -4.0)

    def test_centers_reach_map_boundary_without_radius_projection(self):
        obstacle = self.make_obstacle(x=499.999, y=250, radius=45)
        obstacle.configure_motion((1, 0))
        self.assertAlmostEqual(obstacle.position.x, 499.999)
        self.assertAlmostEqual(obstacle.motion_endpoints[1][0], 500.0)
        obstacle.run()
        self.assertAlmostEqual(obstacle.position.x, 500.0 - (4.0 / 480.0 - 0.001), places=10)
        self.assertEqual(obstacle.velocity.x, -4.0)

    def test_configurable_half_range_and_reset(self):
        obstacle = self.make_obstacle()
        obstacle.configure_motion((1, 0), half_range=2.0)
        self.assertEqual(obstacle.motion_endpoints, [[248.0, 250.0, 0.0], [252.0, 250.0, 0.0]])
        obstacle.run(1.0)
        self.assertAlmostEqual(obstacle.position.x, 250.0)
        self.assertEqual(obstacle.velocity.x, -4.0)
        obstacle.reset_motion()
        self.assertEqual((obstacle.position.x, obstacle.position.y), (250.0, 250.0))
        self.assertEqual(obstacle.elapsed_time, 0.0)
        self.assertEqual(obstacle.velocity.x, 4.0)
        self.assertEqual(obstacle.history, [[0.0, 250.0, 250.0, 0.0]])
        obstacle.clear_motion()
        obstacle.run(20.0)
        self.assertFalse(obstacle.is_dynamic)
        self.assertEqual(obstacle.position.x, 250.0)

    def test_invalid_parameters_are_rejected(self):
        for parameters in ({'direction': (0, 0)},
                           {'direction': (1, 0), 'speed_m_s': -1},
                           {'direction': (1, 0), 'control_dt': 0},
                           {'direction': (1, 0), 'record_hz': 0},
                           {'direction': (1, 0), 'half_range': 0}):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    self.make_obstacle().configure_motion(**parameters)
        obstacle = self.make_obstacle()
        obstacle.configure_motion((1, 0))
        with self.assertRaises(ValueError):
            obstacle.run(float('nan'))


if __name__ == '__main__':
    unittest.main()
