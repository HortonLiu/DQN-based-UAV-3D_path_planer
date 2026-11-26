"""Verify the opt-in manuscript controller at navigation and physical time scales."""

import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "Agents", ROOT / "BaseClass"):
    sys.path.insert(0, str(directory))

from BaseClass.CalMod import Loc, XML2Dict

UAV = importlib.import_module("Agents.UAV_Single_RL+VOAPF S2R").UAV


class Environment:
    len = 500
    width = 500
    h = 100

    def __init__(self):
        self.buildings = []
        self.blocked = False

    def Threaten_rate(self, position):
        return int(self.blocked)


class ManuscriptControllerTests(unittest.TestCase):
    def make_agent(self, hardware_hz=0):
        params = XML2Dict(str(ROOT / "config/UAV.xml"))["Agent"]
        params.update(name="manuscript_test", j=0, Hardware_Hz_Enabled=hardware_hz,
                      Max_V=1, controller_profile="manuscript")
        env = Environment()
        with patch.object(UAV, "Init_Record_Mod", return_value=None):
            agent = UAV(params, env)
        agent.reset(scene=SimpleNamespace(start=(50, 50), goal=(150, 50), heading_rad=0.0))
        return agent, env

    def set_stagnant(self, agent):
        agent.Control_step = agent.stuck_step_limit * agent.controls_per_decision
        agent.sim_time = agent.stuck_window_seconds
        for _ in range(agent.stuck_history_samples):
            agent.stuck_history.append(Loc(50, 50, 0))

    def test_profile_keeps_one_decision_per_second_and_900_step_base(self):
        for hardware in (0, 1):
            with self.subTest(hardware=hardware):
                agent, _ = self.make_agent(hardware)
                self.assertEqual(agent.controls_per_decision, 480 if hardware else 1)
                self.assertEqual(agent.base_duration_s, 900)
                self.assertEqual(agent.original_max_step, 900 * 2 * agent.control_hz)
                self.assertEqual(agent.bonus_duration_s, 50)
                self.assertEqual(agent.stuck_window_seconds, 10)
                self.assertEqual(agent.Scan_Radius, 5.6)

    def test_shared_scene_sets_start_goal_heading_without_random_draws(self):
        agent, _ = self.make_agent()
        scene = SimpleNamespace(start=(20, 8), goal=(420, 450), heading_rad=math.pi / 2)
        with patch("random.uniform", side_effect=AssertionError("Unexpected scene draw")):
            agent.reset(scene=scene)
        self.assertEqual((agent.position.x, agent.position.y, agent.position.z), (20, 8, 0))
        self.assertEqual((agent.goal.x, agent.goal.y, agent.goal.z), (420, 450, 0))
        self.assertAlmostEqual(agent.V_vector.x, 0.0)
        self.assertAlmostEqual(agent.V_vector.y, 1.0)

    def test_stagnation_uses_nearest_midpoint_and_extends_once(self):
        agent, env = self.make_agent()
        env.buildings = [SimpleNamespace(position=Loc(56, 50, 0), _R=1),
                         SimpleNamespace(position=Loc(56, 50, 0), _R=2)]
        self.set_stagnant(agent)
        agent._update_virtual_sources()
        (source,) = agent.active_virtual_obstacles
        self.assertEqual((source["pos"].x, source["pos"].y), (52, 50))
        self.assertEqual(source["force"], 10)
        self.assertEqual(agent.Max_Step, agent.original_max_step + 100)
        agent._update_virtual_sources()
        self.assertEqual(source["force"], 10)
        for _ in range(20):
            agent.Control_step += agent.controls_per_decision
            agent._update_virtual_sources()
        self.assertEqual(source["force"], 40)
        self.assertEqual(len(agent.active_virtual_obstacles), 1)
        self.assertEqual(agent.Max_Step, agent.original_max_step + 100)

    def test_no_midpoint_for_wall_or_out_of_range_obstacle(self):
        agent, env = self.make_agent()
        env.buildings = [SimpleNamespace(position=Loc(58, 50, 0), _R=2)]
        self.set_stagnant(agent)
        agent._update_virtual_sources()
        self.assertEqual(agent.active_virtual_obstacles, [])
        self.assertEqual(agent.Max_Step, agent.original_max_step)
        env.buildings = []
        agent.position, agent.goal = Loc(2, 50, 0), Loc(-10, 50, 0)
        self.assertEqual(agent.get_ray_hit_distance(agent.position, Loc(-1, 0, 0)), 2)
        self.assertIsNone(agent._goal_ray_midpoint())

    def test_reciprocal_force_is_regularized_and_finite_range(self):
        agent, _ = self.make_agent()
        source = {"pos": Loc(52, 50, 0), "force": 10}
        agent.active_virtual_obstacles = [source]
        force = agent.cal_force(agent.position, k_att=0)
        self.assertAlmostEqual(force.x, -10 * (1 / 2 - 1 / 10)**2 * (2 / 2.001))
        for x in (50, 60, 65):
            with self.subTest(x=x):
                source["pos"] = Loc(x, 50, 0)
                force = agent.cal_force(agent.position, k_att=0)
                self.assertAlmostEqual(force.x, 0.0)
                self.assertAlmostEqual(force.y, 0.0)

    def test_observation_preserves_126_values_with_altitude_as_fifth(self):
        agent, _ = self.make_agent()
        agent.position = Loc(50, 50, 2)
        agent.goal = Loc(150, 250, 10)
        agent.V_vector = Loc(0.6, 0.8, 0)
        state = agent.state_PathPlan()
        self.assertEqual(len(state), 126)
        self.assertEqual(list(state[:5]), [0.2, 0.4, 0.6, 0.8, 0.02])

    def test_observation_ignores_solid_ray_hits_beyond_scan_range(self):
        agent, env = self.make_agent()
        last_forward_cell = 10 * 11 + 5
        env.buildings = [SimpleNamespace(position=Loc(57.25, 50, 0), _R=2)]
        self.assertEqual(agent._sample_lidar_grid()[last_forward_cell], 1.0)
        env.buildings[0].position.x = 58
        self.assertEqual(agent._sample_lidar_grid()[last_forward_cell], 0.0)

    def test_hardware_scan_uses_683_heading_centered_rays_and_range(self):
        agent, _ = self.make_agent(hardware_hz=1)
        agent.V_vector = Loc(0, 1, 0)
        with patch.object(agent, "_scan_hit_distances", wraps=agent._scan_hit_distances) as intersections:
            grid = agent._sample_lidar_grid().reshape((11, 11))
        (directions,) = intersections.call_args.args
        self.assertEqual(directions.shape, (683, 2))
        np.testing.assert_allclose(directions[[0, 341, -1]],
                                   [[math.sqrt(3) / 2, -0.5], [0, 1], [-math.sqrt(3) / 2, -0.5]],
                                   atol=1e-12)
        self.assertEqual(grid[5, 9], 0.0)  # observed ahead
        self.assertEqual(grid[5, 1], -1.0)  # outside the field of view
        self.assertEqual(grid[10, 10], -1.0)  # beyond 5.6 m

    def test_hardware_scan_assigns_modal_edges_occlusion_and_ties(self):
        agent, env = self.make_agent(hardware_hz=1)
        env.buildings = [SimpleNamespace(position=Loc(52.25, 50, 0), _R=0.65)]
        grid = agent._sample_lidar_grid().reshape((11, 11))
        self.assertEqual(grid[7, 5], 1.0)
        self.assertEqual(grid[8, 5], -1.0)
        with patch("numpy.linspace", return_value=np.array([0.0, 0.0])), \
             patch.object(agent, "_scan_hit_distances", return_value=np.array([1.0, np.inf])):
            tied = agent._sample_lidar_grid().reshape((11, 11))
        self.assertEqual(tied[6, 5], -1.0)

    def test_hardware_scan_holds_observation_until_next_10_hz_sample(self):
        agent, env = self.make_agent(hardware_hz=1)
        agent._refresh_sensors()
        initial = agent.cached_lidar_grid
        env.buildings = [SimpleNamespace(position=Loc(52.0, 50, 0), _R=0.7)]
        agent.Control_step = 47
        agent._refresh_sensors()
        self.assertIs(agent.cached_lidar_grid, initial)
        self.assertEqual(agent.lidar_updates, 1)
        agent.Control_step = 48
        agent._refresh_sensors()
        self.assertFalse(np.array_equal(agent.cached_lidar_grid, initial))
        self.assertEqual(agent.lidar_updates, 2)

    def test_collision_replaces_substep_reward_and_overrides_timeout(self):
        agent, env = self.make_agent(hardware_hz=1)
        agent.set_episode_duration(2 / 480)
        self.assertEqual(agent.update([0, 0, 0]), (0.0, False, "normal"))
        env.blocked = True
        self.assertEqual(agent.update([0, 0, 0]), (-100.0, True, "meet_threaten"))
        self.assertEqual(agent.collision_count, 1)
        self.assertEqual(agent.total_score, -100.0)

    def test_goal_reward_uses_original_limit_and_strict_distance(self):
        agent, _ = self.make_agent()
        agent.set_episode_duration(3)
        agent.Max_Step += agent.bonus_step_increment
        agent.goal = Loc(66, 50, 0)
        reward, done, result = agent.update([0, 0, 0])
        self.assertAlmostEqual(reward, 1.95)
        self.assertEqual((done, result), (False, "normal"))
        with patch("builtins.open", mock_open()):
            reward, done, result = agent.update([0, 0, 0])
        self.assertAlmostEqual(reward, 151.95)
        self.assertEqual((done, result), (True, "success"))

    def test_timeout_charges_navigation_reward_and_timeout_penalty(self):
        agent, _ = self.make_agent()
        agent.set_episode_duration(1)
        reward, done, result = agent.update([0, 0, 0])
        self.assertAlmostEqual(reward, -48.05)
        self.assertEqual((done, result), (True, "lose"))

    def test_boundary_penalty_applies_once_per_navigation_interval(self):
        agent, _ = self.make_agent(hardware_hz=1)
        agent.position = Loc(500, 50, 0)
        agent.goal = Loc(600, 50, 0)
        with patch.object(agent, "cal_force", return_value=Loc(1, 0, 0)):
            rewards = [agent.update([0, 0, 0])[0] for _ in range(agent.controls_per_decision)]
        self.assertEqual(rewards[:-1], [0.0] * 479)
        self.assertAlmostEqual(rewards[-1], -10.05)


if __name__ == "__main__":
    unittest.main()
