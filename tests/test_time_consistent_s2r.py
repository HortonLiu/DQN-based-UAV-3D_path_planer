"""Regression checks for physical time with the retained Step += 2 counter."""

import copy
import importlib
import math
import os
import tempfile
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "Agents", ROOT / "BaseClass", ROOT / "Envs"):
    sys.path.insert(0, str(directory))

from BaseClass.CalMod import Loc, XML2Dict
from Obstacles.building import building

UAV = importlib.import_module("UAV_Single_RL+VOAPF S2R").UAV
PathPlanCity = importlib.import_module("Envs.PathPlan_City").PathPlan_City


class EmptyEnvironment:
    len = 500
    width = 500
    h = 100

    def __init__(self):
        self.buildings = []
        self.dynamic_buildings = []
        self.sim_time = 0.0
        self.control_updates = 0
        self.blocked = False

    def Threaten_rate(self, position):
        return int(self.blocked)

    def advance_dynamic_obstacles(self, dt):
        self.sim_time += dt
        self.control_updates += 1
        for obstacle in self.buildings:
            obstacle.run(dt)


class TimeConsistentS2RTests(unittest.TestCase):
    def setUp(self):
        original_cwd = os.getcwd()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir, original_cwd)
        os.chdir(temporary.name)

    def make_agent(self, **overrides):
        parameters = XML2Dict(str(ROOT / "config/UAV.xml"))["Agent"]
        parameters.update(name="physical_time_test", j=0)
        parameters.update(overrides)
        env = EmptyEnvironment()
        with patch.object(UAV, "Init_Record_Mod", return_value=None):
            agent = UAV(parameters, env)
        agent.reset()
        # Keep the reset position so the initial RTK/history anchor remains valid.
        # Its protected start region leaves at least ten meters of +X travel.
        agent.goal = Loc(agent.position.x + 200.0, agent.position.y, 0.0)
        agent.V_vector = Loc(agent.Max_V, 0.0, 0.0)
        agent.V = agent.Max_V
        agent.cal_force = lambda *args: Loc(1.0, 0.0, 0.0)
        return agent, env

    def test_one_call_preserves_step_two_and_physical_displacements(self):
        agent, env = self.make_agent()
        start = copy.copy(agent.position)
        obstacle = building({"position": {"x": 100, "y": 100, "z": 0},
                             "_R": 20, "_H": 50}, env)
        obstacle.configure_motion((1, 1), control_dt=agent.control_dt)
        env.buildings.append(obstacle)
        reward, done, info = agent.update([0, 0, 0])
        self.assertFalse(done)
        self.assertEqual(info, "normal")
        self.assertEqual((agent.Step, agent.Control_step), (2, 1))
        self.assertEqual(env.control_updates, 1)
        self.assertAlmostEqual(agent.position.x - start.x, 1.0 / 480.0, places=10)
        self.assertAlmostEqual(agent.path_len, 1.0 / 480.0, places=10)
        self.assertAlmostEqual(agent.sim_time, 1.0 / 480.0)
        self.assertAlmostEqual(env.sim_time, 1.0 / 480.0)
        self.assertAlmostEqual(math.hypot(obstacle.position.x - 100,
                                         obstacle.position.y - 100), 4.0 / 480.0, places=10)
        self.assertAlmostEqual(reward, 0.9 / 480.0, places=10)

    def test_1200_reference_steps_allow_1200_seconds_not_600_updates(self):
        agent, env = self.make_agent(Max_Step="1200")
        start_x = agent.position.x
        self.assertEqual(agent.original_max_step, 1152000)
        self.assertEqual(agent.Max_Step, 1152000)
        self.assertAlmostEqual(agent.base_duration_s, 1200.0)
        for _ in range(625):
            self.assertFalse(agent.update([0, 0, 0])[1])
        self.assertEqual((agent.Step, agent.Control_step), (1250, 625))
        self.assertAlmostEqual(agent.position.x - start_x, 625 / 480.0, places=8)
        self.assertEqual(agent.active_virtual_obstacles, [])
        self.assertEqual(agent.Max_Step, 1152000)

    def test_two_second_budget_expires_after_960_single_updates(self):
        agent, env = self.make_agent(Max_Step="2")
        start_x = agent.position.x
        self.assertEqual(agent.Max_Step, 1920)
        total_reward = 0.0
        for _ in range(959):
            reward, done, info = agent.update([0, 0, 0])
            self.assertFalse(done)
            total_reward += reward
        reward, done, info = agent.update([0, 0, 0])
        total_reward += reward
        self.assertTrue(done)
        self.assertEqual(info, "lose")
        self.assertEqual((agent.Step, agent.Control_step), (1920, 960))
        self.assertAlmostEqual(agent.position.x - start_x, 2.0, places=8)
        self.assertAlmostEqual(agent.sim_time, 2.0)
        self.assertAlmostEqual(total_reward, -18.2, places=8)

    def test_one_second_rewards_are_independent_of_480_hz_subdivision(self):
        agent, env = self.make_agent()
        start_x = agent.position.x
        reward = sum(agent.update([0, 0, 0])[0] for _ in range(480))
        self.assertAlmostEqual(agent.position.x - start_x, 1.0, places=8)
        self.assertEqual((agent.Step, agent.Control_step), (960, 480))
        self.assertAlmostEqual(reward, 0.9, places=8)

    def test_goal_remaining_time_bonus_uses_seconds(self):
        agent, env = self.make_agent(Max_Step="1200")
        for _ in range(480):
            agent.update([0, 0, 0])
        agent.goal = copy.copy(agent.position)
        reward, done, info = agent.update([0, 0, 0])
        self.assertTrue(done)
        self.assertEqual(info, "success")
        self.assertAlmostEqual(reward, 100 + 1199, places=6)
        self.assertEqual((agent.Step, agent.Control_step), (960, 480))

    def test_rtk_lidar_hold_values_for_24_and_48_physical_updates(self):
        agent, env = self.make_agent()
        start_x = agent.position.x
        initial = agent.state().copy()
        lidar = agent.cached_lidar_grid
        for _ in range(23):
            agent.update([0, 0, 0])
            self.assertEqual(agent.state().tolist(), initial.tolist())
        self.assertEqual(agent.Step, 46)
        agent.update([0, 0, 0])
        agent.state()
        self.assertEqual(agent.Step, 48)
        self.assertAlmostEqual(agent.cached_rtk_pos.x, start_x + 0.05, places=9)
        self.assertIs(agent.cached_lidar_grid, lidar)
        for _ in range(24):
            agent.update([0, 0, 0])
            agent.state()
        self.assertEqual(agent.Step, 96)
        self.assertAlmostEqual(agent.cached_rtk_pos.x, start_x + 0.1, places=9)
        self.assertIsNot(agent.cached_lidar_grid, lidar)
        grid = agent.cached_lidar_grid
        pos = agent.cached_rtk_pos
        for _ in range(3):
            agent.state()
        self.assertIs(agent.cached_lidar_grid, grid)
        self.assertIs(agent.cached_rtk_pos, pos)

    def test_ten_seconds_of_forward_motion_does_not_create_virtual_sources(self):
        agent, env = self.make_agent()
        start_x = agent.position.x
        for _ in range(4800):
            agent.update([0, 0, 0])
        self.assertEqual((agent.Step, agent.Control_step), (9600, 4800))
        self.assertAlmostEqual(agent.position.x - start_x, 10.0, places=8)
        self.assertFalse(agent.check_is_stuck())
        self.assertEqual(agent.active_virtual_obstacles, [])
        self.assertEqual(agent.Max_Step, agent.original_max_step)

    def test_stationary_ten_second_window_gets_one_fifty_second_extension(self):
        agent, env = self.make_agent()
        env.blocked = True
        for _ in range(4799):
            agent.update([0, 0, 0])
        self.assertFalse(agent.check_is_stuck())
        self.assertEqual(agent.active_virtual_obstacles, [])
        agent.update([0, 0, 0])
        self.assertTrue(agent.check_is_stuck())
        # Detection may be processed at the completed sample or next control call.
        if not agent.active_virtual_obstacles:
            agent.update([0, 0, 0])
        self.assertEqual(len(agent.active_virtual_obstacles), 1)
        source = agent.active_virtual_obstacles[0]
        self.assertAlmostEqual(source["force"], agent.virtual_force_initial)
        self.assertEqual(agent.Max_Step - agent.original_max_step, 48000)
        self.assertAlmostEqual(agent.path_len, 0.0)
        for _ in range(100):
            agent.update([0, 0, 0])
        self.assertAlmostEqual(source["force"], agent.virtual_force_initial)
        while agent.Control_step < 5281:
            agent.update([0, 0, 0])
        self.assertEqual(len(agent.active_virtual_obstacles), 1)
        self.assertAlmostEqual(source["force"],
                               min(agent.max_virtual_force,
                                   agent.virtual_force_initial + agent.virtual_force_growth))

    def test_virtual_source_coincidence_does_not_resample_random_directions(self):
        agent, env = self.make_agent()
        env.blocked = True
        for _ in range(4801):
            agent.update([0, 0, 0])
        self.assertEqual(len(agent.active_virtual_obstacles), 1)
        with patch("random.uniform", side_effect=AssertionError("per-tick random jitter")):
            forces = [UAV.cal_force(agent, agent.position, 15.0, 1.0)
                      for _ in range(10)]
        self.assertTrue(all(math.isclose(force.x, forces[0].x)
                            and math.isclose(force.y, forces[0].y) for force in forces))

    def test_collision_reward_is_scaled_and_path_length_is_actual_motion(self):
        agent, env = self.make_agent()
        env.blocked = True
        start = copy.copy(agent.position)
        reward, done, info = agent.update([0, 0, 0])
        self.assertFalse(done)
        self.assertEqual(agent.position, start)
        self.assertAlmostEqual(agent.path_len, 0.0)
        self.assertAlmostEqual(reward, -5.1 / 480.0, places=10)
        self.assertEqual(agent.collision_count, 1)

    def test_episode_duration_override_keeps_even_step_counter_units(self):
        agent, env = self.make_agent()
        agent.set_episode_duration(2.001)
        self.assertEqual(agent.Max_Step, 2 * math.ceil(2.001 * 480))
        self.assertEqual(agent.Max_Step % 2, 0)
        self.assertAlmostEqual(agent.base_duration_s, 2.001)


class FakeReplay:
    def __init__(self):
        self.buffer = []

    def add(self, *transition):
        self.buffer.append(transition)


class FakeTrainer:
    Batch_Size = 10000
    Is_Train = 1

    def __init__(self):
        self.replay_memory = FakeReplay()
        self.pushed = []
        self.action_calls = 0

    def get_action(self, state, eps):
        self.action_calls += 1
        return [0.0, 0.0, 0.0]

    def Push_Replay(self, transition):
        self.pushed.append(transition)


class DecisionIntegrationTests(unittest.TestCase):
    setUp = TimeConsistentS2RTests.setUp
    make_agent = TimeConsistentS2RTests.make_agent

    def test_one_second_decision_holds_one_action_and_records_one_transition(self):
        agent, physical_env = self.make_agent()
        trainer = FakeTrainer()
        agent.Trainer = trainer
        controller = PathPlanCity.__new__(PathPlanCity)
        controller.Agents = [agent]
        controller.Reset_Result(0.1)
        controller.run_thread_OffPolicy(agent, 0, 0.1)
        self.assertEqual(trainer.action_calls, 1)
        self.assertEqual((agent.Step, agent.Control_step), (960, 480))
        self.assertEqual(physical_env.control_updates, 480)
        self.assertEqual(len(trainer.replay_memory.buffer), 1)
        self.assertEqual(len(trainer.pushed), 1)
        transition = trainer.replay_memory.buffer[0]
        self.assertAlmostEqual(float(transition[2]), 0.9, places=8)
        self.assertFalse(transition[4])
        self.assertEqual(transition[1], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
