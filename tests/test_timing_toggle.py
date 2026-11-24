"""Both timing modes preserve physical seconds and the public Step += 2 counter."""

import copy
import csv
import importlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'Agents', ROOT / 'BaseClass', ROOT / 'Envs'):
    sys.path.insert(0, str(directory))

from BaseClass.CalMod import Loc, XML2Dict
from FactoryClass.TrainerFactory import TrainerFactory
from Obstacles.building import building
import simulator as simulator_module

DynamicUAV = importlib.import_module('UAV_Single_RL+VOAPF S2R').UAV
StaticUAV = importlib.import_module('UAV_Single_RL+VOAPF S2R_Static').UAV
Environment = importlib.import_module('Envs.PathPlan_City').PathPlan_City


class TimingToggleTests(unittest.TestCase):
    def setUp(self):
        previous_directory = os.getcwd()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir, previous_directory)
        os.chdir(temporary.name)

    def make_scene(self, hardware, dynamic, duration=1200):
        env = Environment.__new__(Environment)
        env.len, env.width, env.h = 500, 500, 100
        env.num_UAV = 1
        env.hardware_hz_enabled = bool(hardware)
        env.dynamic_enabled = True
        env.dynamic_speed = 4.0
        env.motion_half_range = None
        env.dynamic_buildings = []
        env.sim_time = 0.0
        env.epoch = 1
        env.map_reset_loop = 2
        env.buildings_param = XML2Dict(str(ROOT / 'config/buildings.xml'))['buildings']
        env.buildings = [building(parameters, env)
                         for parameters in env.buildings_param['Threaten']]
        env.param = {'UI': {'UI_path': str(ROOT / 'config/UI.xml')}}
        env.Threaten_rate = lambda position: 0
        parameters = XML2Dict(str(ROOT / 'config/UAV.xml'))['Agent']
        parameters.update(name='timing_toggle_test', j=0, Max_V=1,
                          Max_Step=duration, Hardware_Hz_Enabled=int(hardware))
        agent_class = DynamicUAV if dynamic else StaticUAV
        with patch.object(agent_class, 'Init_Record_Mod', return_value=None):
            agent = agent_class(parameters, env)
        env.Agents = [agent]
        agent.reset()
        agent.goal = Loc(agent.position.x + 200, agent.position.y, 0)
        agent.V_vector = Loc(agent.Max_V, 0, 0)
        agent.V = agent.Max_V
        agent.cal_force = lambda *arguments: Loc(1, 0, 0)
        env._configure_dynamic_obstacles()
        return env, agent

    def test_four_combinations_advance_exactly_one_second_with_correct_motion(self):
        for hardware in (False, True):
            for dynamic in (False, True):
                with self.subTest(hardware=hardware, dynamic=dynamic):
                    env, agent = self.make_scene(hardware, dynamic)
                    hz = 480 if hardware else 1
                    start = copy.copy(agent.position)
                    centers = [(ob.position.x, ob.position.y, ob.position.z)
                               for ob in env.buildings]
                    for _ in range(hz):
                        self.assertFalse(agent.update([0, 0, 0])[1])
                    self.assertEqual((agent.Step, agent.Control_step), (2 * hz, hz))
                    self.assertAlmostEqual(agent.position.x - start.x, 1, places=8)
                    self.assertAlmostEqual(agent.path_len, 1, places=8)
                    self.assertAlmostEqual(agent.sim_time, 1)
                    self.assertAlmostEqual(env.sim_time, 1)
                    self.assertEqual(len(env.dynamic_buildings), 6 if dynamic else 0)
                    for index, (ob, center) in enumerate(zip(env.buildings, centers)):
                        displacement = math.dist((ob.position.x, ob.position.y,
                                                  ob.position.z), center)
                        self.assertAlmostEqual(displacement,
                                               4 if dynamic and index < 6 else 0,
                                               places=8)
                    metadata = json.loads(env.getEnvJson())['timing']
                    self.assertEqual(metadata['control_hz'], hz)
                    self.assertEqual(metadata['hardware_hz_enabled'], hardware)
                    self.assertEqual(metadata['control_dt_s'], 1 / hz)
                    self.assertEqual(metadata['legacy_step_count'], 2 * hz)
                    self.assertEqual(metadata['controls_per_decision'], hz)
                    self.assertEqual(agent.original_max_step, 1200 * hz * 2)

    def test_two_second_timeout_and_rewards_are_consistent_in_both_modes(self):
        for hardware in (False, True):
            with self.subTest(hardware=hardware):
                env, agent = self.make_scene(hardware, False, duration=2)
                hz = 480 if hardware else 1
                score = 0.0
                for index in range(2 * hz):
                    reward, done, info = agent.update([0, 0, 0])
                    score += reward
                    self.assertEqual(done, index == 2 * hz - 1)
                self.assertEqual(info, 'lose')
                self.assertEqual(agent.Step, 4 * hz)
                self.assertEqual(agent.Max_Step, 4 * hz)
                self.assertAlmostEqual(agent.sim_time, 2)
                self.assertAlmostEqual(score, -18.2, places=8)

    def test_coarse_sensors_refresh_each_update_and_virtual_window_is_ten_seconds(self):
        env, agent = self.make_scene(False, False)
        self.assertEqual((agent.freq_rtk, agent.freq_lidar), (1, 1))
        self.assertEqual((agent.steps_per_rtk, agent.steps_per_lidar), (1, 1))
        self.assertEqual(agent.stuck_history_samples, 11)
        env.Threaten_rate = lambda position: 1
        agent.state()
        old_lidar = agent.cached_lidar_grid
        for _ in range(9):
            agent.update([0, 0, 0])
        self.assertIsNot(agent.cached_lidar_grid, old_lidar)
        self.assertEqual(agent.active_virtual_obstacles, [])
        agent.update([0, 0, 0])
        self.assertEqual((agent.Step, agent.Control_step), (20, 10))
        self.assertEqual(len(agent.active_virtual_obstacles), 1)
        source = agent.active_virtual_obstacles[0]
        self.assertEqual(source['created_time_s'], 10)
        self.assertEqual(source['force'], agent.virtual_force_initial)
        self.assertEqual(agent.Max_Step - agent.original_max_step, 100)
        agent.update([0, 0, 0])
        self.assertEqual(len(agent.active_virtual_obstacles), 1)
        self.assertEqual(source['force'], agent.virtual_force_initial + agent.virtual_force_growth)

    def test_environment_propagates_mode_before_initialization_and_separates_weights(self):
        for scene in ('static', 'mixed'):
            profile = 'PathPlan_City_Static.xml' if scene == 'static' else 'PathPlan_City.xml'
            for hardware in (False, True):
                with self.subTest(scene=scene, hardware=hardware):
                    parameters = XML2Dict(str(ROOT / 'config' / profile))['simulator']['env']
                    parameters['Hardware_Hz_Enabled'] = int(hardware)
                    for node, key in ((parameters['Agent'], 'xml_path_agent'),
                                      (parameters['Agent']['Trainer'], 'Trainer_path'),
                                      (parameters['Obstacles'], 'buildings')):
                        node[key] = str(ROOT / node[key])
                    requested = []

                    def make_trainer(configuration):
                        requested.append(copy.deepcopy(configuration))
                        return types.SimpleNamespace()

                    with patch.object(DynamicUAV, 'Init_Record_Mod', return_value=None), \
                            patch.object(StaticUAV, 'Init_Record_Mod', return_value=None), \
                            patch.object(TrainerFactory, 'Create_Trainer', side_effect=make_trainer):
                        env = Environment(parameters)
                    agent = env.Agents[0]
                    self.assertEqual(agent.hardware_hz_enabled, hardware)
                    self.assertEqual(agent.control_dt, 1 / (480 if hardware else 1))
                    configured = Path(requested[0]['checkpoint_dir'])
                    self.assertEqual(configured.name == 'one_second', not hardware)
                    self.assertEqual(env.dynamic_enabled, scene == 'mixed')

    def test_csv_records_selected_mode_and_real_seconds_for_all_combinations(self):
        for hardware in (False, True):
            for dynamic in (False, True):
                with self.subTest(hardware=hardware, dynamic=dynamic):
                    env, agent = self.make_scene(hardware, dynamic, duration=2)
                    agent.Trainer = types.SimpleNamespace(epoch=0)
                    hz = 480 if hardware else 1
                    for _ in range(2 * hz):
                        agent.update([0, 0, 0])
                    sim = simulator_module.simulator.__new__(simulator_module.simulator)
                    sim.env, sim.epoch, sim.executed_time = env, 1, 0
                    sim.result_path = str(Path.cwd() / ('scores_%d_%d.csv' % (hardware, dynamic)))
                    sim.writer = sim._result_file = sim.CsvWriter = None
                    sim.infos = [{'success': 0, 'lose': 1}]
                    try:
                        sim.Init_Record_Mod()
                        sim.record_list()
                        with open(sim.result_path, encoding='utf-8', newline='') as source:
                            row = next(csv.DictReader(source))
                        self.assertEqual(int(row['hardware_hz_enabled']), int(hardware))
                        self.assertEqual(int(row['dynamic_obstacles_enabled']), int(dynamic))
                        self.assertEqual(row['scene'], 'mixed' if dynamic else 'static')
                        self.assertEqual(float(row['control_hz']), hz)
                        self.assertEqual(float(row['control_dt_s']), 1 / hz)
                        self.assertEqual(float(row['episode_budget_s']), 2)
                        self.assertEqual(float(row['simulation_time_s']), 2)
                        self.assertEqual(float(row['step']), 4 * hz)
                        self.assertEqual(int(row['control_updates']), 2 * hz)
                        self.assertAlmostEqual(float(row['path_length_m']), 2, places=8)
                    finally:
                        sim.close()


class StartupSelectionTests(unittest.TestCase):
    def test_prompt_retries_invalid_values_and_accepts_only_binary_answers(self):
        with patch('builtins.input', side_effect=['yes', '2', '', ' 1 ']) as ask:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(simulator_module.prompt_binary('timing?'), 1)
        self.assertEqual(ask.call_count, 4)
        with patch('builtins.input', return_value='0'):
            self.assertEqual(simulator_module.prompt_binary('timing?'), 0)

    def test_eof_explains_noninteractive_flags(self):
        with patch('builtins.input', side_effect=EOFError):
            with self.assertRaises(ValueError) as error:
                simulator_module.prompt_binary('timing?')
        self.assertIn('--hardware-hz', str(error.exception))
        self.assertIn('--scene', str(error.exception))

    def test_default_startup_asks_hardware_then_dynamic_then_speed(self):
        for hardware in (0, 1):
            for dynamic in (0, 1):
                with self.subTest(hardware=hardware, dynamic=dynamic):
                    with patch.object(sys, 'argv', ['simulator.py']), \
                            patch('builtins.input', side_effect=[str(hardware), str(dynamic), '4']) as ask, \
                            patch.object(simulator_module, 'simulator') as create:
                        self.assertEqual(simulator_module.main(), 0)
                    self.assertEqual(ask.call_count, 3)
                    questions = [str(call.args[0]) for call in ask.call_args_list]
                    self.assertTrue('Hz' in questions[0] or '480' in questions[0])
                    self.assertIn('動態', questions[1])
                    self.assertIn('速度', questions[2])
                    self.assertEqual(create.call_args.kwargs['hardware_hz'], bool(hardware))
                    self.assertEqual(create.call_args.kwargs['scene'], 'mixed' if dynamic else 'static')
                    self.assertEqual(create.call_args.kwargs['uav_speed'], 4)
                    create.return_value.StartAndTrain.assert_called_once()
                    create.return_value.close.assert_called_once()

    def test_explicit_flags_skip_all_prompts(self):
        arguments = ['simulator.py', '--hardware-hz', '0', '--scene', 'mixed',
                     '--uav-speed', '1', '--episodes', '1', '--max-seconds', '2', '--no-train']
        with patch.object(sys, 'argv', arguments), \
                patch('builtins.input', side_effect=AssertionError('unexpected prompt')), \
                patch.object(simulator_module, 'simulator') as create:
            self.assertEqual(simulator_module.main(), 0)
        self.assertEqual(create.call_args.kwargs['hardware_hz'], False)
        self.assertEqual(create.call_args.kwargs['scene'], 'mixed')
        self.assertEqual(create.call_args.kwargs['uav_speed'], 1)

    def test_partial_flags_ask_only_the_missing_choice(self):
        cases = [(['--hardware-hz', '0', '--uav-speed', '4'], '1', False, 'mixed', '動態'),
                 (['--scene', 'static', '--uav-speed', '4'], '1', True, 'static', 'Hz')]
        for flags, answer, hardware, scene, question in cases:
            with self.subTest(flags=flags):
                with patch.object(sys, 'argv', ['simulator.py'] + flags), \
                        patch('builtins.input', return_value=answer) as ask, \
                        patch.object(simulator_module, 'simulator') as create:
                    self.assertEqual(simulator_module.main(), 0)
                ask.assert_called_once()
                self.assertIn(question, ask.call_args.args[0])
                self.assertEqual(create.call_args.kwargs['hardware_hz'], hardware)
                self.assertEqual(create.call_args.kwargs['scene'], scene)


if __name__ == '__main__':
    unittest.main()
