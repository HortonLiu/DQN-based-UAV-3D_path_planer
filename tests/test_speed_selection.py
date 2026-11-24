"""Startup speed selection controls physical motion in both scenes and time modes."""

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


class SpeedSelectionTests(unittest.TestCase):
    def setUp(self):
        previous_directory = os.getcwd()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir, previous_directory)
        os.chdir(temporary.name)

    def make_scene(self, hardware, dynamic, speed):
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
        parameters.update(name='speed_selection_test', j=0, Max_V=speed,
                          Hardware_Hz_Enabled=int(hardware))
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

    @staticmethod
    def environment_parameters(scene, hardware, speed):
        profile = 'PathPlan_City_Static.xml' if scene == 'static' else 'PathPlan_City.xml'
        parameters = XML2Dict(str(ROOT / 'config' / profile))['simulator']['env']
        parameters['Hardware_Hz_Enabled'] = int(hardware)
        if speed is not None:
            parameters['UAV_Speed_M_S'] = speed
        for node, key in ((parameters['Agent'], 'xml_path_agent'),
                          (parameters['Agent']['Trainer'], 'Trainer_path'),
                          (parameters['Obstacles'], 'buildings')):
            node[key] = str(ROOT / node[key])
        return parameters

    def test_eight_combinations_move_at_selected_speed_and_keep_step_two(self):
        for hardware in (False, True):
            for dynamic in (False, True):
                for speed in (1, 4):
                    with self.subTest(hardware=hardware, dynamic=dynamic, speed=speed):
                        env, agent = self.make_scene(hardware, dynamic, speed)
                        hz = 480 if hardware else 1
                        start = copy.copy(agent.position)
                        centers = [(ob.position.x, ob.position.y, ob.position.z)
                                   for ob in env.buildings]
                        agent.update([0, 0, 0])
                        self.assertEqual((agent.Step, agent.Control_step), (2, 1))
                        self.assertAlmostEqual(agent.position.x - start.x, speed / hz, places=10)
                        for _ in range(hz - 1):
                            self.assertFalse(agent.update([0, 0, 0])[1])
                        self.assertAlmostEqual(agent.position.x - start.x, speed, places=8)
                        self.assertAlmostEqual(agent.path_len, speed, places=8)
                        self.assertEqual((agent.Step, agent.Control_step), (2 * hz, hz))
                        self.assertAlmostEqual(agent.sim_time, 1)
                        for index, (ob, center) in enumerate(zip(env.buildings, centers)):
                            displacement = math.dist((ob.position.x, ob.position.y,
                                                      ob.position.z), center)
                            self.assertAlmostEqual(displacement,
                                                   4 if dynamic and index < 6 else 0,
                                                   places=8)
                        exported = json.loads(env.getEnvJson())
                        self.assertEqual(exported['uav_speed_m_s'], speed)
                        self.assertEqual(len(exported['dynamic_obstacles']), 6 if dynamic else 0)

    def test_environment_overrides_xml_before_initialization_and_isolates_speed_weights(self):
        xml_files = (ROOT / 'config/UAV.xml', ROOT / 'config/UAV_Static.xml')
        original_xml = {path: path.read_bytes() for path in xml_files}
        for scene in ('static', 'mixed'):
            for hardware in (False, True):
                for speed in (None, 1, 4):
                    with self.subTest(scene=scene, hardware=hardware, speed=speed):
                        parameters = self.environment_parameters(scene, hardware, speed)
                        configured_agent = XML2Dict(parameters['Agent']['xml_path_agent'])['Agent']
                        trainer_config = XML2Dict(
                            parameters['Agent']['Trainer']['Trainer_path'])['Trainer']
                        checkpoint = Path(trainer_config['checkpoint_dir'])
                        if not hardware:
                            checkpoint /= 'one_second'
                        if speed is not None:
                            checkpoint /= 'v%d' % speed
                        requested = []

                        def make_trainer(configuration):
                            requested.append(copy.deepcopy(configuration))
                            return types.SimpleNamespace()

                        with patch.object(DynamicUAV, 'Init_Record_Mod', return_value=None), \
                                patch.object(StaticUAV, 'Init_Record_Mod', return_value=None), \
                                patch.object(TrainerFactory, 'Create_Trainer', side_effect=make_trainer):
                            env = Environment(parameters)
                        agent = env.Agents[0]
                        expected_speed = speed if speed is not None else float(configured_agent['Max_V'])
                        self.assertEqual(agent.Max_V, expected_speed)
                        self.assertEqual(float(agent.param['Max_V']), expected_speed)
                        self.assertEqual(Path(requested[0]['checkpoint_dir']), checkpoint)
                        self.assertEqual(env.dynamic_speed, 4.0 if scene == 'mixed' else 0.0)
        self.assertEqual({path: path.read_bytes() for path in xml_files}, original_xml)

    def test_invalid_environment_speed_fails_before_agent_and_trainer_creation(self):
        for speed in (0, 2, -1, 4.1, float('inf'), float('nan')):
            with self.subTest(speed=speed):
                parameters = self.environment_parameters('static', True, speed)
                with patch.object(TrainerFactory, 'Create_Trainer') as trainer:
                    with self.assertRaises(ValueError):
                        Environment(parameters)
                trainer.assert_not_called()

    def test_simulator_passes_explicit_speed_without_changing_xml_default(self):
        requested = []
        environment = types.SimpleNamespace(Agents=[types.SimpleNamespace(Trainer=object())])

        class FakeFactory:
            def Create_Env(self, parameters):
                requested.append(copy.deepcopy(parameters))
                return environment

        sim = simulator_module.simulator.__new__(simulator_module.simulator)
        sim.EnvFactory = FakeFactory()
        sim.scene = 'static'
        sim.hardware_hz_enabled = True
        for speed in (None, 1, 4):
            sim.uav_speed = speed
            sim.Init_From_XML()
            if speed is None:
                self.assertNotIn('UAV_Speed_M_S', requested[-1])
            else:
                self.assertEqual(requested[-1]['UAV_Speed_M_S'], speed)

    def test_csv_records_actual_selected_speed_in_all_eight_combinations(self):
        for hardware in (False, True):
            for dynamic in (False, True):
                for speed in (1, 4):
                    with self.subTest(hardware=hardware, dynamic=dynamic, speed=speed):
                        env, agent = self.make_scene(hardware, dynamic, speed)
                        agent.Trainer = types.SimpleNamespace(epoch=0)
                        agent.update([0, 0, 0])
                        sim = simulator_module.simulator.__new__(simulator_module.simulator)
                        sim.env, sim.epoch, sim.executed_time = env, 1, 0
                        sim.result_path = str(Path.cwd() / ('speed_%d_%d_%d.csv' %
                                                          (hardware, dynamic, speed)))
                        sim.writer = sim._result_file = sim.CsvWriter = None
                        sim.infos = [{'success': 0, 'lose': 0}]
                        try:
                            sim.Init_Record_Mod()
                            sim.record_list()
                            with open(sim.result_path, encoding='utf-8', newline='') as source:
                                row = next(csv.DictReader(source))
                            self.assertNotIn(None, row)
                            self.assertEqual(float(row['uav_speed_m_s']), speed)
                            self.assertEqual(int(row['hardware_hz_enabled']), hardware)
                            self.assertEqual(int(row['dynamic_obstacles_enabled']), dynamic)
                        finally:
                            sim.close()


class SpeedPromptTests(unittest.TestCase):
    def test_speed_prompt_retries_until_one_or_four_and_reports_eof_flags(self):
        with patch('builtins.input', side_effect=['0', '2', '', '4.0', ' 4 ']) as ask:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(simulator_module.prompt_uav_speed('速度?'), 4)
        self.assertEqual(ask.call_count, 5)
        with patch('builtins.input', return_value='1'):
            self.assertEqual(simulator_module.prompt_uav_speed('速度?'), 1)
        with patch('builtins.input', side_effect=EOFError):
            with self.assertRaises(ValueError) as error:
                simulator_module.prompt_uav_speed('速度?')
        self.assertIn('--uav-speed', str(error.exception))

    def test_default_prompt_order_and_answers_map_all_eight_combinations(self):
        for hardware in (0, 1):
            for dynamic in (0, 1):
                for speed in (1, 4):
                    with self.subTest(hardware=hardware, dynamic=dynamic, speed=speed):
                        with patch.object(sys, 'argv', ['simulator.py']), \
                                patch('builtins.input', side_effect=[str(hardware), str(dynamic),
                                                                   str(speed)]) as ask, \
                                patch.object(simulator_module, 'simulator') as create:
                            self.assertEqual(simulator_module.main(), 0)
                        questions = [call.args[0] for call in ask.call_args_list]
                        self.assertEqual(len(questions), 3)
                        self.assertIn('Hz', questions[0])
                        self.assertIn('動態', questions[1])
                        self.assertIn('速度', questions[2])
                        self.assertEqual(create.call_args.kwargs['uav_speed'], speed)
                        self.assertEqual(create.call_args.kwargs['hardware_hz'], bool(hardware))
                        self.assertEqual(create.call_args.kwargs['scene'], 'mixed' if dynamic else 'static')

    def test_cli_speed_option_and_partial_flags_prompt_only_missing_choices(self):
        cases = [(['--hardware-hz', '0', '--scene', 'mixed'], ['4'], ['速度'], 4),
                 (['--uav-speed', '1'], ['1', '0'], ['Hz', '動態'], 1),
                 (['--hardware-hz', '0', '--scene', 'static', '--uav-speed', '4'], [], [], 4)]
        for flags, answers, expected_questions, speed in cases:
            with self.subTest(flags=flags):
                with patch.object(sys, 'argv', ['simulator.py'] + flags), \
                        patch('builtins.input', side_effect=answers) as ask, \
                        patch.object(simulator_module, 'simulator') as create:
                    self.assertEqual(simulator_module.main(), 0)
                self.assertEqual(ask.call_count, len(expected_questions))
                for call, phrase in zip(ask.call_args_list, expected_questions):
                    self.assertIn(phrase, call.args[0])
                self.assertEqual(create.call_args.kwargs['uav_speed'], speed)


if __name__ == '__main__':
    unittest.main()
