"""All-static profile keeps physical timing, virtual sources, and separate weights."""

import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'Agents', ROOT / 'BaseClass', ROOT / 'Envs'):
    sys.path.insert(0, str(directory))

from BaseClass.CalMod import Loc, XML2Dict
from Obstacles.building import building
import simulator as simulator_module

environment_module = importlib.import_module('Envs.PathPlan_City')
Environment = environment_module.PathPlan_City
StaticUAV = importlib.import_module('UAV_Single_RL+VOAPF S2R_Static').UAV


class StaticSimulationTests(unittest.TestCase):
    def setUp(self):
        previous_directory = os.getcwd()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir, previous_directory)
        os.chdir(temporary.name)

    def make_environment(self):
        env = Environment.__new__(Environment)
        env.len, env.width, env.h = 500, 500, 100
        env.num_UAV = 1
        # Start with an already configured mixed scene to exercise the static
        # module's guard as well as the static XML setting.
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
        env.Agents = [types.SimpleNamespace(control_dt=1 / 480)]
        env._configure_dynamic_obstacles()
        self.assertEqual(len(env.dynamic_buildings), 6)
        env.param = {'UI': {'UI_path': str(ROOT / 'config/UI.xml')}}
        env.Threaten_rate = lambda position: 0
        parameters = XML2Dict(str(ROOT / 'config/UAV_Static.xml'))['Agent']
        parameters.update(name='static_timing_test', j=0, Max_V='1')
        with patch.object(StaticUAV, 'Init_Record_Mod', return_value=None):
            agent = StaticUAV(parameters, env)
        env.Agents = [agent]
        agent.reset()
        agent.goal = Loc(agent.position.x + 200.0, agent.position.y, 0.0)
        agent.V_vector = Loc(agent.Max_V, 0.0, 0.0)
        agent.V = agent.Max_V
        agent.cal_force = lambda *arguments: Loc(1.0, 0.0, 0.0)
        return env, agent

    def test_static_configuration_uses_existing_geometry_and_separate_checkpoints(self):
        static = XML2Dict(str(ROOT / 'config/PathPlan_City_Static.xml'))['simulator']['env']
        mixed = XML2Dict(str(ROOT / 'config/PathPlan_City.xml'))['simulator']['env']
        self.assertEqual(int(static['Dynamic_Obstacles']['enabled']), 0)
        self.assertEqual(int(mixed['Dynamic_Obstacles']['enabled']), 1)
        self.assertEqual(static['Obstacles']['buildings'], mixed['Obstacles']['buildings'])
        self.assertEqual(len(XML2Dict(str(ROOT / 'config/buildings.xml'))['buildings']['Threaten']), 26)
        agent_parameters = XML2Dict(str(ROOT / static['Agent']['xml_path_agent']))['Agent']
        self.assertEqual(agent_parameters['Agent_Module'], 'UAV_Single_RL+VOAPF S2R_Static')
        static_trainer = XML2Dict(str(ROOT / static['Agent']['Trainer']['Trainer_path']))['Trainer']
        mixed_trainer = XML2Dict(str(ROOT / mixed['Agent']['Trainer']['Trainer_path']))['Trainer']
        self.assertEqual(static_trainer['checkpoint_dir'], 'Mod/S2R_Static_time_consistent')
        self.assertNotEqual(static_trainer['checkpoint_dir'], mixed_trainer['checkpoint_dir'])

    def test_all_twenty_six_centers_stay_fixed_while_uav_moves_at_480_hz(self):
        env, agent = self.make_environment()
        centers = [(ob.position.x, ob.position.y, ob.position.z) for ob in env.buildings]
        geometry = [(ob._R, ob._H) for ob in env.buildings]
        env._configure_dynamic_obstacles()  # Later scene setup cannot re-enable motion.
        self.assertFalse(env.dynamic_enabled)
        self.assertEqual(env.dynamic_buildings, [])
        self.assertFalse(any(ob.is_dynamic for ob in env.buildings))
        self.assertTrue(all(ob.speed_m_s == 0 for ob in env.buildings))
        start = copy.copy(agent.position)
        reward, done, info = agent.update([0, 0, 0])
        self.assertFalse(done)
        self.assertEqual(info, 'normal')
        self.assertEqual((agent.Step, agent.Control_step), (2, 1))
        self.assertAlmostEqual(agent.position.x - start.x, 1 / 480, places=10)
        for _ in range(479):
            agent.update([0, 0, 0])
        self.assertAlmostEqual(agent.position.x - start.x, 1.0, places=8)
        self.assertEqual((agent.Step, agent.Control_step), (960, 480))
        self.assertAlmostEqual(agent.sim_time, 1.0)
        self.assertAlmostEqual(env.sim_time, 1.0)
        self.assertEqual(agent.original_max_step, 1152000)
        self.assertEqual([(ob.position.x, ob.position.y, ob.position.z) for ob in env.buildings], centers)
        self.assertEqual([(ob._R, ob._H) for ob in env.buildings], geometry)
        self.assertEqual(json.loads(env.getEnvJson())['dynamic_obstacles'], [])

    def test_static_stagnation_still_exports_purple_virtual_source_and_red_cylinders(self):
        env, agent = self.make_environment()
        env.Threaten_rate = lambda position: 1
        for _ in range(4801):
            agent.update([0, 0, 0])
        self.assertEqual(len(agent.active_virtual_obstacles), 1)
        self.assertEqual(agent.Max_Step - agent.original_max_step, 48000)
        captured = {}
        render = environment_module.Grid.render

        def capture_render(chart, path):
            captured.update(json.loads(chart.dump_options()))
            return render(chart, path)

        with patch.object(environment_module.Grid, 'render', capture_render):
            env.render({'uuid': 'static_virtual_source'})
        html = Path(env.last_render_path)
        self.assertTrue(html.is_file())
        exported = json.loads(html.with_suffix('.json').read_text(encoding='utf-8'))
        self.assertEqual(exported['dynamic_obstacles'], [])
        self.assertEqual(len(exported['virtual_obstacles']), 1)
        self.assertEqual(exported['virtual_obstacles'][0]['created_time_s'], 10.0)
        self.assertEqual(exported['virtual_obstacles'][0]['geometric_radius_m'], 0.0)
        series = captured['series']
        cylinders = [item for item in series if item['type'] == 'surface']
        self.assertEqual(len(cylinders), 26)
        self.assertTrue(all(item['itemStyle']['color'] == 'red' for item in cylinders))
        self.assertFalse(any(item.get('name') == 'Dynamic obstacle path' for item in series))
        source = next(item for item in series if item.get('name') == 'Virtual obstacle')
        self.assertEqual(source['itemStyle']['color'], 'purple')
        uav_path = next(item for item in series if item.get('name') == 'UAV path')
        self.assertEqual(uav_path['lineStyle']['color'], '#5070DD')

    def test_default_profile_is_static_and_mixed_remains_selectable(self):
        requested = []
        environment = types.SimpleNamespace(Agents=[types.SimpleNamespace(Trainer=object())])

        class FakeFactory:
            def Create_Env(self, parameters):
                requested.append(parameters)
                return environment

        sim = simulator_module.simulator.__new__(simulator_module.simulator)
        sim.EnvFactory = FakeFactory()
        sim.scene = 'static'
        sim.Init_From_XML()
        self.assertEqual(int(requested[-1]['Dynamic_Obstacles']['enabled']), 0)
        self.assertTrue(requested[-1]['Agent']['xml_path_agent'].endswith('UAV_Static.xml'))
        sim.scene = 'mixed'
        sim.Init_From_XML()
        self.assertEqual(int(requested[-1]['Dynamic_Obstacles']['enabled']), 1)
        self.assertTrue(requested[-1]['Agent']['xml_path_agent'].endswith('UAV.xml'))
        # Explicit XML paths remain usable regardless of the scene default.
        sim.scene = 'static'
        sim.Init_From_XML(ROOT / 'config/PathPlan_City.xml')
        self.assertEqual(int(requested[-1]['Dynamic_Obstacles']['enabled']), 1)


if __name__ == '__main__':
    unittest.main()
