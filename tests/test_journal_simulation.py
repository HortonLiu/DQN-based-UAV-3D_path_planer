"""Integration checks for journal obstacle counts, physical time, and HTML colors."""

import importlib
import json
import math
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

UAV = importlib.import_module('UAV_Single_RL+VOAPF S2R').UAV
environment_module = importlib.import_module('Envs.PathPlan_City')
PathPlanCity = environment_module.PathPlan_City


class EmptyEnvironment:
    len = 500
    width = 500
    h = 100

    def __init__(self):
        self.buildings = []
        self.sim_time = 0.0
        self.control_updates = 0

    def Threaten_rate(self, position):
        return 0

    def advance_dynamic_obstacles(self, dt):
        self.sim_time += dt
        self.control_updates += 1
        for obstacle in self.buildings:
            obstacle.run(dt)


class JournalEnvironmentTests(unittest.TestCase):
    def make_environment(self):
        env = PathPlanCity.__new__(PathPlanCity)
        env.len, env.width, env.h = 500, 500, 100
        env.num_UAV = 1
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
        env.Agents = [types.SimpleNamespace(
            control_dt=1.0 / 480.0, control_hz=480, Control_step=1,
            steps_per_rtk=24, steps_per_lidar=48, Step=2,
            step_units_per_second=960, reference_dt=1.0, controls_per_decision=480,
            base_duration_s=1200.0, original_max_step=1152000, Max_Step=1152000,
            virtual_influence_radius=15.0, active_virtual_obstacles=[],
            virtual_obstacle_records=[], collision_count=0,
            position=Loc(11, 11, 0), goal=Loc(470, 470, 0),
            path=[[10, 10, 0], [11, 11, 0]], path_times=[0.0, 1.0 / 480.0],
            reset=lambda: None)]
        env.param = {'UI': {'UI_path': str(ROOT / 'config/UI.xml')}}
        return env

    def test_six_existing_cylinders_move_and_reset_without_geometry_change(self):
        env = self.make_environment()
        positions = [(ob.position.x, ob.position.y, ob.position.z) for ob in env.buildings]
        geometry = [(ob._R, ob._H) for ob in env.buildings]
        env._configure_dynamic_obstacles()
        self.assertEqual(len(env.buildings), 26)
        self.assertEqual(sum(ob.is_dynamic for ob in env.buildings), 6)
        self.assertEqual(len(env.dynamic_buildings), 6)
        self.assertEqual([(ob._R, ob._H) for ob in env.buildings], geometry)
        self.assertEqual([(ob.position.x, ob.position.y, ob.position.z)
                          for ob in env.buildings], positions)
        modes = [(abs(ob.velocity.x) > 0, abs(ob.velocity.y) > 0)
                 for ob in env.dynamic_buildings]
        self.assertEqual(modes.count((True, False)), 2)
        self.assertEqual(modes.count((False, True)), 2)
        self.assertEqual(modes.count((True, True)), 2)
        for _ in range(480):
            env.advance_dynamic_obstacles(1.0 / 480.0)
        self.assertAlmostEqual(env.sim_time, 1.0)
        self.assertTrue(any((ob.position.x, ob.position.y, ob.position.z) != initial
                            for ob, initial in zip(env.dynamic_buildings, positions)))
        env.Scene_Random_Reset()  # Same map on the second episode: reset motion phase.
        self.assertEqual(env.sim_time, 0.0)
        self.assertEqual([(ob.position.x, ob.position.y, ob.position.z)
                          for ob in env.buildings], positions)
        self.assertTrue(all(len(ob.history) == 1 for ob in env.dynamic_buildings))
        self.assertEqual([(ob._R, ob._H) for ob in env.buildings], geometry)

    def test_disable_motion_keeps_all_twenty_six_objects_static(self):
        env = self.make_environment()
        env._configure_dynamic_obstacles()
        env.dynamic_enabled = False
        env._configure_dynamic_obstacles()
        self.assertEqual(len(env.buildings), 26)
        self.assertFalse(any(ob.is_dynamic for ob in env.buildings))
        self.assertEqual(env.dynamic_buildings, [])

    def test_html_series_use_journal_obstacle_and_trajectory_colors(self):
        env = self.make_environment()
        env._configure_dynamic_obstacles()
        captured = {}
        original_render = environment_module.Grid.render

        def capture_render(chart, path):
            captured['options'] = json.loads(chart.dump_options())
            return original_render(chart, path)

        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch.object(environment_module.Grid, 'render', capture_render):
                    env.render({'uuid': 'journal_colors'})
                html = Path(env.last_render_path)
                self.assertTrue(html.is_file())
                self.assertGreater(html.stat().st_size, 1000)
                exported = json.loads(html.with_suffix('.json').read_text(encoding='utf-8'))
                self.assertEqual(len(exported['dynamic_obstacles']), 6)
            finally:
                os.chdir(previous_cwd)
        series = captured['options']['series']
        cylinders = [item for item in series if item['type'] == 'surface']
        self.assertEqual(len(cylinders), 26)
        self.assertEqual(sum(item['itemStyle']['color'] == 'red' for item in cylinders), 20)
        self.assertEqual(sum(item['itemStyle']['color'] == 'green' for item in cylinders), 6)
        tracks = [item for item in series if item.get('name') == 'Dynamic obstacle path']
        self.assertEqual(len(tracks), 6)
        for track in tracks:
            self.assertEqual(track['lineStyle']['color'], '#90EE90')
            self.assertEqual(track['lineStyle']['width'], 7)
        path = [item for item in series if item.get('name') == 'UAV path']
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0]['lineStyle']['color'], '#5070DD')


if __name__ == '__main__':
    unittest.main()
