"""Virtual point-source output and trustworthy episode metrics."""

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
for directory in (ROOT, ROOT / "Agents", ROOT / "BaseClass", ROOT / "Envs"):
    sys.path.insert(0, str(directory))

from BaseClass.CalMod import Loc, XML2Dict
from Obstacles.building import building

module = importlib.import_module("Envs.PathPlan_City")
Environment = module.PathPlan_City


class VirtualOutputTests(unittest.TestCase):
    def make_environment(self):
        env = Environment.__new__(Environment)
        env.len, env.width, env.h = 500, 500, 100
        env.num_UAV = 1
        env.epoch = 1
        env.sim_time = 10.0
        env.dynamic_enabled = True
        env.dynamic_speed = 4.0
        env.motion_half_range = None
        env.dynamic_buildings = []
        env.buildings_param = XML2Dict(str(ROOT / "config/buildings.xml"))["buildings"]
        env.buildings = [building(parameters, env)
                         for parameters in env.buildings_param["Threaten"]]
        env.param = {"UI": {"UI_path": str(ROOT / "config/UI.xml")}}
        env.Agents = [types.SimpleNamespace(
            control_dt=1 / 480, control_hz=480, Control_step=4800, Step=9600,
            step_units_per_second=960, controls_per_decision=480,
            base_duration_s=1200.0, original_max_step=1152000, Max_Step=1200000,
            steps_per_rtk=24, steps_per_lidar=48,
            position=Loc(120, 90, 0), goal=Loc(470, 470, 0),
            path=[[10, 10, 0], [120, 90, 0]], path_times=[0.0, 10.0],
            score=-3.75, path_len=136.0, sim_time=10.0, collision_count=2,
            virtual_influence_radius=15.0,
            active_virtual_obstacles=[{
                "id": 7, "pos": Loc(120, 90, 0), "force": 12.0,
                "created_time_s": 10.0}],
            virtual_obstacle_records=[])]
        env._configure_dynamic_obstacles()
        return env

    def test_virtual_point_sources_export_position_gain_and_creation_time(self):
        env = self.make_environment()
        data = json.loads(env.getEnvJson())
        self.assertEqual(data["virtual_obstacles"], [{
            "id": 7, "position_m": [120, 90, 0], "force_gain": 12.0,
            "created_time_s": 10.0, "influence_radius_m": 15.0,
            "geometric_radius_m": 0.0}])
        self.assertEqual(data["collision_count"], 2)
        self.assertEqual(data["timing"]["step_mode"], "time_consistent_s2r")
        self.assertEqual(data["timing"]["legacy_step_count"], 9600)
        self.assertEqual(data["timing"]["step_units_per_second"], 960)
        self.assertEqual(data["timing"]["base_step_budget"], 1152000)
        self.assertEqual(data["timing"]["active_duration_budget_s"], 1250)

    def test_purple_virtual_centers_keep_all_existing_journal_colors(self):
        env = self.make_environment()
        captured = {}
        render = module.Grid.render
        def inspect_render(chart, path):
            captured.update(json.loads(chart.dump_options()))
            return render(chart, path)
        old_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch.object(module.Grid, "render", inspect_render):
                    env.render({"uuid": "virtual_sources"})
                html = Path(env.last_render_path)
                self.assertTrue(html.is_file())
                self.assertIn("assets/echarts.min.js", html.read_text(encoding="utf-8"))
                data = json.loads(html.with_suffix(".json").read_text(encoding="utf-8"))
                self.assertEqual(data["virtual_obstacles"][0]["id"], 7)
            finally:
                os.chdir(old_directory)
        series = captured["series"]
        points = [item for item in series if item.get("name") == "Virtual obstacle"]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["type"], "scatter3D")
        self.assertEqual(points[0]["itemStyle"]["color"], "purple")
        self.assertEqual(points[0]["data"][0]["value"], [120, 90, 0])
        cylinders = [item for item in series if item["type"] == "surface"]
        self.assertEqual(len(cylinders), 26)
        self.assertEqual(sum(item["itemStyle"]["color"] == "red" for item in cylinders), 20)
        self.assertEqual(sum(item["itemStyle"]["color"] == "green" for item in cylinders), 6)
        paths = [item for item in series if item.get("name") == "Dynamic obstacle path"]
        self.assertEqual(len(paths), 6)
        self.assertTrue(all(item["lineStyle"]["color"] == "#90EE90" for item in paths))
        uav_path = [item for item in series if item.get("name") == "UAV path"]
        self.assertEqual(uav_path[0]["lineStyle"]["color"], "#5070DD")

    def test_render_restores_deleted_or_truncated_local_plotting_libraries(self):
        env = self.make_environment()
        vendor = ROOT / "assets" / "pyecharts"
        names = ("echarts.min.js", "echarts-gl.min.js")
        old_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                env.render({"uuid": "first_render"})
                target = Path(env.last_render_path).parent / "assets"
                for name in names:
                    self.assertEqual((target / name).read_bytes(), (vendor / name).read_bytes())
                # Simulate clearing one output dependency and truncating the other.
                (target / names[0]).unlink()
                (target / names[1]).write_bytes(b"truncated")
                env.render({"uuid": "second_render"})
                for name in names:
                    self.assertEqual((target / name).read_bytes(), (vendor / name).read_bytes())
                html = Path(env.last_render_path).read_text(encoding="utf-8")
                self.assertIn('src="assets/echarts.min.js"', html)
                self.assertIn('src="assets/echarts-gl.min.js"', html)
                self.assertNotIn("https://assets.pyecharts.org", html)
            finally:
                os.chdir(old_directory)

    def test_float_training_loss_and_measured_episode_metrics_are_not_zeroed(self):
        env = self.make_environment()
        env.Reset_Result(0.1)
        env.result["lose"] = 1
        env.Train_statistics([{"loss": 2.5, "sum_epoch": 17}])
        self.assertEqual(env.result["loss"], 2.5)
        self.assertEqual(env.result["sum_epoch"], 17)
        self.assertEqual(env.result["score"], -3.75)
        self.assertEqual(env.result["average_score"], -3.75)
        self.assertEqual(env.result["step"], 9600)
        self.assertEqual(env.result["control_updates"], 4800)
        self.assertEqual(env.result["simulation_time_s"], 10.0)
        self.assertEqual(env.result["path_len"], 136.0)
        self.assertEqual(env.result["virtual_obstacle_count"], 1)
        self.assertEqual(env.result["active_step_budget"], 1200000)
        self.assertEqual(env.result["collision_count"], 2)
        self.assertEqual(env.result["meet_threaten"], 2)


if __name__ == "__main__":
    unittest.main()
