import datetime
import json
import time
from BaseClass.BaseEnv import *
from BaseClass.CalMod import *
import torch
import numpy as np
import math
import random
import csv
import threading
import copy
import os
import uuid
import shutil
from pathlib import Path
import pyecharts.options as opts
from pyecharts.charts import Line3D, Scatter3D, Surface3D, Grid
from DataBase.Connector import MySQLConnector

class PathPlan_City(BaseEnv):
    def __init__(self, param: dict) -> None:
        super().__init__(param)
        self.eps = float(None2Value(param.get('eps'), 0.1))
        self.Is_On_Policy = int(param.get('Is_On_Policy'))
        self.param = param
        hardware_flag = int(param.get('Hardware_Hz_Enabled', 1))
        if hardware_flag not in (0, 1):
            raise ValueError('Hardware_Hz_Enabled must be 0 or 1')
        self.hardware_hz_enabled = bool(hardware_flag)
        self.uav_speed_override = param.get('UAV_Speed_M_S')
        if self.uav_speed_override is not None:
            try:
                self.uav_speed_override = float(self.uav_speed_override)
            except (TypeError, ValueError) as error:
                raise ValueError('UAV_Speed_M_S must be 1 or 4 m/s') from error
            if self.uav_speed_override not in (1.0, 4.0):
                raise ValueError('UAV_Speed_M_S must be 1 or 4 m/s')
        
        self.len = int(param.get('len', 500))
        self.width = int(param.get('width', 500))
        self.h = int(param.get('h', 100))

        self.map = np.zeros((self.width, self.len)) 
        self.buildings = []
        self.map_reset_loop = 2
        motion = param.get('Dynamic_Obstacles') or {}
        self.dynamic_enabled = bool(int(motion.get('enabled', 1)))
        self.dynamic_speed = float(motion.get('speed', 4.0))
        self.motion_half_range = (float(motion['half_range'])
                                  if motion.get('half_range') is not None else None)
        self.sim_time = 0.0
        self.dynamic_buildings = []
        
        threaten_params = param.get('Obstacles')
        if threaten_params is not None:
            buildings_param_path = os.path.normpath(threaten_params.get('buildings'))
            threaten_config_dict = XML2Dict(buildings_param_path)
            self.buildings_param = threaten_config_dict.get('buildings')
            
            if self.buildings_param is not None:
                threatens = self.buildings_param.get("Threaten")
                if not isinstance(threatens, list):
                    threatens = [threatens]

                for param_threaten in threatens:
                    obj_threaten = self.ThreatenFactory.Create_Threaten(param_threaten, self)
                    self.buildings.append(obj_threaten)

        self.num_UAV = int(param.get('num_UAV'))
        agents_params = param.get('Agent')
        agents_xml_path = os.path.normpath(agents_params['xml_path_agent'])
        config_dict = XML2Dict(agents_xml_path)
        uav_params = config_dict.get('Agent')
        uav_params['Hardware_Hz_Enabled'] = int(self.hardware_hz_enabled)
        if self.uav_speed_override is not None:
            # Apply before construction, including velocity and power initialization.
            uav_params['Max_V'] = self.uav_speed_override
        for i in range(self.num_UAV):
            uav_params['name'] = 'UAV_' + str(i)
            uav_params['j'] = i
            obj_agent = self.AgentFactory.Create_Agent(uav_params, self)
            if obj_agent is None:
                raise RuntimeError("Cannot initialize UAV %r from %s" %
                                   (uav_params['name'], agents_xml_path))
            trainer_xml_path = agents_params['Trainer']
            trainer_xml_path = os.path.normpath(trainer_xml_path.get('Trainer_path'))
            trainer_config_dict = XML2Dict(trainer_xml_path)
            agent_trainer = trainer_config_dict.get('Trainer')
            agent_trainer['name'] = obj_agent.name
            if not self.hardware_hz_enabled:
                # Coarse updates use different dynamics; preserve existing 480 Hz weights.
                agent_trainer['checkpoint_dir'] = os.path.join(
                    agent_trainer.get('checkpoint_dir') or 'Mod', 'one_second')
            if self.uav_speed_override is not None:
                # Keep explicit speed selections separate from legacy mixed-speed runs.
                agent_trainer['checkpoint_dir'] = os.path.join(
                    agent_trainer.get('checkpoint_dir') or 'Mod',
                    'v%d' % self.uav_speed_override)
            obj_agent.Trainer = self.TrainerFactory.Create_Trainer(agent_trainer)
            if obj_agent.Trainer is None:
                raise RuntimeError("Trainer initialization returned None for %r from %s" %
                                   (obj_agent.name, trainer_xml_path))
            self.Agents.append(obj_agent)

        self.result = {'success': 0, 'lose': 0, 'meet_threaten': 0, 'normal': 0, 'loss': None, 'sum_epoch': 0, 'eps': 0.1}
        self.epoch = 0
        self.print_loop = int(param.get('print_loop'))
        self.Is_FL = int(param.get('Is_FL'))
        self.FL_Loop = int(param.get('FL_Loop'))
        self.threads = []
        self.executed_time = 0
        self.SQL_connector = None 

    def Scene_Random_Reset(self):
        self.reset_threads = []
        # Restore the same initial layout on the second episode sharing this map.
        for obstacle in self.dynamic_buildings:
            obstacle.reset_motion()
        if self.epoch == 0 or self.epoch % self.map_reset_loop == 0:
            safe_zones = [
                (10, 210, 1, 10),      # 起點區域
                (330, 490, 420, 490)   # 終點區域
            ]

            if self.buildings_param is not None:
                threaten_data = self.buildings_param.get("Threaten")
                if not isinstance(threaten_data, list):
                    threaten_data = [threaten_data]

                for i, b in enumerate(self.buildings):
                    if hasattr(b, 'reset_random'):
                        b.reset_random(safe_zones)
                    
                    if i < len(threaten_data):
                        threaten_data[i]['position']['x'] = str(b.position.x)
                        threaten_data[i]['position']['y'] = str(b.position.y)

            self.map = self.Draw_threaten_map()

        self._configure_dynamic_obstacles()
        self.sim_time = 0.0
        for indx, uav in enumerate(self.Agents):
            uav.reset()

    def _configure_dynamic_obstacles(self):
        # Six of the original 26 objects move; preserve their sampled radii/positions.
        for obstacle in self.buildings:
            obstacle.clear_motion()
        self.dynamic_buildings = []
        if not self.dynamic_enabled:
            return
        if len(self.buildings) != 26 or self.num_UAV != 1:
            raise ValueError("The journal mixed scene requires 26 obstacles and one UAV")
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))
        dt = self.Agents[0].control_dt
        for obstacle, direction in zip(self.buildings[:6], directions):
            obstacle.configure_motion(direction, self.dynamic_speed, control_dt=dt,
                                      half_range=self.motion_half_range)
            self.dynamic_buildings.append(obstacle)

    def advance_dynamic_obstacles(self, dt):
        for obstacle in self.dynamic_buildings:
            obstacle.run(dt)
        self.sim_time += dt

    @staticmethod
    def _virtual_source_data(agent):
        """Serialize point sources without treating their influence as geometry."""
        active = getattr(agent, 'active_virtual_obstacles', None)
        records = active if active else getattr(agent, 'virtual_obstacle_records', [])
        sources = []
        for index, record in enumerate(records):
            pos = record['pos']
            coordinate = [pos.x, pos.y, pos.z] if hasattr(pos, 'x') else list(pos)
            sources.append({
                'id': record.get('id', index + 1),
                'position_m': coordinate,
                'force_gain': float(record.get('force', record.get('max_force', 0.0))),
                'created_time_s': float(record.get('created_time_s', 0.0)),
                'influence_radius_m': float(getattr(agent, 'virtual_influence_radius', 15.0)),
                'geometric_radius_m': 0.0,
            })
        return sources

    def getEnvJson(self):
        environment = copy.deepcopy(self.buildings_param or {})
        environment['path'] = []
        environment['epoch'] = self.epoch
        environment['simulation_time_s'] = self.sim_time
        environment['virtual_obstacles'] = []
        if self.Agents:
            agent = self.Agents[0]
            scale = float(getattr(agent, 'step_units_per_second', 2.0 / agent.control_dt))
            environment['path'] = list(agent.path)
            environment['path_time_s'] = list(getattr(agent, 'path_times', []))
            environment['goal_position_m'] = [agent.goal.x, agent.goal.y, agent.goal.z]
            environment['uav_speed_m_s'] = (float(agent.Max_V)
                                             if hasattr(agent, 'Max_V') else None)
            environment['virtual_obstacles'] = self._virtual_source_data(agent)
            environment['collision_count'] = int(getattr(agent, 'collision_count', 0))
            environment['timing'] = {
                'hardware_hz_enabled': bool(getattr(agent, 'hardware_hz_enabled',
                                                     agent.control_dt < 1.0)),
                'control_dt_s': float(agent.control_dt),
                'control_hz': getattr(agent, 'control_hz', 1.0 / agent.control_dt),
                'control_ticks': getattr(agent, 'Control_step', 0),
                'step_mode': 'time_consistent_s2r',
                'legacy_step_count': agent.Step,
                'step_units_per_second': scale,
                'base_duration_s': float(getattr(agent, 'base_duration_s', 0.0)),
                'base_step_budget': int(getattr(agent, 'original_max_step', 0)),
                'active_step_budget': int(getattr(agent, 'Max_Step', 0)),
                'active_duration_budget_s': float(getattr(agent, 'Max_Step', 0)) / scale,
                'controls_per_decision': int(getattr(agent, 'controls_per_decision', 1)),
                'rtk_refresh_control_period': getattr(agent, 'steps_per_rtk', None),
                'lidar_refresh_control_period': getattr(agent, 'steps_per_lidar', None),
                'rtk_hz': getattr(agent, 'freq_rtk', None),
                'lidar_hz': getattr(agent, 'freq_lidar', None),
            }
        environment['dynamic_obstacles'] = [{
            'index': index, 'radius_m': obstacle._R,
            'speed_m_s': obstacle.speed_m_s,
            'position_m': [obstacle.position.x, obstacle.position.y, obstacle.position.z],
            'motion_endpoints_m': obstacle.motion_endpoints,
            'trajectory_txyz': obstacle.history,
        } for index, obstacle in enumerate(self.buildings) if obstacle.is_dynamic]
        return json.dumps(environment)

    @staticmethod
    def _ensure_plot_assets(output_directory):
        """Restore local HTML dependencies after the experience folder is cleaned."""
        vendor_directory = Path(__file__).resolve().parents[1] / 'assets' / 'pyecharts'
        output_assets = Path(output_directory) / 'assets'
        output_assets.mkdir(parents=True, exist_ok=True)
        for name in ('echarts.min.js', 'echarts-gl.min.js'):
            source = vendor_directory / name
            if not source.is_file():
                raise FileNotFoundError('Missing bundled plotting library: %s' % source)
            destination = output_assets / name
            if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
                shutil.copyfile(source, destination)

    def render(self, data=None):
        init_opts, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts = self.set_grid()
        path_data = self.Agents[0].path if len(self.Agents) > 0 else []
            
        path = self.draw_path(path_data, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts)
        threaten_3D = self.draw_threaten_3D(None, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts)
        markers = self.draw_start_end(x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts)
        virtual_points = self.draw_virtual_obstacles(x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts)
        
        path.set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
        threaten_3D.set_global_opts(legend_opts=opts.LegendOpts(is_show=False))

        if markers:
            markers.set_global_opts(legend_opts=opts.LegendOpts(is_show=True, pos_top="2%", pos_left="center"))

        grid = Grid(init_opts=init_opts)
        # Ship plotting libraries beside the HTML so local results work offline.
        grid.js_host = 'assets/'
        for obstacle in self.dynamic_buildings:
            track = Line3D(init_opts=init_opts).add(
                series_name="Dynamic obstacle path", data=obstacle.motion_endpoints,
                xaxis3d_opts=x_axis3d_opts, yaxis3d_opts=y_axis3d_opts,
                zaxis3d_opts=z_axis3d_opts, grid3d_opts=grid3d_opts,
                itemstyle_opts=opts.ItemStyleOpts(color="#90EE90"))
            track.set_series_opts(linestyle_opts=opts.LineStyleOpts(color="#90EE90", width=7))
            track.set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
            grid.add(track, grid_opts=opts.GridOpts())
        grid.add(path, grid_opts=opts.GridOpts())
        grid.add(threaten_3D, grid_opts=opts.GridOpts())
        if virtual_points:
            grid.add(virtual_points, grid_opts=opts.GridOpts())
        if markers:
            grid.add(markers, grid_opts=opts.GridOpts())
            
        path_json = self.getEnvJson()
        if data is None:
            import uuid
            data = {'uuid': str(uuid.uuid4())}
        data2 = {'uuid': data['uuid'], 'json': path_json}
        
        save_dir = 'DataBase/experience/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self._ensure_plot_assets(save_dir)
        save_path = save_dir + data['uuid'] + '.html'
        data['html_dir'] = save_path
        grid.render(save_path)
        self.last_render_path = os.path.abspath(save_path)
        with open(save_path[:-5] + '.json', 'w', encoding='utf-8') as output:
            output.write(path_json)

    def draw_start_end(self, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts):
        if len(self.Agents) == 0: return None
        agent = self.Agents[0]
        start_pos = [[agent.path[0][0], agent.path[0][1], agent.path[0][2]]] if len(agent.path) > 0 else [[agent.position.x, agent.position.y, agent.position.z]]
        goal_pos = [[agent.goal.x, agent.goal.y, agent.goal.z]]

        scatter = Scatter3D(init_opts=init_opts)
        scatter.add(series_name="Start Point", data=start_pos, itemstyle_opts=opts.ItemStyleOpts(color="#00FF00", opacity=1))
        scatter.add(series_name="Goal Point", data=goal_pos, itemstyle_opts=opts.ItemStyleOpts(color="#FF0000", opacity=1))
        return scatter

    def draw_virtual_obstacles(self, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts):
        if not self.Agents:
            return None
        sources = self._virtual_source_data(self.Agents[0])
        if not sources:
            return None
        points = Scatter3D(init_opts=init_opts).add(
            series_name="Virtual obstacle", data=[{
                'name': 'Virtual #%s | gain %.3f | created %.3f s' %
                        (source['id'], source['force_gain'], source['created_time_s']),
                'value': source['position_m'],
            } for source in sources],
            xaxis3d_opts=x_axis3d_opts, yaxis3d_opts=y_axis3d_opts,
            zaxis3d_opts=z_axis3d_opts, grid3d_opts=grid3d_opts,
            itemstyle_opts=opts.ItemStyleOpts(color='purple', opacity=1.0))
        points.set_series_opts(symbolSize=9)
        points.set_global_opts(legend_opts=opts.LegendOpts(
            is_show=True, pos_top="2%", pos_left="left"))
        return points

    def draw_threaten_3D(self, marked_points, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts):
        re = Surface3D()
        for th in self.buildings:
            re.add(series_name="", shading="color", data=list(surface3d_data(th)), xaxis3d_opts=x_axis3d_opts, yaxis3d_opts=y_axis3d_opts, zaxis3d_opts=z_axis3d_opts, grid3d_opts=grid3d_opts, itemstyle_opts=opts.ItemStyleOpts(color='green' if th.is_dynamic else 'red', opacity=0.7))
        return re

    def set_grid(self):
        ui_param = self.param.get('UI')
        ui_param_path = os.path.normpath(ui_param.get('UI_path'))
        ui_all_dict = XML2Dict(ui_param_path)
        self.ui_dict = ui_all_dict.get('UI')
        InitOpts_dict = self.ui_dict.get('InitOpt')
        init_opts = opts.InitOpts(width=InitOpts_dict.get('width').strip('\"'), height=InitOpts_dict.get('height').strip('\"'))

        self.x_axis3d_opts_dict = self.ui_dict.get('x_axis3d_opts')
        x_axis3d_opts = opts.Axis3DOpts(type_="value", max_=self.x_axis3d_opts_dict.get('MAX'), min_=self.x_axis3d_opts_dict.get('MIN'), split_number=self.x_axis3d_opts_dict.get('split_number'))
        self.y_axis3d_opts_dict = self.ui_dict.get('y_axis3d_opts')
        y_axis3d_opts = opts.Axis3DOpts(type_="value", max_=self.y_axis3d_opts_dict.get('MAX'), min_=self.y_axis3d_opts_dict.get('MIN'), split_number=self.y_axis3d_opts_dict.get('split_number'))
        self.z_axis3d_opts_dict = self.ui_dict.get('z_axis3d_opts')
        z_axis3d_opts = opts.Axis3DOpts(type_="value", max_=self.h, min_=self.z_axis3d_opts_dict.get('MIN'), split_number=5)
        self.grid3dopts_dict = self.ui_dict.get('Grid3DOpt')
        grid3dopts = opts.Grid3DOpts(width=self.grid3dopts_dict.get('width'), height=40, depth=self.grid3dopts_dict.get('depth'), view_control_alpha=60, view_control_beta=40)
        grid3dopts.opts['viewControl']['distance'] = 400
        return init_opts, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3dopts

    def draw_path(self, path_data, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts):
        path = Line3D(init_opts=init_opts).add(series_name="UAV path", shading="color", data=path_data, xaxis3d_opts=x_axis3d_opts, yaxis3d_opts=y_axis3d_opts, zaxis3d_opts=z_axis3d_opts, grid3d_opts=grid3d_opts, itemstyle_opts=opts.ItemStyleOpts(color="#5070DD"))
        path.set_series_opts(linestyle_opts=opts.LineStyleOpts(color="#5070DD", width=2))
        return path

    def Threaten_rate(self, p: Loc):
        if p.x < 0 or p.x > self.width or p.y < 0 or p.y > self.width or p.z < 0 or p.z > self.h:
            return 1
        for th in self.buildings:
            if th.check_threaten(p) > 0: return 1
        return 0

    def Draw_threaten_map(self, h=0):
        # Preserve the original map[x, y] convention while avoiding millions of calls.
        x, y = np.ogrid[:self.width, :self.len]
        occupancy = np.zeros((self.width, self.len), dtype=float)
        if h < 0 or h > self.h:
            occupancy.fill(1.0)
            return occupancy
        for obstacle in self.buildings:
            if h <= obstacle._H:
                occupied = ((x - obstacle.position.x) ** 2
                            + (y - obstacle.position.y) ** 2 < obstacle._R ** 2)
                occupancy[occupied] = 1.0
        return occupancy

    def Check_uav_Done(self):
        for agent in self.Agents:
            if not agent.done: return False
        return True

    def Choose_Action2(self, index: int, eps=0.2):
        state = self.Agents[index].state()
        return self.Agents[index].get_action(state, eps)

    def Reset_Result(self, eps_rate):
        self.result = {'success': 0, 'lose': 0, 'meet_threaten': 0, 'normal': 0, 'loss': 0, 'sum_epoch': 0, 'eps': eps_rate, 'score': 0, 'average_score': 0, 'step': 0}

    def run_thread_OffPolicy(self, uav, ind, eps_rate):
        if uav.done:
            return
        state_test = uav.state()
        action = self.Choose_Action2(ind, eps_rate)
        reward = 0.0
        next_state = state_test
        done = False
        # SAC selects one action per reference second. Depending on the mode,
        # advance 480 small updates or one one-second update, each with Step += 2.
        controls = max(1, int(getattr(uav, 'controls_per_decision', 1)))
        for _ in range(controls):
            next_state, tick_reward, done, info = self.Move_Agent(ind, action)
            reward += float(tick_reward)
            self.Run_statistics(info)
            if done:
                break
        if not int(getattr(uav.Trainer, 'Is_Train', 1)):
            return
        uav.Trainer.Push_Replay((FloatTensor(np.asarray([state_test])), action,
                                FloatTensor([[reward]]), FloatTensor(np.asarray([next_state])),
                                FloatTensor([[done]])))
        uav.Trainer.replay_memory.add(state_test, action, reward, next_state, done)
        if len(uav.Trainer.replay_memory.buffer) > uav.Trainer.Batch_Size:
            b_s, b_a, b_r, b_ns, b_d, idx, weights = uav.Trainer.replay_memory.sample2(uav.Trainer.Batch_Size)
            uav.transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns,
                                   'rewards': b_r, 'dones': b_d, 'idx': idx, 'weights': weights}

    def run_eposide(self, eps_rate=0.1):
        self.Reset_Result(eps_rate)
        self.Scene_Random_Reset()
        last_progress_report = time.monotonic()
        while True:
            for ind, uav in enumerate(self.Agents):
                uav.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                self.run_thread_OffPolicy(uav, ind, eps_rate)
            train_info = self.update()
            if self.Check_uav_Done():
                break
            if time.monotonic() - last_progress_report >= 30.0:
                agent = self.Agents[0]
                print("Episode %d | simulated %.1f s | Step %d / %d | goal %.1f m | virtual %d" %
                      (self.epoch + 1, getattr(agent, 'sim_time', 0.0), agent.Step,
                       agent.Max_Step, Eu_Loc_distance(agent.position, agent.goal),
                       len(getattr(agent, 'active_virtual_obstacles', []))), flush=True)
                last_progress_report = time.monotonic()
                
        self.Train_statistics(train_info)
        for uav in self.Agents:
            uav.last_episode_result = dict(self.result)
            uav.last_episode_result['episode'] = self.epoch + 1
        self.epoch += 1
        for uav in self.Agents:
            uav.record_list()
        
        data_result = self.generate_train_result()
        self.render(data_result)
        self.result['html_path'] = self.last_render_path
        return self.result

    def generate_train_result(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            'uuid': str(uuid.uuid4()), 'date': current_time, 'html_dir': '<html lang="en"></html>',
            'train_score': self.Agents[0].total_score, 'epoch': self.Agents[0].Train_epoch, 'Is_Scored': 0,
            'ReachGoal': self.Agents[0].reach_goal, 'train_time': time.time() - self.Agents[0].Train_start,
            'path_len': self.Agents[0].path_len, 'score_quality': 0, 'score_len': 0, 'score_threaten': 0,
            'start2goal': self.Agents[0].start2goal, 'len_Astar': 0,
        }

    def Run_statistics(self, info):
        self.result[info] = self.result[info] + 1

    @staticmethod
    def _scalar(value):
        if hasattr(value, 'item'):
            return float(value.item())
        return float(value)

    def Train_statistics(self, train_info):
        number = max(1, len(train_info))
        self.result['loss'] = sum(self._scalar(info.get('loss', 0.0)) for info in train_info) / number
        self.result['sum_epoch'] = sum(self._scalar(info.get('sum_epoch', 0.0)) for info in train_info) / number
        self.result['score'], self.result['average_score'] = self.Cal_Score()
        agents = self.Agents
        count = max(1, len(agents))
        self.result['step'] = sum(agent.Step for agent in agents) / count
        self.result['control_updates'] = sum(getattr(agent, 'Control_step', 0) for agent in agents)
        self.result['simulation_time_s'] = max((getattr(agent, 'sim_time', 0.0) for agent in agents), default=0.0)
        self.result['path_len'] = sum(getattr(agent, 'path_len', 0.0) for agent in agents) / count
        self.result['virtual_obstacle_count'] = sum(len(getattr(agent, 'active_virtual_obstacles', [])) for agent in agents)
        self.result['base_duration_s'] = max((getattr(agent, 'base_duration_s', 0.0) for agent in agents), default=0.0)
        self.result['active_step_budget'] = max((getattr(agent, 'Max_Step', 0) for agent in agents), default=0)
        self.result['collision_count'] = sum(getattr(agent, 'collision_count', 0) for agent in agents)
        self.result['meet_threaten'] = self.result['collision_count']

    def Cal_Score(self):
        sum_score = sum(agent.score for agent in self.Agents)
        return sum_score, sum_score / len(self.Agents)

    def update(self):
        re = []
        for uav in self.Agents:
            item = uav.Train_nn()
            item['score'] = uav.score; item['average_score'] = uav.score; item['step'] = uav.Step
            item['energy_cost'] = uav.energy_cost_total; item['task_collect'] = uav.task_collect
            item['Energy_Efficent'] = uav.task_collect / (uav.energy_cost_total + 0.001)
            item['UE_waiting_time'] = sum(ue.Caculate_wait_time for ue in uav.UEs) / (len(uav.UEs) + 1)
            re.append(item)
        return re
