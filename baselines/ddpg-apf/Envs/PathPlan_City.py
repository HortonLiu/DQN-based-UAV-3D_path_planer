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
import pyecharts.options as opts
from pyecharts.charts import Line3D, Scatter3D, Surface3D, Grid
from DataBase.Connector import MySQLConnector

class PathPlan_City(BaseEnv):
    def __init__(self, param: dict) -> None:
        super().__init__(param)
        self.eps = float(None2Value(param.get('eps'), 0.1))
        self.Is_On_Policy = int(param.get('Is_On_Policy'))
        self.param = param
        
        self.len = int(param.get('len', 500))
        self.width = int(param.get('width', 500))
        self.h = int(param.get('h', 100))
        self.fixed_obstacles = int(None2Value(param.get('Fixed_Obstacles'), 1))
        self.obstacle_count = int(None2Value(param.get('Obstacle_Count'), 15))
        self.obstacle_clearance = float(None2Value(param.get('Obstacle_Clearance'), 2.0))
        self.allow_obstacle_overlap = int(None2Value(param.get('Allow_Obstacle_Overlap'), 0))
        self.start_goal_clearance = float(None2Value(param.get('Start_Goal_Clearance'), 15.0))
        self.obstacle_reset_max_retries = int(None2Value(param.get('Obstacle_Reset_Max_Retries'), 3000))
        self.random_obstacle_radius_min = float(None2Value(param.get('Random_Obstacle_Radius_Min'), 10.0))
        self.random_obstacle_radius_max = float(None2Value(param.get('Random_Obstacle_Radius_Max'), 45.0))
        if self.random_obstacle_radius_min > self.random_obstacle_radius_max:
            self.random_obstacle_radius_min, self.random_obstacle_radius_max = self.random_obstacle_radius_max, self.random_obstacle_radius_min
        fixed_height = param.get('Fixed_Obstacle_Height')
        self.fixed_obstacle_height = None if fixed_height is None else float(fixed_height)
        self.map_initialized = False

        self.map = np.zeros((self.width, self.len)) 
        self.buildings = []
        self.base_buildings = []
        self.base_threatens = []
        self.map_reset_loop = 2 
        
        threaten_params = param.get('Obstacles')
        if threaten_params is not None:
            buildings_param_path = os.path.normpath(threaten_params.get('buildings'))
            threaten_config_dict = XML2Dict(buildings_param_path)
            self.buildings_param = threaten_config_dict.get('buildings')
            
            if self.buildings_param is not None:
                threatens = self.buildings_param.get("Threaten")
                if not isinstance(threatens, list):
                    threatens = [threatens]
                if self.obstacle_count > 0:
                    threatens = threatens[:self.obstacle_count]
                    self.buildings_param["Threaten"] = threatens
                self.base_threatens = copy.deepcopy(threatens)

                for param_threaten in threatens:
                    obj_threaten = self.ThreatenFactory.Create_Threaten(param_threaten, self)
                    self.buildings.append(obj_threaten)
                self._apply_fixed_obstacle_height(self.buildings, threatens)
                self.base_buildings = list(self.buildings)

        self.num_UAV = int(param.get('num_UAV'))
        agents_params = param.get('Agent')
        agents_xml_path = os.path.normpath(agents_params['xml_path_agent'])
        config_dict = XML2Dict(agents_xml_path)
        uav_params = config_dict.get('Agent')
        for i in range(self.num_UAV):
            uav_params['name'] = 'UAV_' + str(i)
            uav_params['j'] = i
            obj_agent = self.AgentFactory.Create_Agent(uav_params, self)
            trainer_xml_path = agents_params['Trainer']
            trainer_xml_path = os.path.normpath(trainer_xml_path.get('Trainer_path'))
            trainer_config_dict = XML2Dict(trainer_xml_path)
            agent_trainer = trainer_config_dict.get('Trainer')
            agent_trainer['name'] = obj_agent.name
            obj_agent.Trainer = self.TrainerFactory.Create_Trainer(agent_trainer)
            self.Agents.append(obj_agent)

        self.result = {'success': 0, 'lose': 0, 'meet_threaten': 0, 'normal': 0, 'loss': None, 'sum_epoch': 0, 'eps': 0.1}
        self.epoch = 0
        self.print_loop = int(param.get('print_loop'))
        self.Is_FL = int(param.get('Is_FL'))
        self.FL_Loop = int(param.get('FL_Loop'))
        self.threads = []
        self.executed_time = 0
        self.SQL_connector = None 

    def _apply_fixed_obstacle_height(self, buildings, threatens=None):
        if self.fixed_obstacle_height is None:
            return
        for building in buildings:
            building._H = self.fixed_obstacle_height
        if threatens is None:
            return
        if not isinstance(threatens, list):
            threatens = [threatens]
        for threaten in threatens:
            threaten['_H'] = str(self.fixed_obstacle_height)

    def Scene_Random_Reset(self):
        self.reset_threads = []
        if self.fixed_obstacles == 1:
            self.buildings = list(self.base_buildings)
            if not self.map_initialized:
                self.map = self.Draw_threaten_map()
                self.map_initialized = True
            for indx, uav in enumerate(self.Agents):
                uav.reset()
            return

        for indx, uav in enumerate(self.Agents):
            uav.reset()

        safe_zones = []
        protected_points = self._start_goal_protected_points()
        self._reset_random_non_overlapping_buildings(safe_zones, protected_points)
        self.map = self.Draw_threaten_map()

    def _start_goal_protected_points(self):
        protected_points = []
        for uav in self.Agents:
            protected_points.append((uav.position.x, uav.position.y, self.start_goal_clearance))
            protected_points.append((uav.goal.x, uav.goal.y, self.start_goal_clearance))
        return protected_points

    def _overlaps_safe_zone(self, x, y, radius, safe_zones):
        for x_min, x_max, y_min, y_max in safe_zones:
            if x + radius > x_min and x - radius < x_max and y + radius > y_min and y - radius < y_max:
                return True
        return False

    def _overlaps_buildings(self, x, y, radius, placed_buildings):
        for building in placed_buildings:
            distance = math.sqrt((x - building.position.x) ** 2 + (y - building.position.y) ** 2)
            if distance < radius + building._R + self.obstacle_clearance:
                return True
        return False

    def _overlaps_protected_points(self, x, y, radius, protected_points):
        for px, py, clearance in protected_points:
            distance = math.sqrt((x - px) ** 2 + (y - py) ** 2)
            if distance < radius + clearance:
                return True
        return False

    def _sample_random_obstacle_radius(self):
        return random.uniform(self.random_obstacle_radius_min, self.random_obstacle_radius_max)

    def _try_place_building(self, building, safe_zones, placed_buildings, radius, protected_points=None):
        protected_points = protected_points or []
        min_x = radius
        max_x = self.len - radius
        min_y = radius
        max_y = self.width - radius
        if min_x >= max_x or min_y >= max_y:
            return False
        for _ in range(self.obstacle_reset_max_retries):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            if self._overlaps_safe_zone(x, y, radius, safe_zones):
                continue
            if self._overlaps_protected_points(x, y, radius, protected_points):
                continue
            if self.allow_obstacle_overlap == 0 and self._overlaps_buildings(x, y, radius, placed_buildings):
                continue
            building._R = radius
            if self.fixed_obstacle_height is not None:
                building._H = self.fixed_obstacle_height
            building.position.x = x
            building.position.y = y
            building.position_ = building.position
            placed_buildings.append(building)
            return True
        return False

    def _reset_random_non_overlapping_buildings(self, safe_zones, protected_points=None):
        protected_points = protected_points or []
        placed_buildings = []
        candidate_buildings = list(self.base_buildings[:self.obstacle_count]) if self.obstacle_count > 0 else list(self.base_buildings)
        for building in candidate_buildings:
            placed = False
            for _ in range(self.obstacle_reset_max_retries):
                radius = self._sample_random_obstacle_radius()
                if not self._try_place_building(building, safe_zones, placed_buildings, radius, protected_points):
                    continue
                placed = True
                break
            if not placed:
                if not self._try_place_building(building, safe_zones, placed_buildings, self.random_obstacle_radius_min, protected_points):
                    raise RuntimeError("Could not place all non-overlapping random obstacles. Reduce Random_Obstacle_Radius_Min/Max or obstacle count.")

        self.buildings = placed_buildings
        if self.buildings_param is not None:
            threaten_data = copy.deepcopy(self.base_threatens)
            if not isinstance(threaten_data, list):
                threaten_data = [threaten_data]
            active_threatens = threaten_data[:len(self.buildings)]
            for i, building in enumerate(self.buildings):
                active_threatens[i]['position']['x'] = str(building.position.x)
                active_threatens[i]['position']['y'] = str(building.position.y)
                active_threatens[i]['_R'] = str(building._R)
                active_threatens[i]['_H'] = str(building._H)
            self._apply_fixed_obstacle_height(self.buildings, active_threatens)
            self.buildings_param["Threaten"] = active_threatens

    def getEnvJson(self):
        environment = copy.copy(self.buildings_param)
        environment['path'] = []
        environment['epoch'] = self.epoch
        environment['virtual_obstacles'] = []
        if len(self.Agents) > 0:
            for indx, p in enumerate(self.Agents[0].path):
                environment['path'].append(p)
            agent = self.Agents[0]
            obstacles = getattr(agent, 'virtual_obstacle_history', getattr(agent, 'virtual_obstacles', []))
            for obstacle in obstacles:
                environment['virtual_obstacles'].append({
                    'x': obstacle.x,
                    'y': obstacle.y,
                    'z': obstacle.z,
                    'r': getattr(obstacle, 'radius', agent._float_param('virtual_obstacle_radius', 5.0) if hasattr(agent, '_float_param') else 5.0),
                    'source': getattr(obstacle, 'source', 'virtual_obstacle'),
                })
        json_str = json.dumps(environment)
        return json_str

    def render(self, data=None):
        init_opts, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts = self.set_grid()
        path_data = self.Agents[0].path if len(self.Agents) > 0 else []
            
        path = self.draw_path(path_data, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts)
        threaten_3D = self.draw_threaten_3D(None, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts)
        virtual_obstacle_3D = self.draw_virtual_obstacles(x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts)
        markers = self.draw_start_end(x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts)
        
        path.set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
        threaten_3D.set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
        if virtual_obstacle_3D:
            virtual_obstacle_3D.set_global_opts(legend_opts=opts.LegendOpts(is_show=True, pos_top="2%", pos_left="center"))

        if markers:
            markers.set_global_opts(legend_opts=opts.LegendOpts(is_show=True, pos_top="2%", pos_left="center"))

        grid = Grid()
        grid.add(path, grid_opts=opts.GridOpts())
        grid.add(threaten_3D, grid_opts=opts.GridOpts())
        if virtual_obstacle_3D:
            grid.add(virtual_obstacle_3D, grid_opts=opts.GridOpts())
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
            
        save_path = save_dir + data['uuid'] + '.html'
        data['html_dir'] = save_path
        grid.render(save_path)

    def draw_start_end(self, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts):
        if len(self.Agents) == 0: return None
        agent = self.Agents[0]
        start_pos = [[agent.path[0][0], agent.path[0][1], agent.path[0][2]]] if len(agent.path) > 0 else [[agent.position.x, agent.position.y, agent.position.z]]
        goal_pos = [[agent.goal.x, agent.goal.y, agent.goal.z]]

        scatter = Scatter3D(init_opts=init_opts)
        scatter.add(series_name="Start Point", data=start_pos, itemstyle_opts=opts.ItemStyleOpts(color="#00FF00", opacity=1))
        scatter.add(series_name="Goal Point", data=goal_pos, itemstyle_opts=opts.ItemStyleOpts(color="#FF0000", opacity=1))
        return scatter

    def draw_threaten_3D(self, marked_points, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts):
        re = Surface3D()
        for th in self.buildings:
            re.add(series_name="", shading="color", data=list(surface3d_data(th)), xaxis3d_opts=x_axis3d_opts, yaxis3d_opts=y_axis3d_opts, zaxis3d_opts=z_axis3d_opts, grid3d_opts=grid3d_opts, itemstyle_opts=opts.ItemStyleOpts(color='red', opacity=0.7))
        return re

    def draw_virtual_obstacles(self, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts):
        if len(self.Agents) == 0:
            return None
        agent = self.Agents[0]
        obstacles = getattr(agent, 'virtual_obstacle_history', getattr(agent, 'virtual_obstacles', []))
        if len(obstacles) == 0:
            return None
        scatter = Scatter3D(init_opts=init_opts)
        for obstacle in obstacles:
            radius = getattr(obstacle, 'radius', 5.0)
            scatter.add(
                series_name="Virtual Obstacle",
                data=[[obstacle.x, obstacle.y, obstacle.z]],
                xaxis3d_opts=x_axis3d_opts,
                yaxis3d_opts=y_axis3d_opts,
                zaxis3d_opts=z_axis3d_opts,
                grid3d_opts=grid3d_opts,
                itemstyle_opts=opts.ItemStyleOpts(color="#8000FF", opacity=1),
            )
            if scatter.options.get("series"):
                scatter.options["series"][-1]["symbolSize"] = max(6, min(30, radius * 2))
        return scatter

    def set_grid(self):
        ui_param = self.param.get('UI')
        ui_param_path = os.path.normpath(ui_param.get('UI_path'))
        ui_all_dict = XML2Dict(ui_param_path)
        self.ui_dict = ui_all_dict.get('UI')
        InitOpts_dict = self.ui_dict.get('InitOpt')
        init_opts = opts.InitOpts(width=InitOpts_dict.get('width'), height=InitOpts_dict.get('height'))

        self.x_axis3d_opts_dict = self.ui_dict.get('x_axis3d_opts')
        x_axis3d_opts = opts.Axis3DOpts(type_="value", max_=self.x_axis3d_opts_dict.get('MAX'), min_=self.x_axis3d_opts_dict.get('MIN'), split_number=self.x_axis3d_opts_dict.get('split_number'))
        self.y_axis3d_opts_dict = self.ui_dict.get('y_axis3d_opts')
        y_axis3d_opts = opts.Axis3DOpts(type_="value", max_=self.y_axis3d_opts_dict.get('MAX'), min_=self.y_axis3d_opts_dict.get('MIN'), split_number=self.y_axis3d_opts_dict.get('split_number'))
        self.z_axis3d_opts_dict = self.ui_dict.get('z_axis3d_opts')
        z_axis3d_opts = opts.Axis3DOpts(type_="value", max_=self.z_axis3d_opts_dict.get('MAX'), min_=self.z_axis3d_opts_dict.get('MIN'), split_number=self.z_axis3d_opts_dict.get('split_number'))
        self.grid3dopts_dict = self.ui_dict.get('Grid3DOpt')
        grid3dopts = opts.Grid3DOpts(width=self.grid3dopts_dict.get('width'), height=self.grid3dopts_dict.get('height'), depth=self.grid3dopts_dict.get('depth'))
        return init_opts, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3dopts

    def draw_path(self, path_data, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts):
        return Line3D(init_opts=init_opts).add(series_name="", shading="color", data=path_data, xaxis3d_opts=x_axis3d_opts, yaxis3d_opts=y_axis3d_opts, zaxis3d_opts=z_axis3d_opts, grid3d_opts=grid3d_opts)

    def Threaten_rate(self, p: Loc):
        if p.x < 0 or p.x > self.width or p.y < 0 or p.y > self.width or p.z < 0 or p.z > self.h:
            return 1
        for th in self.buildings:
            if th.check_threaten(p) > 0: return 1
        return 0

    def Draw_threaten_map(self, h=0):
        map = np.zeros((self.width, self.len))
        for i in range(self.width):
            for j in range(self.len):
                map[i, j] = self.Threaten_rate(Loc(i, j, h))
        return map

    def Check_uav_Done(self):
        for agent in self.Agents:
            if not agent.done: return False
        return True

    def Choose_Action2(self, index: int, eps=0.2):
        state = self.Agents[index].state()
        return self.Agents[index].Trainer.get_action(state, eps)

    def Reset_Result(self, eps_rate):
        self.result = {'success': 0, 'lose': 0, 'meet_threaten': 0, 'normal': 0, 'loss': 0, 'sum_epoch': 0, 'eps': eps_rate, 'score': 0, 'average_score': 0, 'step': 0}

    def run_thread_OffPolicy(self, uav, ind, eps_rate):
        if uav.done: return
        state_test = uav.state()
        action = self.Choose_Action2(ind, eps_rate)
        next_state, reward, done, info = self.Move_Agent(ind, action)
        self.Run_statistics(info)
        uav.Trainer.Push_Replay((FloatTensor(np.array([state_test])), action, FloatTensor([[reward]]), FloatTensor(np.array([next_state])), FloatTensor([[done]])))
        uav.Trainer.replay_memory.add(state_test, action, reward, next_state, done)
        if len(uav.Trainer.replay_memory.buffer) > uav.Trainer.Batch_Size:
            b_s, b_a, b_r, b_ns, b_d, idx, weights = uav.Trainer.replay_memory.sample2(uav.Trainer.Batch_Size)
            uav.transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d, 'idx': idx, 'weights': weights}

    def run_eposide(self, eps_rate=0.1):
        self.Reset_Result(eps_rate)
        self.Scene_Random_Reset() 
        while True:
            for ind, uav in enumerate(self.Agents):
                uav.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                self.run_thread_OffPolicy(uav, ind, eps_rate)
            train_info = self.update()
            if self.Check_uav_Done(): break
                
        self.Train_statistics(train_info)
        self.epoch += 1
        if self.epoch % self.print_loop == 0:
            for uav in self.Agents: uav.record_list()
        
        data_result = self.generate_train_result()
        self.render(data_result)
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

    def Train_statistics(self, train_info):
        num = len(train_info)
        for index, info in enumerate(train_info):
            if type(info['loss']) != int: self.result['loss'] += info['loss'].item()
            else: self.result['loss'] += info['loss']
            self.result['sum_epoch'] += info['sum_epoch']
            self.result['score'] += self.Cal_Score()[0]
            self.result['average_score'] += self.Cal_Score()[1]
            self.result['step'] += self.Agents[index].Step
        if num != 0:
            self.result['loss'] /= num; self.result['sum_epoch'] /= num
            self.result['score'] /= num; self.result['average_score'] /= num; self.result['step'] /= num

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
