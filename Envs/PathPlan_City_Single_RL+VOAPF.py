import datetime
import json
import time
from tkinter import Grid
from BaseClass.BaseEnv import *
from BaseClass.CalMod import *
import torch
import numpy as np
import math
import random
import csv
import threading
import copy
import heapq
from pyecharts.charts import Surface3D
from torch.autograd import Variable
import pyecharts.options as opts
from pyecharts.charts import Line3D, Scatter3D, Grid
import uuid

from DataBase.Connector import MySQLConnector

# [新增] 產生虛擬障礙物所需結構類別
class DummyVirtualObstacle:
    def __init__(self, pos, r):
        self.position = pos
        self._R = r
        self._H = 30 # 高度稍微高於普通障礙物(通常是20)，以便在圖中明顯突出

class PathPlan_City(BaseEnv):
    def __init__(self, param: dict) -> None:
        super().__init__(param)
        self.eps = float(None2Value(param.get('eps'), 0.1))
        self.Is_On_Policy = int(param.get('Is_On_Policy'))
        self.param = param
        
        self.len = int(param.get('len', 500))
        self.width = int(param.get('width', 500))
        self.h = int(param.get('h', 100))

        self.map = np.zeros((self.width, self.len)) 
        self.buildings = []
        
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

                for param_threaten in threatens:
                    obj_threaten = self.ThreatenFactory.Create_Threaten(param_threaten, self)
                    self.buildings.append(obj_threaten)

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

        self.result = {'success': 0, 'failed:': 0, 'meet_threaten': 0, 'normal': 0, 'loss': None, 'sum_epoch': 0, 'eps': 0.1}
        self.epoch = 0
        self.print_loop = int(param.get('print_loop'))
        self.Is_FL = int(param.get('Is_FL'))
        self.FL_Loop = int(param.get('FL_Loop'))
        self.threads = []
        self.executed_time = 0
        
        self.SQL_connector = None 

        self.Train_info_line = {
            'uuid': '12345', 'date': '2024-04-02', 'html_dir': '', 'train_score': 0.75,
            'epoch': 10, 'Is_Scored': 1, 'train_time': 123.45, 'path_len': 50.0,
            'score_quality': 3, 'score_len': 4, 'score_threaten': 2
        }

    def Scene_Random_Reset(self):
        self.reset_threads = []
        
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

        for indx, uav in enumerate(self.Agents):
            uav.reset()

    def getEnvJson(self):
        environment = copy.copy(self.buildings_param)
        environment['path'] = []
        environment['epoch'] = self.epoch
        if len(self.Agents) > 0:
            for indx, p in enumerate(self.Agents[0].path):
                environment['path'].append(p)
        json_str = json.dumps(environment)
        return json_str

    # [新增] 繪製虛擬障礙物的作用範圍 (紫色圓柱體)
    def draw_virtual_surface(self, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts):
        if len(self.Agents) == 0:
            return None
        records = getattr(self.Agents[0], 'virtual_obstacle_records', [])
        if not records:
            return None
            
        re = Surface3D()
        for rec in records:
            pos = rec['pos']
            # 計算影響半徑：將斥力數值等比例轉化為視覺半徑 (公式可依畫面比例微調)
            # 例如 max_force = 40.0 時，畫出的半徑為 25.0
            visual_radius = rec['max_force'] * 0.5 + 5.0 
            dummy_th = DummyVirtualObstacle(pos, visual_radius)
            
            re.add(
                series_name="Virtual Force Range",
                shading="color",
                data=list(surface3d_data(dummy_th)),
                xaxis3d_opts=x_axis3d_opts,
                yaxis3d_opts=y_axis3d_opts,
                zaxis3d_opts=z_axis3d_opts,
                grid3d_opts=grid3d_opts,
                # 使用紫色半透明來區分紅色真實障礙物
                itemstyle_opts=opts.ItemStyleOpts(color='purple', opacity=0.4), 
            )
        return re

    # [新增] 繪製虛擬障礙物本體中心點 (深紫色實心點)
    def draw_virtual_center(self, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts):
        if len(self.Agents) == 0:
            return None
        records = getattr(self.Agents[0], 'virtual_obstacle_records', [])
        if not records:
            return None
            
        scatter = Scatter3D(init_opts=init_opts)
        data = [[rec['pos'].x, rec['pos'].y, rec['pos'].z] for rec in records]
        scatter.add(
            series_name="Virtual Center",
            data=data,
            itemstyle_opts=opts.ItemStyleOpts(color="#8A2BE2", opacity=1), 
        )
        return scatter

    def render(self, data=None):
        init_opts, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts = self.set_grid()
        
        path_data = []
        if len(self.Agents) > 0:
            path_data = self.Agents[0].path
            
        path = self.draw_path(path_data, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts)
        threaten_3D = self.draw_threaten_3D(None, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts)
        markers = self.draw_start_end(x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts)
        
        # [載入虛擬障礙物圖層]
        v_surface = self.draw_virtual_surface(x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts)
        v_center = self.draw_virtual_center(x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts)

        # ==========================================
        # [修改] 動態排列圖例，解決間距不均與自動擴充問題
        # ==========================================
        # 1. 隱藏不必要的圖例
        path.set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
        threaten_3D.set_global_opts(legend_opts=opts.LegendOpts(is_show=False))

        # 2. 收集「有設定標籤」的圖表
        legend_charts = []
        if markers: legend_charts.append(markers)
        if v_surface: legend_charts.append(v_surface)
        if v_center: legend_charts.append(v_center)
        
        # 3. 利用系統內建對齊方式，每 3 個圖表模組自動換行
        alignments = ["left", "center", "right"] 
        for i, chart in enumerate(legend_charts):
            row = i // 3            # 計算目前在第幾行 (0, 1, 2...)
            col = i % 3             # 計算目前在該行的哪個位置
            top_percent = 2 + (row * 6) # 第一行在頂端 2%，換行就往下挪 6%
            
            chart.set_global_opts(
                legend_opts=opts.LegendOpts(
                    is_show=True,
                    pos_top=f"{top_percent}%",
                    pos_left=alignments[col],   # 讓系統自動貼齊左、中、右
                    orient="horizontal"         # 模組內部的標籤維持水平
                )
            )
        # ==========================================

        grid = Grid()
        grid.add(path, grid_opts=opts.GridOpts())
        grid.add(threaten_3D, grid_opts=opts.GridOpts())
        
        if markers:
            grid.add(markers, grid_opts=opts.GridOpts())
            
        # [將虛擬障礙物加入畫面]
        if v_surface:
            grid.add(v_surface, grid_opts=opts.GridOpts())
        if v_center:
            grid.add(v_center, grid_opts=opts.GridOpts())
        
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
        
        if self.SQL_connector is not None and self.SQL_connector.connection is not None:
            self.SQL_connector.insert_data('Train_info', data)
            self.SQL_connector.insert_data('TrainInfo_Json', data2)

    def draw_start_end(self, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts):
        from pyecharts.charts import Scatter3D
        from pyecharts import options as opts

        if len(self.Agents) == 0:
            return None

        agent = self.Agents[0]
        start_pos = []
        if len(agent.path) > 0:
            start_pos = [[agent.path[0][0], agent.path[0][1], agent.path[0][2]]]
        else:
            start_pos = [[agent.position.x, agent.position.y, agent.position.z]]

        goal_pos = [[agent.goal.x, agent.goal.y, agent.goal.z]]

        scatter = Scatter3D(init_opts=init_opts)
        scatter.add(
            series_name="Start Point",
            data=start_pos,
            itemstyle_opts=opts.ItemStyleOpts(color="#00FF00", opacity=1), 
        )
        scatter.add(
            series_name="Goal Point",
            data=goal_pos,
            itemstyle_opts=opts.ItemStyleOpts(color="#FF0000", opacity=1), 
        )
        return scatter

    def draw_threaten_3D(self, marked_points, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts):
        re = Surface3D()
        for th in self.buildings:
            re.add(
                series_name="",
                shading="color",
                data=list(surface3d_data(th)),
                xaxis3d_opts=x_axis3d_opts,
                yaxis3d_opts=y_axis3d_opts,
                zaxis3d_opts=z_axis3d_opts,
                grid3d_opts=grid3d_opts,
                itemstyle_opts=opts.ItemStyleOpts(color='red', opacity=0.7),
            )
        return re

    def set_grid(self):
        ui_param = self.param.get('UI')
        ui_param_path = os.path.normpath(ui_param.get('UI_path'))
        ui_all_dict = XML2Dict(ui_param_path)
        self.ui_dict = ui_all_dict.get('UI')

        InitOpts_dict = self.ui_dict.get('InitOpt')
        init_opts = opts.InitOpts(width=InitOpts_dict.get('width'), height=InitOpts_dict.get('height'))

        self.x_axis3d_opts_dict = self.ui_dict.get('x_axis3d_opts')
        x_axis3d_opts = opts.Axis3DOpts(
            type_="value",
            max_=self.x_axis3d_opts_dict.get('MAX'),
            min_=self.x_axis3d_opts_dict.get('MIN'),
            split_number=self.x_axis3d_opts_dict.get('split_number')
        )

        self.y_axis3d_opts_dict = self.ui_dict.get('y_axis3d_opts')
        y_axis3d_opts = opts.Axis3DOpts(
            type_="value",
            max_=self.y_axis3d_opts_dict.get('MAX'),
            min_=self.y_axis3d_opts_dict.get('MIN'),
            split_number=self.y_axis3d_opts_dict.get('split_number')
        )

        self.z_axis3d_opts_dict = self.ui_dict.get('z_axis3d_opts')
        z_axis3d_opts = opts.Axis3DOpts(
            type_="value",
            max_=self.z_axis3d_opts_dict.get('MAX'),
            min_=self.z_axis3d_opts_dict.get('MIN'),
            split_number=self.z_axis3d_opts_dict.get('split_number')
        )

        self.grid3dopts_dict = self.ui_dict.get('Grid3DOpt')
        grid3dopts = opts.Grid3DOpts(
            width=self.grid3dopts_dict.get('width'),
            height=self.grid3dopts_dict.get('height'),
            depth=self.grid3dopts_dict.get('depth')
        )
        return init_opts, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3dopts

    def draw_path(self, path_data, x_axis3d_opts, y_axis3d_opts, z_axis3d_opts, grid3d_opts, init_opts):
        path = (
            Line3D(init_opts=init_opts)
            .add(
                series_name="",
                shading="color",
                data=path_data,
                xaxis3d_opts=x_axis3d_opts,
                yaxis3d_opts=y_axis3d_opts,
                zaxis3d_opts=z_axis3d_opts,
                grid3d_opts=grid3d_opts,
            )
        )
        return path

    def Threaten_rate(self, p: Loc):
        if p.x < 0 or p.x > self.width or p.y < 0 or p.y > self.width or p.z < 0 or p.z > self.h:
            return 1
        for th in self.buildings:
            if th.check_threaten(p) > 0:
                return 1
        return 0

    def Draw_threaten_map(self, h=0):
        map = np.zeros((self.width, self.len))
        for i in range(self.width):
            for j in range(self.len):
                map[i, j] = self.Threaten_rate(Loc(i, j, h))
        return map

    def Reset_Thread_UAV(self, uav):
        uav.reset()

    def Set_Space_Scale(self, len: int, width: int, h: int):
        self.len = len
        self.width = width
        self.h = h

    def Draw_static_map(self):
        pass

    def Check_uav_Done(self):
        for agent in self.Agents:
            if not agent.done:
                return False
        return True

    def check_threaten(self, p: Loc):
        pass

    def check_total_threaten(self, x, y, z):
        try:
            ix, iy = int(x), int(y)
            if 0 <= ix < self.width and 0 <= iy < self.len:
                if self.map[ix, iy] > 0:
                    return 1
                else:
                    return 0
            else:
                return 1 
        except Exception as e:
            return 1

    def run(self):
        pass

    def Choose_Action(self, index: int, eps=0.2):
        if self.Is_AC == 1:
            sample = random.random()
            if sample > eps:
                state = self.Agents[index].state()
                tensor_state = FloatTensor(np.array(state))
                probs = self.Agents[index].Trainer.actor(tensor_state)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
                return torch.tensor([[action.item()]], device=device)
            else:
                return torch.tensor([[random.randrange(self.Agents[index].act_num)]], device=device)
        elif self.Is_AC == 2:
            sample = random.random()
            if sample > eps:
                state = self.Agents[index].state()
                tensor_state = FloatTensor(np.array(state))
                action = self.Agents[index].Trainer.actor(tensor_state).item()
                action = action + 0.01 * np.random.randn(1)
            else:
                return torch.tensor([random.uniform(-1, 1)], device=device)
            return action
        elif self.Is_AC == 3:
            state = self.Agents[index].state()
            tensor_state = FloatTensor(np.array(state))
            probs = self.Agents[index].Trainer.actor(tensor_state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return torch.tensor([[action.item()]], device=device)
        elif self.Is_AC == 4:
            state = self.Agents[index].state()
            state = torch.tensor([state], dtype=torch.float).to(device)
            action = self.Agents[index].Trainer.actor(state)[0]
            return action.unsqueeze(0) if action.dim() == 1 else action
        else:
            state = self.Agents[index].state()
            tensor_state = FloatTensor(np.array(state))
            sample = random.random()
            if sample > eps:
                with torch.no_grad():
                    y = self.Agents[index].Trainer.q_local(Variable(tensor_state).type(FloatTensor))
                    value = y.data.max(0)[1].view(1, 1)
                    return value
            else:
                return torch.tensor([[random.randrange(self.Agents[index].act_num)]], device=device)

    def Choose_Action2(self, index: int, eps=0.2):
        state = self.Agents[index].state()
        return self.Agents[index].Trainer.get_action(state, eps)

    def Evaluation_Action(self, index: int):
        pass

    def record(self, result):
        pass

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

    def run_thread_OnPolicy(self, uav, ind, eps_rate):
        if uav.done: return
        state = uav.state()
        action = self.Choose_Action2(ind, eps_rate)
        next_state, reward, done, info = self.Move_Agent(ind, action)
        self.Run_statistics(info)
        uav.transition_dict['states'].append(state)
        uav.transition_dict['actions'].append(action)
        uav.transition_dict['next_states'].append(next_state)
        uav.transition_dict['rewards'].append(reward)
        uav.transition_dict['dones'].append(done)

    def run_eposide(self, eps_rate=0.1):
        self.Reset_Result(eps_rate)
        self.Scene_Random_Reset() 
        self.threads = []
        if self.Is_On_Policy == 1:
            for ind, uav in enumerate(self.Agents):
                uav.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            while True:
                self.run()
                for ind, uav in enumerate(self.Agents):
                    thread = threading.Thread(target=self.run_thread_OnPolicy, args=(uav, ind, eps_rate,))
                    self.threads.append(thread)
                    thread.start()
                for thread in self.threads:
                    thread.join()
                if self.Check_uav_Done():
                    break
            train_info = self.update()
        else:
            while True:
                self.run()
                for ind, uav in enumerate(self.Agents):
                    uav.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                    self.run_thread_OffPolicy(uav, ind, eps_rate)
                train_info = self.update()
                if self.Check_uav_Done():
                    break
        self.Train_statistics(train_info)
        self.epoch += 1
        
        if self.epoch % self.print_loop == 0:
            for uav in self.Agents:
                uav.record_list()
        
        if self.epoch % self.FL_Loop == 0 and self.Is_FL:
            if self.Is_AC:
                self.Federated_Learning_AC()
            else:
                self.Federated_Learning_choice() 
        
        data_result = self.generate_train_result()
        self.render(data_result)
        return self.result

    def generate_train_result(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        train_score = self.Agents[0].total_score
        epoch = self.Agents[0].Train_epoch
        train_time = time.time() - self.Agents[0].Train_start
        path_len = self.Agents[0].path_len
        start2goal = self.Agents[0].start2goal
        len_Astar = self.Agents[0].len_Astar
        ReachGoal = self.Agents[0].reach_goal
        Train_info_line = {
            'uuid': str(uuid.uuid4()),
            'date': current_time,
            'html_dir': '<html lang="en"></html>',
            'train_score': train_score,
            'epoch': epoch,
            'Is_Scored': 0,
            'ReachGoal': ReachGoal,
            'train_time': train_time,
            'path_len': path_len,
            'score_quality': 0,
            'score_len': 0,
            'score_threaten': 0,
            'start2goal': start2goal,
            'len_Astar': len_Astar,
        }
        return Train_info_line

    def Run_statistics(self, info):
        self.result[info] = self.result[info] + 1

    def Train_statistics(self, train_info):
        num = len(train_info)
        for index, info in enumerate(train_info):
            if type(info['loss']) != int:
                self.result['loss'] += info['loss'].item()
            else:
                self.result['loss'] += info['loss']
            self.result['sum_epoch'] += info['sum_epoch']
            self.result['score'] += self.Cal_Score()[0]
            self.result['average_score'] += self.Cal_Score()[1]
            self.result['step'] += self.Agents[index].Step
        if num != 0:
            self.result['loss'] /= num
            self.result['sum_epoch'] /= num
            self.result['score'] /= num
            self.result['average_score'] /= num
            self.result['step'] /= num

    def Cal_Score(self):
        sum_score = 0
        for agent in self.Agents:
            sum_score += agent.score
        return sum_score, sum_score / len(self.Agents)

    def run_XML_scene(self):
        pass

    def Load_Scene_FromXML(self, XML_path="../config/FMEC.xml"):
        p_dict = XML2Dict(XML_path)
        if p_dict is not None:
            env_dict = p_dict.get('env')
            if env_dict is not None:
                agents_params = env_dict.get('Agents')
                if agents_params is not None:
                    agents = agents_params.get('Agent')
                    for agent_param in agents:
                        obj_agent = self.AgentFactory.Create_Agent(agent_param, self)
                        agent_trainer = agent_param.get('Trainer')
                        if obj_agent is not None:
                            agent_trainer['name'] = obj_agent.name
                            obj_agent.Trainer = self.TrainerFactory.Create_Trainer(agent_trainer)
                            self.Agents.append(obj_agent)
                
                threaten_params = env_dict.get('buildings')
                if threaten_params is not None:
                    buildings = threaten_params.get('Threaten')
                    for threaten_param in buildings:
                        obj_threaten = self.ThreatenFactory.Create_Threaten(threaten_param, self)
                        if obj_threaten is not None:
                            self.buildings.append(obj_threaten)
                            
                trainer_params = env_dict.get('Trainer')
                if trainer_params is not None:
                    self.Trainer = self.TrainerFactory.Create_Trainer(trainer_params)

    def train(self):
        re = self.Trainer.learn()
        return re

    def Federated_Learning_AC(self):
        global_model = copy.deepcopy(self.Agents[0].Trainer.actor)
        for k in global_model.state_dict().keys():
            for ind, uav in enumerate(self.Agents):
                if ind == 0: continue
                global_model.state_dict()[k] += uav.Trainer.actor.state_dict()[k]
            global_model.state_dict()[k] = torch.div(global_model.state_dict()[k], len(self.Agents))
        for uav in self.Agents:
            uav.Trainer.replace_param(global_model)

    def Federated_Learning(self):
        for uav in self.Agents:
            start = time.time()
            target_SPN = []
            SPN_KL = []
            if len(uav.Trainer.replay_memory.buffer) > uav.Trainer.Batch_Size:
                b_s, b_a, b_r, b_ns, b_d = uav.Trainer.replay_memory.sample2(uav.Trainer.Batch_Size)
            else:
                b_s, b_a, b_r, b_ns, b_d = uav.Trainer.replay_memory.sample2(len(uav.Trainer.replay_memory.buffer) - 1)
            test_sample = b_s
            SelfPolicy = uav.Trainer.get_policy_DFRL(test_sample)
            for target_uav in self.Agents:
                target_policy = target_uav.Trainer.get_policy_DFRL(test_sample)
                KL_ = [kl_divergence_FDRL(tp, Sp) for tp, Sp in zip(target_policy, SelfPolicy)]
                KL_ = sum(KL_) / len(KL_)
                target_SPN.append(target_uav.Trainer.SPN_param())
                SPN_KL.append(KL_)
            sorted_indices = sorted(enumerate(SPN_KL), key=lambda x: x[1], reverse=False)
            SPN_aggregation = [target_SPN[index] for index, value in sorted_indices[:uav.Top_G]]
            with torch.no_grad():
                aggregation_model = copy.deepcopy(uav.Trainer.SPN_param())
                for k in aggregation_model.state_dict().keys():
                    for model in SPN_aggregation:
                        aggregation_model.state_dict()[k] += model.state_dict()[k]
                    aggregation_model.state_dict()[k].data.copy_(torch.div(aggregation_model.state_dict()[k], len(SPN_aggregation)))
            end = time.time()
            uav.Train_time += (start - end)
            uav.Update_SPN_Soft(aggregation_model)

    def Federated_Learning_choice(self):
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for ind_uav, uav_p in enumerate(self.Agents):
                global_model = copy.deepcopy(uav_p.Trainer.q_local)
                global_model_target = copy.deepcopy(uav_p.Trainer.q_target)
                FL_list = []
                transitions = uav_p.Trainer.replay_memory.sample(10)
                from collections import namedtuple
                Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
                batch = Transition(*zip(*transitions))
                states = torch.cat(batch.state)
                q_value = uav_p.Trainer.q_local(states)
                for ind, uav in enumerate(self.Agents):
                    if ind == ind_uav: continue
                    q_u = uav.Trainer.q_local(states)
                    loss = mse_loss(q_value, q_u)
                    FL_list.append([ind, loss])
                sorted_list = sorted(FL_list, key=lambda x: x[1])
                chosed_list = sorted_list[:len(sorted_list) // 2]
                for k in global_model.state_dict().keys():
                    for it in chosed_list:
                        global_model.state_dict()[k] += self.Agents[it[0]].Trainer.q_local.state_dict()[k]
                        global_model_target.state_dict()[k] += self.Agents[it[0]].Trainer.q_target.state_dict()[k]
                    global_model.state_dict()[k].data.copy_(torch.div(global_model.state_dict()[k], len(chosed_list) + 1))
                    global_model_target.state_dict()[k].data.copy_(torch.div(global_model_target.state_dict()[k], len(chosed_list) + 1))
                uav_p.Trainer.replace_param(global_model)

    def train_off_policy_FL(self):
        re = []
        for uav in self.Agents:
            item = uav.Trainer.learn_off_policy()
            item['score'] = uav.score
            item['average_score'] = uav.score
            item['step'] = uav.Step
            item['energy_cost'] = uav.energy_cost_total
            item['task_collect'] = uav.task_collect
            item['Energy_Efficent'] = uav.task_collect / (uav.energy_cost_total + 0.001)
            item['UE_waiting_time'] = 0
            for ue in uav.UEs:
                item['UE_waiting_time'] += ue.Caculate_wait_time
            item['UE_waiting_time'] = item['UE_waiting_time'] / len(uav.UEs)
            uav.infos.append(item)
            re.append(item)
        if self.Is_FL:
            self.Federated_Learning_choice()
        return re

    def train_off_policy(self):
        re = []
        for uav in self.Agents:
            item = uav.Trainer.learn_off_policy()
            item['score'] = uav.score
            item['average_score'] = uav.score
            item['step'] = uav.Step
            item['energy_cost'] = uav.energy_cost_total
            item['task_collect'] = uav.task_collect
            item['Energy_Efficent'] = uav.task_collect / (uav.energy_cost_total + 0.001)
            item['UE_waiting_time'] = 0
            item['ALL_UEs_F'] = 0
            item['ALL_UEs_D'] = 0
            for ue in uav.UEs:
                item['ALL_UEs_D'] += ue.D_t
                item['ALL_UEs_F'] += ue.F_t
                item['UE_waiting_time'] += ue.Caculate_wait_time
            item['UE_waiting_time'] = item['UE_waiting_time'] / len(uav.UEs)
            re.append(item)
        return re

    def train_on_policy(self):
        re = []
        for uav in self.Agents:
            item = uav.Trainer.learn_on_policy(uav.transition_dict)
            item['score'] = uav.score
            item['average_score'] = uav.score
            item['step'] = uav.Step
            item['energy_cost'] = uav.energy_cost_total
            item['task_collect'] = uav.task_collect
            item['Energy_Efficent'] = uav.task_collect / (uav.energy_cost_total + 0.001)
            item['UE_waiting_time'] = 0
            for ue in uav.UEs:
                item['UE_waiting_time'] += ue.Caculate_wait_time
            item['UE_waiting_time'] = item['UE_waiting_time'] / len(uav.UEs)
            re.append(item)
        return re

    def update(self):
        re = []
        for uav in self.Agents:
            item = uav.Train_nn()
            item['score'] = uav.score
            item['average_score'] = uav.score
            item['step'] = uav.Step
            item['energy_cost'] = uav.energy_cost_total
            item['task_collect'] = uav.task_collect
            item['Energy_Efficent'] = uav.task_collect / (uav.energy_cost_total + 0.001)
            item['UE_waiting_time'] = 0
            for ue in uav.UEs:
                item['UE_waiting_time'] += ue.Caculate_wait_time
            item['UE_waiting_time'] = item['UE_waiting_time'] / (len(uav.UEs) + 1)
            re.append(item)
        return re

    def store_experience(self, replay):
        self.Trainer.Push_Replay(replay)