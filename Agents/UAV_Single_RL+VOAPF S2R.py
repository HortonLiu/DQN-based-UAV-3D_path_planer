import time
from BaseClass.BaseAgent import *
from BaseClass.CalMod import *
from BaseClass.BaseCNN import *
from GN import *
import random
import csv
import copy
import os
import collections
root=os.path.dirname(os.path.abspath(__file__)) #当前根目录
from torch.autograd import Variable
import numpy as np

class UAV(BaseAgent):
    def __init__(self,param:dict,env=None) -> None:
        super().__init__(param)
        self.param=param
        self.name=param.get("name")
        self.j=param.get("j")
        self.V_vector=Loc(0,0,0)
        self.Max_V=float(param.get("Max_V"))
        self.Steering_angle=float(param.get("Steering_angle"))/180*math.pi
        self.V=self.Calc_V()
        
        self.R=int(param.get("R"))+0.5*self.j
        self.R_comm=int(param.get("R_comm"))+0.75*self.j
        self.Scan_Radius = float(param.get("Scan_Radius", 40.0))
        self.ac=int(param.get("Acceration"))+0.25*self.j
        
        # XML budgets remain in the original one-second navigation reference.
        # The public Step counter still increases by two per physical update.
        hardware_flag = int(param.get('Hardware_Hz_Enabled', 1))
        if hardware_flag not in (0, 1):
            raise ValueError('Hardware_Hz_Enabled must be 0 or 1')
        self.hardware_hz_enabled = bool(hardware_flag)
        self.reference_dt = 1.0
        self.control_hz = self.freq_rl = 480 if self.hardware_hz_enabled else 1
        self.control_dt = 1.0 / self.control_hz
        self.step_increment = 2
        self.step_units_per_second = self.step_increment * self.control_hz
        self.controls_per_decision = self.control_hz
        self.set_episode_duration(float(param.get("Max_Step")) * self.reference_dt)
        self.Step = 0
        self.f_max=float(param.get("f_max"))+0.1*self.j

        # ==========================================
        # [新增] 從 XML 讀取卡死偵測與虛擬障礙物參數
        # ==========================================
        self.stuck_step_limit = int(param.get("stuck_step_limit", 30))
        self.stuck_distance_threshold = float(param.get("stuck_distance_threshold", 3.0))
        self.virtual_force_initial = float(param.get("virtual_force_initial", 10.0))
        self.virtual_force_growth = float(param.get("virtual_force_growth", 2.0))
        self.max_virtual_force = float(param.get("max_virtual_force", 40.0))
        self.bonus_steps_per_v_obs = int(param.get("bonus_steps_per_v_obs", 50))
        self.bonus_duration_s = self.bonus_steps_per_v_obs * self.reference_dt
        self.bonus_step_increment = self._seconds_to_step_units(self.bonus_duration_s)
        self.stuck_window_seconds = self.stuck_step_limit * self.reference_dt
        self.virtual_influence_radius = 15.0
        self.max_virtual_obstacles = int(param.get("max_virtual_obstacle", 200))
        # ==========================================

        # [動作空間] 3 維標準 APF 控制
        # 0: Steering_Bias
        # 1: k_rep_obs
        # 2: k_att
        
        self.path=[]
        self.V_record=[]
        self.R_record=[]
        self.Trainer=None
        
        Power_param=param.get("Power_param")
        Fly_power=Power_param.get("Fly_power")
        self.P_i=float(Fly_power.get("P_i"))
        self.v_0=float(Fly_power.get("v_0"))
        self.d_0=float(Fly_power.get("d_0"))
        self.rho=float(Fly_power.get("rho"))
        self.s=float(Fly_power.get("s"))
        self.A=float(Fly_power.get("A"))+0.03*self.j
        self.P_b=float(Fly_power.get("P_b"))
        self.F_b=float(Fly_power.get("F_b"))
        self.xi=0.8+0.02*self.j
        self.P_fly=self.Calc_Fly_Power()
        self.Communication_power=float(Power_param.get("Communication_power"))
        
        self.transition_dict= {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        self.env=env
        self.Min_dis=self.R
        self.CsvWriter=None
        self.infos=[]
        self.Init_Record_Mod()
        self.goal=Loc(75,75,0)
        self.data_num=0

        self.UEs_param=param.get("UEs_param")
        self.UEs=[]
        self.num_UEs=int(self.UEs_param.get('UES_num'))
        for i in range(self.num_UEs):
            param_ue=copy.copy(self.UEs_param)
            param_ue['uav']=self
            param_ue['position']=Loc(random.uniform(0,self.env.len),random.uniform(0,self.env.width),0)
            self.UEs.append(self.env.AgentFactory.Create_Agent(param_ue))

        self.energy_cost_total=0
        self.task_collect=0
        self.task_executed=0
        self.ue_waiting_time=0
        
        self.Train_time=0
        self.Testing_time=0
        self.sum_KL=[]

        self.update_function = self.update_PathPlan
        self.state_function_name=param.get("state_function_name")
        self.state_function= getattr(self,self.state_function_name)
        
        self.goal=Loc(470,470,0)
        
        self.sub_granularity=int(param.get("sub_granularity"))
        self.map_granularity=int(param.get("map_granularity"))
        self.sub_goals=[] 
        self.APF_Enabled=int(param.get("APF_Enabled"))
        self.sub_goals = [self.goal]
        
        self.total_score=0
        self.Train_start=time.time()
        self.Train_pointcut=time.time()
        self.path_len=0
        self.Train_epoch=0
        self.reach_goal=0
        self.start2goal=Eu_Loc_distance(self.position, self.goal)
        self.len_Astar=0 

        self.freq_rtk = 20 if self.hardware_hz_enabled else 1
        self.freq_lidar = 10 if self.hardware_hz_enabled else 1
        self.steps_per_rtk = max(1, self.control_hz // self.freq_rtk)
        self.steps_per_lidar = max(1, self.control_hz // self.freq_lidar)
        self.path_record_period = self.steps_per_rtk
        # Include both endpoints: 201 positions at 20 Hz, or 11 at one update/s.
        self.stuck_history_samples = int(math.ceil(self.stuck_window_seconds * self.freq_rtk)) + 1
        self.stuck_history = collections.deque(maxlen=self.stuck_history_samples)
        self.active_virtual_obstacles = []
        self.virtual_obstacle_records = []
        self._reset_control_clock()

    def _seconds_to_step_units(self, seconds):
        return int(math.ceil(float(seconds) * self.control_hz - 1e-10)) * self.step_increment

    def set_episode_duration(self, seconds):
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Episode duration must be a positive number of simulated seconds")
        self.base_duration_s = seconds
        self.original_max_step = self._seconds_to_step_units(seconds)
        self.Max_Step = self.original_max_step

    def _reset_control_clock(self):
        self.Control_step = 0
        self.sim_time = 0.0
        self.path_times = []
        self.cached_rtk_pos = self.cached_rtk_vel = self.cached_lidar_grid = None
        self._last_rtk_tick = self._last_lidar_tick = -1
        self._last_virtual_update_tick = 0
        self.rtk_updates = self.lidar_updates = 0
        self.collision_count = 0
        self._next_virtual_id = 1
        self.last_episode_result = {}
        self.terminal_info = None

    def check_is_stuck(self):
        if len(self.stuck_history) < self.stuck_history_samples:
            return False
        displacement = Eu_Loc_distance(self.stuck_history[0], self.stuck_history[-1])
        return displacement < self.stuck_distance_threshold

    def _update_virtual_sources(self):
        # Gain growth and source insertion retain their one-second reference cadence.
        if self.Control_step - self._last_virtual_update_tick < self.controls_per_decision:
            return
        self._last_virtual_update_tick = self.Control_step
        if not self.check_is_stuck():
            return
        for source in self.active_virtual_obstacles:
            if Eu_Loc_distance(self.position, source['pos']) < 5.0:
                source['force'] = min(self.max_virtual_force,
                                      source['force'] + self.virtual_force_growth)
                break
        else:
            if len(self.active_virtual_obstacles) >= self.max_virtual_obstacles:
                return
            angle = random.uniform(0, 2 * math.pi)
            self.active_virtual_obstacles.append({
                'id': self._next_virtual_id, 'pos': copy.copy(self.position),
                'force': self.virtual_force_initial, 'created_time_s': self.sim_time,
                # Choose the near-coincident escape direction once, not 480 times/s.
                'escape_direction': (math.cos(angle), math.sin(angle)),
            })
            self._next_virtual_id += 1
            self.Max_Step += self.bonus_step_increment
        self.virtual_obstacle_records = [{
            'id': source['id'], 'pos': copy.copy(source['pos']),
            'max_force': source['force'], 'created_time_s': source['created_time_s'],
        } for source in self.active_virtual_obstacles]

    def cal_force(self, current_pos, k_rep_obs=15.0, k_att=1.0):
        total_force = Loc(0, 0, 0)
        f_obs_sum = Loc(0, 0, 0)
        
        repulsion_range = 30.0 

        # 1. 真實障礙物斥力
        for threaten in self.env.buildings:
            dis = Eu_Loc_distance(current_pos, threaten.position)
            
            if dis < threaten._R + repulsion_range:
                vec_x = current_pos.x - threaten.position.x
                vec_y = current_pos.y - threaten.position.y
                norm = math.sqrt(vec_x**2 + vec_y**2) + 0.001
                unit_x = vec_x / norm
                unit_y = vec_y / norm

                d_safe = max(dis - threaten._R, 0.1)
                rep_strength = k_rep_obs * (1.0/d_safe - 1.0/repulsion_range)**2
                f_obs_sum.x += rep_strength * unit_x
                f_obs_sum.y += rep_strength * unit_y

        # 2. 目標引力
        f_att = Loc(0, 0, 0)
        att_strength = k_att * 2.0 
        att_angle = calculate_angle(current_pos, self.goal)
        f_att.x = att_strength * math.cos(att_angle)
        f_att.y = att_strength * math.sin(att_angle)

        # 3. 牆壁斥力
        f_wall = Loc(0, 0, 0)
        wall_threshold = 30 
        k_wall = k_rep_obs 
        
        if current_pos.x < wall_threshold:
            d = max(current_pos.x, 0.1)
            f_wall.x += k_wall * (1.0/d - 1.0/wall_threshold)**2
        elif current_pos.x > self.env.len - wall_threshold:
            d = max(self.env.len - current_pos.x, 0.1)
            f_wall.x -= k_wall * (1.0/d - 1.0/wall_threshold)**2
            
        if current_pos.y < wall_threshold:
            d = max(current_pos.y, 0.1)
            f_wall.y += k_wall * (1.0/d - 1.0/wall_threshold)**2
        elif current_pos.y > self.env.width - wall_threshold:
            d = max(self.env.width - current_pos.y, 0.1)
            f_wall.y -= k_wall * (1.0/d - 1.0/wall_threshold)**2

        # 4. 虛擬障礙物斥力
        f_virtual = Loc(0, 0, 0)
        for v_obs in self.active_virtual_obstacles:
            vx = current_pos.x - v_obs['pos'].x
            vy = current_pos.y - v_obs['pos'].y
            
            distance = math.hypot(vx, vy)
            if distance < 0.1:
                unit_vx, unit_vy = v_obs.get('escape_direction', (1.0, 0.0))
            else:
                unit_vx = vx / (distance + 0.001)
                unit_vy = vy / (distance + 0.001)
            dist_to_vobs = distance
            v_range = self.virtual_influence_radius
            if dist_to_vobs < v_range:
                current_v_force = v_obs['force'] * (1.0 - dist_to_vobs/v_range)
                f_virtual.x += unit_vx * current_v_force
                f_virtual.y += unit_vy * current_v_force

        total_force.x = f_obs_sum.x + f_att.x + f_wall.x + f_virtual.x
        total_force.y = f_obs_sum.y + f_att.y + f_wall.y + f_virtual.y
        
        max_force = 20.0
        current_force_mag = math.sqrt(total_force.x**2 + total_force.y**2)
        if current_force_mag > max_force:
            scale = max_force / current_force_mag
            total_force.x *= scale
            total_force.y *= scale
            
        return total_force

    def Load_DSN(self):
        pass

    def Calc_Fly_Power(self):
        self.V=self.Calc_V() 
        induced=self.P_i*math.sqrt(math.sqrt(1+(self.V**4)/(4*(self.v_0**4)))-(self.V**2)/(2*(self.v_0**2)))
        parasite=0.5*self.d_0*self.rho*self.s*self.A*(self.V**3)
        blade=self.xi*self.P_b*(1+3*(self.V**2)/(self.F_b**2))
        return induced+parasite+blade 
    
    def Calc_V(self):
        V=Eu_Loc_distance(Loc(0,0,0),self.V_vector)
        if V>self.Max_V:
            self.V_vector.x=self.V_vector.x*(self.Max_V/V)
            self.V_vector.y=self.V_vector.y*(self.Max_V/V)
            V=self.Max_V
        return V
    
    def Set_Env(self,env):
        self.env=env
  
    def Init_Record_Mod(self):
        import datetime
        cur_time=datetime.datetime.now()
        cur_time= cur_time.strftime('%m_%d_%Y(%H_%M_%S)')
        if not os.path.exists('logs'):
            os.makedirs('logs')
        path='logs/%s_%s.csv' %(self.name,cur_time)
        self._record_file = open(path, 'a+', newline="", encoding="utf-8")
        self.CsvWriter = csv.writer(self._record_file)
        self.CsvWriter.writerow(["sum_Episode","Episode"," Score"," Avg.Score","eps-greedy","success","failed","meet_threaten",'loss','ALL_UEs_D','ALL_UEs_F','energy_cost','task_collect','Energy_Efficent','UE_waiting_time','Covered_rate','task_executed','executed_rate','KL','Train_time','Testing_time'])
        self._record_file.flush()

    def record_list(self):
        info = self.last_episode_result
        success = int(info.get('success', self.reach_goal))
        lose = int(info.get('lose', self.terminal_info == 'lose'))
        loss = float(info.get('loss', 0.0))
        self.CsvWriter.writerow([
            self.Trainer.epoch, info.get('episode', 0), self.score, self.score,
            info.get('eps', 0.0), success, lose, self.collision_count, loss,
            sum(ue.D_t for ue in self.UEs), sum(ue.F_t for ue in self.UEs),
            self.energy_cost_total, self.task_collect,
            self.task_collect / (self.energy_cost_total + 0.001),
            self.ue_waiting_time, 0, self.task_executed,
            self.task_executed / (self.task_collect + 1), 0,
            self.Train_time, self.Testing_time])
        self._record_file.flush()
        self.infos = []

    def update(self,action):
        return self.update_function(action)
    
    def state(self):
        return self.state_function()
    
    def Train_nn(self):
        start=time.time()
        re=self.Trainer.update(self.transition_dict)
        end=time.time()
        self.Train_time+=(end-start)
        return re

    def reset(self,option=None):
        self._reset_control_clock()
        if option=="local reset":
            self.Step=0
            self.score=0
            self.V=self.Calc_V()
        else:
            self.Step=0
            self.score=0
            self.done=False
            self.path=[] 
            self.V_record=[]
            self.R_record=[]
            self.V_vector=Loc(5,3,0)
            seta=random.uniform(0,2*math.pi)
            self.V_dir=seta
            self.V_vector.x=self.Max_V*math.cos(seta)
            self.V_vector.y=self.Max_V*math.sin(seta)
            self.V=self.Calc_V()
            self.infos=[]
            self.data_num=0
            
            x=random.uniform(10,210)
            y=random.uniform(1,10)
            self.position=Loc(x,y,0)
            
            x=random.uniform(330,490)
            y=random.uniform(420,490)
            self.goal=Loc(x,y,0)
            
            self.sub_goals = [self.goal]
            
            self.total_score=0
            self.path_len=0
            self.reach_goal=0
            self.start2goal=Eu_Loc_distance(self.position,self.goal)
            self.len_Astar=self.start2goal 
            
            self.Max_Step = self.original_max_step
            self.stuck_history.clear()
            self.active_virtual_obstacles = []
            self.virtual_obstacle_records = []

            # [新增] 重置感測器暫存
            self.cached_rtk_pos = None
            self.cached_rtk_vel = None
            self.cached_lidar_grid = None

        self.path = [[self.position.x, self.position.y, self.position.z]]
        self.path_times = [0.0]
        self.stuck_history.clear()

    def get_action(self,state,eps):
        start=time.time()
        action=self.Trainer.get_action(state,eps)
        end=time.time()
        self.Testing_time+=(end-start)
        return action
        
    def is_in_building(self,position):
        return False
    def Set_Max_Step(self,n:int):
        self.max_step=n
    def Set_State_Map(self,L:int,W:int,H:int):
        self.state_map=np.zeros((1,H,L,W))
    def Set_Actions(self,Action_Dic):
        self.Actions=Action_Dic
    def Observation(self,env):
        pass

    def _record_path(self, force=False):
        if force or self.Control_step % self.path_record_period == 0:
            if not self.path_times or self.path_times[-1] != self.sim_time:
                self.path.append([self.position.x, self.position.y, self.position.z])
                self.path_times.append(self.sim_time)

    def _finish_update(self, reward, info):
        if info in ('success', 'lose'):
            self.done = True
            self.terminal_info = info
            self._record_path(force=True)
            if info == 'success':
                self.reach_goal = 1
                self.target_step = self.Step
                reward += 100.0 + max(0.0, self.base_duration_s - self.sim_time) / self.reference_dt
                with open('path.csv', 'w', newline='') as output:
                    csv.writer(output).writerows(self.path)
            else:
                reward -= 20.0
        self.R_record.append(reward)
        self.score += reward
        self.total_score += reward
        return reward, self.done, info

    def update_PathPlan(self, action):
        """One physical update: Step +2, displacement = speed * selected dt."""
        if self.done:
            return 0.0, True, self.terminal_info
        if hasattr(action, 'cpu'):
            action = action.cpu().detach().numpy()
        action = np.asarray(action, dtype=float).reshape(-1)
        steering_bias = float(action[0]) if len(action) else 0.0
        k_rep_obs = max(5.0, (float(action[1]) + 1.0) * 15.0) if len(action) > 1 else 15.0
        k_att = max(0.5, (float(action[2]) + 1.0) * 1.5) if len(action) > 2 else 1.0
        self._refresh_sensors()
        distance_before = Eu_Loc_distance(self.position, self.goal)
        if distance_before < 15.0:
            return self._finish_update(0.0, 'success')
        old_position = copy.copy(self.position)
        advance = getattr(self.env, 'advance_dynamic_obstacles', None)
        if advance is not None:
            advance(self.control_dt)
        total_force = self.cal_force(self.position, k_rep_obs, k_att)
        direction = self.V_vector if total_force.x == 0 and total_force.y == 0 else total_force
        angle = calculate_angle(Loc(0, 0, 0), direction) + steering_bias * self.Steering_angle
        self.V_vector = Loc(self.Max_V * math.cos(angle), self.Max_V * math.sin(angle), 0)
        self.V = self.Calc_V()
        self.position.x += self.V_vector.x * self.control_dt
        self.position.y += self.V_vector.y * self.control_dt
        self.position.x = max(0, min(self.position.x, self.env.len))
        self.position.y = max(0, min(self.position.y, self.env.width))
        collision = self.env.Threaten_rate(self.position) > 0
        if collision:
            self.position = old_position
            self.collision_count += 1
        self.Step += 2
        self.Control_step += 1
        self.sim_time = self.Control_step * self.control_dt
        self.path_len += Eu_Loc_distance(old_position, self.position)
        self.V_record.append(self.V)
        self.Train_epoch += 1
        self._refresh_sensors()
        self._update_virtual_sources()
        self._record_path()
        distance_after = Eu_Loc_distance(self.position, self.goal)
        reward = distance_before - distance_after
        reward -= (0.1 + 0.05 * abs(steering_bias)) * self.control_dt / self.reference_dt
        if collision:
            reward -= 5.0 * self.control_dt / self.reference_dt
        if distance_after < 15.0:
            return self._finish_update(reward, 'success')
        if self.Step >= self.Max_Step:
            return self._finish_update(reward, 'lose')
        return self._finish_update(reward, 'normal')

    def get_ray_hit_distance(self, A, d_hat):
        T_min = float('inf')

        if d_hat.x < -1e-5:
            t = -A.x / d_hat.x
            if 0 < t < T_min: T_min = t
        elif d_hat.x > 1e-5:
            t = (self.env.len - A.x) / d_hat.x
            if 0 < t < T_min: T_min = t
        
        if d_hat.y < -1e-5:
            t = -A.y / d_hat.y
            if 0 < t < T_min: T_min = t
        elif d_hat.y > 1e-5:
            t = (self.env.width - A.y) / d_hat.y
            if 0 < t < T_min: T_min = t

        for b in self.env.buildings:
            vx = A.x - b.position.x
            vy = A.y - b.position.y
            b_coef = d_hat.x * vx + d_hat.y * vy
            c_coef = vx**2 + vy**2 - b._R**2
            
            if c_coef < 0:
                return 0.0

            discriminant = b_coef**2 - c_coef
            if discriminant >= 0:
                t1 = -b_coef - math.sqrt(discriminant)
                if 0 < t1 < T_min:
                    T_min = t1
                    
        return T_min

    def _sample_lidar_grid(self):
        # Preserve the existing ray-casting observation geometry and encoding.
        grid_dim, stride = 11, 10.0
        offset = (grid_dim // 2) * stride
        grid = np.zeros(121)
        for i in range(grid_dim):
            for j in range(grid_dim):
                dx, dy = i * stride - offset, j * stride - offset
                distance = math.hypot(dx, dy)
                if distance < 1e-3:
                    continue
                hit = self.get_ray_hit_distance(self.position, Loc(dx / distance, dy / distance, 0))
                if hit > distance + stride / 2.0 - 1e-5:
                    grid[i * grid_dim + j] = 0.0
                elif hit >= distance - stride / 2.0:
                    grid[i * grid_dim + j] = 1.0
                else:
                    grid[i * grid_dim + j] = -1.0
        return grid

    def _refresh_sensors(self):
        tick = self.Control_step
        if (self.cached_rtk_pos is None or tick % self.steps_per_rtk == 0) and self._last_rtk_tick != tick:
            self.cached_rtk_pos = copy.copy(self.position)
            self.cached_rtk_vel = copy.copy(self.V_vector)
            self._last_rtk_tick = tick
            self.rtk_updates += 1
            self.stuck_history.append(copy.copy(self.cached_rtk_pos))
        if (self.cached_lidar_grid is None or tick % self.steps_per_lidar == 0) and self._last_lidar_tick != tick:
            self.cached_lidar_grid = self._sample_lidar_grid()
            self._last_lidar_tick = tick
            self.lidar_updates += 1

    def state_PathPlan(self):
        self._refresh_sensors()
        pos, velocity = self.cached_rtk_pos, self.cached_rtk_vel
        state = np.empty(126)
        state[:5] = [(self.goal.x - pos.x) / 500.0,
                     (self.goal.y - pos.y) / 500.0,
                     (self.goal.z - pos.z) / 100.0,
                     velocity.x / self.Max_V, velocity.y / self.Max_V]
        state[5:] = self.cached_lidar_grid
        return state
