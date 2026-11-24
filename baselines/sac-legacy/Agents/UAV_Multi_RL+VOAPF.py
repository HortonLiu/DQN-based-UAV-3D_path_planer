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
import math
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
        self.Max_V=int(param.get("Max_V"))
        self.Steering_angle=float(param.get("Steering_angle"))/180*math.pi
        self.V=self.Calc_V()
        
        self.R=int(param.get("R"))+0.5*self.j
        self.R_comm=int(param.get("R_comm"))+0.75*self.j
        self.Scan_Radius = float(param.get("Scan_Radius", 40.0))
        self.ac=int(param.get("Acceration"))+0.25*self.j
        
        self.original_max_step = int(param.get("Max_Step"))
        self.Max_Step = self.original_max_step
        self.Step=0
        self.f_max=float(param.get("f_max"))+0.1*self.j

        self.stuck_step_limit = int(param.get("stuck_step_limit", 30))
        self.stuck_distance_threshold = float(param.get("stuck_distance_threshold", 3.0))
        self.virtual_force_initial = float(param.get("virtual_force_initial", 10.0))
        self.virtual_force_growth = float(param.get("virtual_force_growth", 2.0))
        self.max_virtual_force = float(param.get("max_virtual_force", 40.0))
        self.bonus_steps_per_v_obs = int(param.get("bonus_steps_per_v_obs", 50))

        self.is_leader = (self.j == 0)
        formation_offsets = [
            Loc(0, 0, 0),         # Leader 在中間
            Loc(0, 40, 0),        # Follower 1 在左邊 40 單位
            Loc(0, -40, 0)        # Follower 2 在右邊 40 單位
        ]
        if self.j < len(formation_offsets):
            self.formation_offset = formation_offsets[self.j]
        else:
            self.formation_offset = Loc(0, 0, 0)
            
        self.formation_tolerance = 15.0 
        self.shared_state_map = None    

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

        self.stuck_history = collections.deque(maxlen=self.stuck_step_limit)
        self.active_virtual_obstacles = []
        self.virtual_obstacle_records = [] 

    def get_dynamic_goal(self):
        if self.is_leader:
            return self.goal
        else:
            leader = self.env.Agents[0] 
            target_x = leader.position.x + self.formation_offset.x
            target_y = leader.position.y + self.formation_offset.y
            target_z = leader.position.z + self.formation_offset.z
            return Loc(target_x, target_y, target_z)

    def check_is_stuck(self):
        if len(self.stuck_history) < self.stuck_step_limit:
            return False
        displacement = Eu_Loc_distance(self.stuck_history[0], self.stuck_history[-1])
        if displacement < self.stuck_distance_threshold:
            return True
        return False

    def cal_force(self, current_pos, k_rep_obs=15.0, k_att=1.0):
        total_force = Loc(0, 0, 0)
        f_obs_sum = Loc(0, 0, 0)
        repulsion_range = 30.0 

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

        f_virtual = Loc(0, 0, 0)
        for v_obs in self.active_virtual_obstacles:
            vx = current_pos.x - v_obs['pos'].x
            vy = current_pos.y - v_obs['pos'].y
            if abs(vx) < 0.1 and abs(vy) < 0.1:
                vx = random.uniform(-1, 1)
                vy = random.uniform(-1, 1)
            norm_v = math.sqrt(vx**2 + vy**2) + 0.001
            dist_to_vobs = norm_v
            v_range = 15.0
            if dist_to_vobs < v_range:
                current_v_force = v_obs['force'] * (1.0 - dist_to_vobs/v_range)
                f_virtual.x += (vx / norm_v) * current_v_force
                f_virtual.y += (vy / norm_v) * current_v_force

        f_teammate = Loc(0, 0, 0)
        k_rep_team = k_rep_obs * 1.5 
        team_repulsion_range = 25.0
        for other_uav in self.env.Agents:
            if other_uav.j != self.j: 
                dis = Eu_Loc_distance(current_pos, other_uav.position)
                if dis < team_repulsion_range and dis > 0.1:
                    vec_x = current_pos.x - other_uav.position.x
                    vec_y = current_pos.y - other_uav.position.y
                    norm = math.sqrt(vec_x**2 + vec_y**2) + 0.001
                    rep_strength = k_rep_team * (1.0/dis - 1.0/team_repulsion_range)**2
                    f_teammate.x += rep_strength * (vec_x / norm)
                    f_teammate.y += rep_strength * (vec_y / norm)

        f_att = Loc(0, 0, 0)
        att_strength = k_att * 2.0 
        dynamic_target = self.get_dynamic_goal() 
        att_angle = calculate_angle(current_pos, dynamic_target)
        f_att.x = att_strength * math.cos(att_angle)
        f_att.y = att_strength * math.sin(att_angle)

        total_force.x = f_obs_sum.x + f_att.x + f_wall.x + f_virtual.x + f_teammate.x
        total_force.y = f_obs_sum.y + f_att.y + f_wall.y + f_virtual.y + f_teammate.y
        
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
        file=open(path,'a+',newline="")
        self.CsvWriter=csv.writer(file)
        self.CsvWriter.writerow(["sum_Episode","Episode"," Score"," Avg.Score","eps-greedy","success","failed","meet_threaten",'loss','ALL_UEs_D','ALL_UEs_F','energy_cost','task_collect','Energy_Efficent','UE_waiting_time','Covered_rate','task_executed','executed_rate','KL','Train_time','Testing_time'])

    def record_list(self):
        ALL_UEs_D=ALL_UEs_F=executed_rate=task_executed=Covered_rate=sum_epoch=score=average_score=eps=success=lose=meet_threaten=loss=step=energy_cost=task_collect=UE_waiting_time=Energy_Efficent=0
        sum_epoch=self.Trainer.epoch
        score=self.score
        average_score=self.score
        loss+=0
        for ue in self.UEs:
            ALL_UEs_D+=ue.D_t 
            ALL_UEs_F+=ue.F_t 
            UE_waiting_time=ue.Caculate_wait_time

        energy_cost=self.energy_cost_total
        task_collect=self.task_collect
        Energy_Efficent=self.task_collect/(self.energy_cost_total+0.001)
        Covered_rate=len([UE for UE in self.UEs  if UE.Is_covered==1])/(len(self.UEs)+1)
        task_executed=self.task_executed
        executed_rate=(task_executed/(task_collect+1))
        avg_KL = [0] 
        self.CsvWriter.writerow([sum_epoch,self.Trainer.epoch,score,average_score,eps,success,lose,meet_threaten,loss,ALL_UEs_D,ALL_UEs_F,energy_cost,task_collect,Energy_Efficent,UE_waiting_time,Covered_rate,task_executed,executed_rate,avg_KL,self.Train_time,self.Testing_time])
        self.infos=[]

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
            self.shared_state_map = None

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

    def update_PathPlan(self, action):
        if hasattr(action, 'cpu'): action = action.cpu().detach().numpy()
        if hasattr(action, 'flatten'): action = action.flatten()
        elif isinstance(action, list): action = np.array(action).flatten()
            
        steering_bias = action[0] if len(action) > 0 else 0
        k_rep_obs = max(5.0, (float(action[1]) + 1.0) * 15.0) if len(action) > 1 else 15.0
        k_att = max(0.5, (float(action[2]) + 1.0) * 1.5) if len(action) > 2 else 1.0

        global_r = 0
        dynamic_target = self.get_dynamic_goal()
        dist_to_target = Eu_Loc_distance(self.position, dynamic_target)

        # ==========================================
        # [修改] 團隊到達終點的判定邏輯
        # ==========================================
        leader = self.env.Agents[0]
        dist_to_final_goal = Eu_Loc_distance(leader.position, leader.goal)

        if self.is_leader:
            if dist_to_final_goal < 15: 
                self.done = True
                global_r += 100 
                global_r += max(0, self.original_max_step - self.Step)
                self.score += global_r
                self.target_step = self.Step
                self.reach_goal = 1
                self.total_score += global_r
                with open('path.csv', 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for data in self.path:
                        csvwriter.writerow(data)
                return global_r, True, 'success'
        else:
            # Follower 的結束條件：Leader 到達終點，且自身也到達動態陣型位置
            if dist_to_final_goal < 15:
                if dist_to_target < self.formation_tolerance + 5.0:
                    self.done = True
                    global_r += 50 
                    self.score += global_r
                    self.total_score += global_r
                    self.reach_goal = 1
                    return global_r, True, 'success'
                # 若 Leader 到達終點，但 Follower 還沒回到陣型位置，Follower 會繼續移動！

        self.Step += 1
        old_position = copy.copy(self.position)
        dis2target_old = dist_to_target
        
        self.stuck_history.append(copy.copy(self.position))
        
        if self.check_is_stuck():
            found_nearby = False
            for v_obs in self.active_virtual_obstacles:
                if Eu_Loc_distance(self.position, v_obs['pos']) < 5.0: 
                    v_obs['force'] = min(self.max_virtual_force, v_obs['force'] + self.virtual_force_growth)
                    found_nearby = True
                    break
            if not found_nearby:
                new_v_obs = {
                    'pos': copy.copy(self.position),
                    'force': self.virtual_force_initial 
                }
                self.active_virtual_obstacles.append(new_v_obs)
                self.Max_Step += self.bonus_steps_per_v_obs
                
        self.virtual_obstacle_records = [
            {'pos': v['pos'], 'max_force': v['force']} for v in self.active_virtual_obstacles
        ]

        total_force = self.cal_force(self.position, k_rep_obs, k_att)
        
        if total_force.x == 0 and total_force.y == 0:
             apf_angle = calculate_angle(Loc(0,0,0), self.V_vector)
        else:
             apf_angle = calculate_angle(Loc(0,0,0), total_force)

        final_angle = apf_angle + steering_bias * self.Steering_angle
        
        self.V_vector.x = self.Max_V * math.cos(final_angle)
        self.V_vector.y = self.Max_V * math.sin(final_angle)
        self.V = self.Calc_V()
        
        self.position.x += self.V_vector.x
        self.position.y += self.V_vector.y
        self.position.x = max(0, min(self.position.x, self.env.len))
        self.position.y = max(0, min(self.position.y, self.env.width))

        if self.env.Threaten_rate(self.position) == 1:
            global_r -= 5.0 
            self.position = old_position 
        
        dis2target_new = Eu_Loc_distance(self.position, dynamic_target)
        self.path.append([self.position.x, self.position.y, self.position.z])
        self.V_record.append(self.V)

        progress = dis2target_old - dis2target_new
        global_r += 1.0 * progress 

        if not self.is_leader:
            if dis2target_new > self.formation_tolerance:
                global_r -= 2.0 

        global_r -= 0.05 * abs(steering_bias)
        global_r -= 0.1 
        
        self.R_record.append(global_r)
        self.path_len += self.V
        self.Train_epoch += 1

        if self.Step >= self.Max_Step:
            self.done = True
            global_r -= 20 
            self.score += global_r
            self.total_score += global_r
            return global_r, True, 'lose'
        else:
            self.score += global_r
            self.total_score += global_r
            return global_r, False, 'normal'

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

    def state_PathPlan(self):
        if self.is_leader:
            self.shared_state_map = self._compute_ray_grid(self.position)
            state_map = copy.deepcopy(self.shared_state_map)
        else:
            leader = self.env.Agents[0]
            if hasattr(leader, 'shared_state_map') and leader.shared_state_map is not None:
                state_map = copy.deepcopy(leader.shared_state_map)
            else:
                state_map = self._compute_ray_grid(self.position)

        dynamic_target = self.get_dynamic_goal()
        state_map[0, 0, 0, 0] = (dynamic_target.x - self.position.x) / 500.0
        state_map[0, 0, 0, 1] = (dynamic_target.y - self.position.y) / 500.0
        state_map[0, 0, 0, 2] = (dynamic_target.z - self.position.z) / 100.0
        state_map[0, 0, 0, 3] = self.V_vector.x / self.Max_V
        state_map[0, 0, 0, 4] = self.V_vector.y / self.Max_V
        
        return state_map[0, 0, 0, :]

    def _compute_ray_grid(self, center_pos):
        grid_dim = 11
        state_map = np.zeros((1, 1, 1, 126))
        stride = 10.0
        offset = (grid_dim // 2) * stride
        
        idx = 5
        for i in range(grid_dim):
            for j in range(grid_dim):
                dx = i * stride - offset
                dy = j * stride - offset
                
                L = math.sqrt(dx**2 + dy**2)
                
                if L < 1e-3:
                    state_map[0, 0, 0, idx] = 0.0
                else:
                    d_hat = Loc(dx / L, dy / L, 0)
                    T_min = self.get_ray_hit_distance(center_pos, d_hat)
                    
                    if T_min > L + stride / 2.0 - 1e-5:
                        state_map[0, 0, 0, idx] = 0.0 
                    elif T_min >= L - stride / 2.0:
                        state_map[0, 0, 0, idx] = 1.0 
                    else:
                        state_map[0, 0, 0, idx] = -1.0 
                        
                idx += 1
                
        return state_map