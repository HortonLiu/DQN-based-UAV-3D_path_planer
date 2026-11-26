import time
from BaseClass.BaseAgent import *
from BaseClass.CalMod import *
from BaseClass.BaseCNN import *
from GN import *
import random
import csv
import copy
import os
root=os.path.dirname(os.path.abspath(__file__))
from torch.autograd import Variable
import numpy as np

# ==========================================
# LiDAR 模擬器 (240度, 683道射線, 5.6m規格)
# ==========================================
class LidarSimulator:
    def __init__(self, fov_deg=240, num_rays=683, max_range=5.6):
        self.fov = math.radians(fov_deg)
        self.num_rays = num_rays
        self.max_range = max_range
        self.angles = np.linspace(-self.fov/2, self.fov/2, self.num_rays)

    def scan(self, pos, heading, buildings):
        distances = np.full(self.num_rays, self.max_range)
        for i, angle in enumerate(self.angles):
            global_angle = heading + angle
            dx = math.cos(global_angle)
            dy = math.sin(global_angle)
            min_dist = self.max_range

            for b in buildings:
                vx = pos.x - b.position.x
                vy = pos.y - b.position.y
                b_coef = 2.0 * (vx * dx + vy * dy)
                c_coef = (vx**2 + vy**2) - b._R**2
                
                discriminant = b_coef**2 - 4 * c_coef
                if discriminant >= 0:
                    t1 = (-b_coef - math.sqrt(discriminant)) / 2.0
                    if 0 < t1 < min_dist:
                        min_dist = t1
                        
            distances[i] = min_dist
        return distances.tolist(), self.angles.tolist()

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
        self.ac=int(param.get("Acceration"))+0.25*self.j
        
        self.original_max_step = int(param.get("Max_Step"))
        self.Max_Step = self.original_max_step
        self.Step=0
        self.f_max=float(param.get("f_max"))+0.1*self.j

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

        self.update_function_name = param.get("update_function_name")
        self.update_function = getattr(self, self.update_function_name, self.update_PathPlan)
        self.state_function_name=param.get("state_function_name")
        self.state_function= getattr(self,self.state_function_name)
        
        self.goal=Loc(470,470,0)
        self.total_score=0
        self.Train_start=time.time()
        self.Train_pointcut=time.time()
        self.path_len=0
        self.Train_epoch=0
        self.reach_goal=0
        self.start2goal=Eu_Loc_distance(self.position, self.goal)

        # 初始化 LiDAR
        self.lidar = LidarSimulator(fov_deg=240, num_rays=683, max_range=5.6)
        self.virtual_obstacles = []
        self.virtual_obstacle_history = []
        self.goal_distance_history = []

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
            self.V_vector.z=self.V_vector.z*(self.Max_V/V)
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
        for ue in self.UEs:
            ALL_UEs_D+=ue.D_t; ALL_UEs_F+=ue.F_t 
            UE_waiting_time=ue.Caculate_wait_time

        energy_cost=self.energy_cost_total
        task_collect=self.task_collect
        Energy_Efficent=self.task_collect/(self.energy_cost_total+0.001)
        Covered_rate=len([UE for UE in self.UEs if UE.Is_covered==1])/(len(self.UEs)+1)
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

    def reset(self, option=None, *, scene=None):
        if option=="local reset":
            self.Step=0
            self.score=0
            self.virtual_obstacles = []
            self.virtual_obstacle_history = []
            self.goal_distance_history = []
            self.V=self.Calc_V()
        else:
            self.Step=0; self.score=0; self.done=False
            self.V_record=[]; self.R_record=[]
            self.virtual_obstacles = []
            self.virtual_obstacle_history = []
            self.goal_distance_history = []
            self.V_vector=Loc(5,3,0)
            seta = scene.heading_rad if scene is not None else random.uniform(0, 2 * math.pi)
            self.V_dir=seta
            self.V_vector.x=self.Max_V*math.cos(seta)
            self.V_vector.y=self.Max_V*math.sin(seta)
            self.V=self.Calc_V()
            self.infos=[]; self.data_num=0
            
            x, y = scene.start if scene is not None else (random.uniform(10, 210), random.uniform(1, 10))
            
            # 【關鍵修復1】高度拉高到 5.0 防止出生在 Z=0 瞬間判定撞地死亡
            self.position = Loc(x,y,5.0)
            
            # 【關鍵修復2】強制初始化路徑紀錄，給予第一個點，確保 Line3D 不為空
            self.path = [[self.position.x, self.position.y, self.position.z]]
            
            x, y = scene.goal if scene is not None else (random.uniform(330, 490), random.uniform(420, 490))
            self.goal=Loc(x,y,0)
            
            self.total_score=0; self.path_len=0; self.reach_goal=0
            self.start2goal=Eu_Loc_distance(self.position,self.goal)
            self.Max_Step = self.original_max_step 

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

    # =========================================================
    # [純 RL 端到端運動學更新] 
    # =========================================================
    def update_PathPlan(self, action):
        if hasattr(action, 'cpu'): action = action.cpu().detach().numpy()
        if hasattr(action, 'flatten'): action = action.flatten()
        elif isinstance(action, list): action = np.array(action).flatten()
            
        global_r=0
        self.Step+=1
        dis2goal_old = Eu_Loc_distance(self.position, self.goal)
        
        # 1. 直接讀取 Action 變更轉向
        steering_bias = action[0] if len(action) > 0 else 0
        seta_old = calculate_angle(Loc(0,0,0), self.V_vector)
        seta_new = seta_old + steering_bias * self.Steering_angle
        
        self.V_vector.x = self.Max_V * math.cos(seta_new)
        self.V_vector.y = self.Max_V * math.sin(seta_new)
        self.V = self.Calc_V()
        
        # 2. 座標更新
        self.position.x += self.V_vector.x
        self.position.y += self.V_vector.y
        self.position.x = max(0, min(self.position.x, self.env.len))
        self.position.y = max(0, min(self.position.y, self.env.width))

        # 【關鍵修復3】在判斷撞牆死亡「之前」，先將新座標存入 path。
        # 這樣就算死亡，畫面上也會留下一條「起點到撞牆點」的線段！
        self.path.append([self.position.x, self.position.y, self.position.z])
        self.V_record.append(self.V)

        # 3. 碰撞檢查：端到端 RL 撞擊即死亡
        if self.env.Threaten_rate(self.position) == 1:
            self.done = True
            global_r -= 50.0 
            self.score += global_r
            self.total_score += global_r
            return global_r, True, 'meet_threaten'
        
        dis2goal_new = Eu_Loc_distance(self.position, self.goal)

        # 4. 抵達終點檢查
        if dis2goal_new < 15: 
            self.done = True
            global_r += 100.0 
            global_r += (self.original_max_step - self.Step) 
            self.score += global_r
            self.target_step = self.Step
            self.reach_goal = 1
            self.total_score += global_r
            with open('path.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for data in self.path:
                    csvwriter.writerow(data)
            return global_r, True, 'success'

        # 5. 稠密獎勵引導
        progress = dis2goal_old - dis2goal_new
        global_r += 1.0 * progress 
        global_r -= 0.05 * abs(steering_bias)
        global_r -= 0.1 
        
        # 6. 超時檢查
        if self.Step >= self.Max_Step:
            self.done = True
            global_r -= 20.0 
            self.score += global_r
            self.total_score += global_r
            return global_r, True, 'lose'

        self.score += global_r
        self.total_score += global_r
        return global_r, False, 'normal'

    # =========================================================
    # [狀態空間] 126維：定位 + LiDAR柵格化
    # =========================================================
    def state_PathPlan(self):
        grid_dim = 11
        state_map = np.zeros((1, 1, 1, 126))
        
        state_map[0, 0, 0, 0] = (self.goal.x - self.position.x) / 500.0
        state_map[0, 0, 0, 1] = (self.goal.y - self.position.y) / 500.0
        state_map[0, 0, 0, 2] = (self.goal.z - self.position.z) / 100.0
        state_map[0, 0, 0, 3] = self.V_vector.x / self.Max_V
        state_map[0, 0, 0, 4] = self.V_vector.y / self.Max_V
        
        heading = calculate_angle(Loc(0,0,0), self.V_vector)
        dists, angles = self.lidar.scan(self.position, heading, self.env.buildings)
        
        grid = np.full((grid_dim, grid_dim), -1.0)
        resolution = 0.8  
        offset = grid_dim // 2 
        
        for dist, angle in zip(dists, angles):
            global_angle = heading + angle
            dx = math.cos(global_angle)
            dy = math.sin(global_angle)
            
            step_size = resolution / 3.0 
            current_dist = 0.0
            
            while current_dist < dist and current_dist <= 5.6:
                x = current_dist * dx; y = current_dist * dy
                idx_x = int(round(x / resolution)) + offset
                idx_y = int(round(y / resolution)) + offset
                
                if 0 <= idx_x < grid_dim and 0 <= idx_y < grid_dim:
                    if grid[idx_x, idx_y] == -1.0: grid[idx_x, idx_y] = 0.0
                current_dist += step_size
                
            if dist < self.lidar.max_range and dist <= 5.6:
                hx = dist * dx; hy = dist * dy
                idx_x = int(round(hx / resolution)) + offset
                idx_y = int(round(hy / resolution)) + offset
                if 0 <= idx_x < grid_dim and 0 <= idx_y < grid_dim:
                    grid[idx_x, idx_y] = 1.0
                    
        state_map[0, 0, 0, 5:126] = grid.flatten()
        return copy.copy(state_map[0, 0, 0, :])

    def _float_param(self, name, default):
        return float(None2Value(self.param.get(name), default))

    def _int_param(self, name, default):
        return int(float(None2Value(self.param.get(name), default)))

    def _obstacle_center(self, obstacle):
        z = obstacle.position.z + getattr(obstacle, '_H', 0.0) * 0.5
        return Loc(obstacle.position.x, obstacle.position.y, z)

    def _obstacle_surface_distance(self, obstacle, position=None):
        position = self.position if position is None else position
        center = self._obstacle_center(obstacle)
        horizontal = math.sqrt((position.x - center.x) ** 2 + (position.y - center.y) ** 2)
        radial_clearance = horizontal - getattr(obstacle, '_R', 0.0)
        z_min = obstacle.position.z
        z_max = obstacle.position.z + getattr(obstacle, '_H', 0.0)
        if z_min <= position.z <= z_max:
            vertical_clearance = 0.0
        else:
            vertical_clearance = min(abs(position.z - z_min), abs(position.z - z_max))
        if radial_clearance >= 0:
            return math.sqrt(radial_clearance ** 2 + vertical_clearance ** 2)
        return radial_clearance

    def _detected_obstacles(self, position=None):
        position = self.position if position is None else position
        detection_range = self._float_param('obstacle_detection_range', 160.0)
        limit = self._int_param('detected_obstacle_limit', 20)
        detected = []
        if self.env is None or not hasattr(self.env, 'buildings'):
            return detected
        for obstacle in self.env.buildings:
            surface_distance = self._obstacle_surface_distance(obstacle, position)
            if surface_distance <= detection_range:
                detected.append((surface_distance, obstacle))
        detected.sort(key=lambda item: item[0])
        return detected[:limit]

    def _unit_vector(self, vec):
        norm = Eu_Loc_distance(Loc(0, 0, 0), vec)
        if norm < 1e-8:
            return Loc(0.0, 0.0, 0.0), 0.0
        return Loc(vec.x / norm, vec.y / norm, vec.z / norm), norm

    def _angle_delta(self, target, current):
        return (target - current + math.pi) % (2.0 * math.pi) - math.pi

    def _current_yaw_pitch(self):
        horizontal = math.sqrt(self.V_vector.x ** 2 + self.V_vector.y ** 2)
        yaw = math.atan2(self.V_vector.y, self.V_vector.x)
        pitch = math.atan2(self.V_vector.z, horizontal)
        return yaw, pitch

    def _constrained_velocity_from_force(self, force):
        unit, force_norm = self._unit_vector(force)
        if force_norm < 1e-8:
            goal_vec = Loc(self.goal.x - self.position.x, self.goal.y - self.position.y, self.goal.z - self.position.z)
            unit, force_norm = self._unit_vector(goal_vec)
            if force_norm < 1e-8:
                return Loc(0.0, 0.0, 0.0)

        current_yaw, current_pitch = self._current_yaw_pitch()
        target_yaw = math.atan2(unit.y, unit.x)
        target_pitch = math.atan2(unit.z, math.sqrt(unit.x ** 2 + unit.y ** 2))
        yaw_limit = self.Steering_angle
        pitch_delta_limit = self._float_param('pitch_delta_angle', 30.0) / 180.0 * math.pi
        pitch_limit = self._float_param('pitch_angle', 45.0) / 180.0 * math.pi

        yaw = current_yaw + max(-yaw_limit, min(yaw_limit, self._angle_delta(target_yaw, current_yaw)))
        pitch = current_pitch + max(-pitch_delta_limit, min(pitch_delta_limit, target_pitch - current_pitch))
        pitch = max(-pitch_limit, min(pitch_limit, pitch))

        speed = self.Max_V
        return Loc(
            speed * math.cos(pitch) * math.cos(yaw),
            speed * math.cos(pitch) * math.sin(yaw),
            speed * math.sin(pitch),
        )

    def _eta_from_action(self, action):
        eta_min = self._float_param('eta_min', 0.0)
        eta_max = self._float_param('eta_max', 20.0)
        limit = self._int_param('detected_obstacle_limit', 20)
        if hasattr(action, 'cpu'):
            action = action.cpu().detach().numpy()
        action = np.array(action, dtype=float).flatten()
        if action.size < limit:
            action = np.pad(action, (0, limit - action.size), mode='constant')
        action = np.clip(action[:limit], -1.0, 1.0)
        return eta_min + (action + 1.0) * 0.5 * (eta_max - eta_min)

    def _repulsive_force_for_obstacle(self, obstacle, eta, goal_distance):
        repulsion_range = self._float_param('repulsion_range', 160.0)
        n = self._float_param('repulsion_distance_power', 2.0)
        center = self._obstacle_center(obstacle)
        from_obstacle = Loc(
            self.position.x - center.x,
            self.position.y - center.y,
            self.position.z - center.z,
        )
        away_unit, center_distance = self._unit_vector(from_obstacle)
        surface_distance = max(self._obstacle_surface_distance(obstacle), 1e-3)
        if surface_distance > repulsion_range or center_distance < 1e-8:
            return Loc(0.0, 0.0, 0.0)

        goal_vec = Loc(self.goal.x - self.position.x, self.goal.y - self.position.y, self.goal.z - self.position.z)
        goal_unit, _ = self._unit_vector(goal_vec)
        distance_term = max(0.0, 1.0 / surface_distance - 1.0 / repulsion_range)
        rep1 = eta * distance_term * (goal_distance ** n) / max(surface_distance ** 2, 1e-6)
        rep2 = 0.5 * n * eta * (distance_term ** 2) * (goal_distance ** max(n - 1.0, 0.0))
        return Loc(
            rep1 * away_unit.x + rep2 * goal_unit.x,
            rep1 * away_unit.y + rep2 * goal_unit.y,
            rep1 * away_unit.z + rep2 * goal_unit.z,
        )

    def _resultant_force(self, etas, detected):
        goal_vec = Loc(self.goal.x - self.position.x, self.goal.y - self.position.y, self.goal.z - self.position.z)
        goal_unit, goal_distance = self._unit_vector(goal_vec)
        attractive_gain = self._float_param('attractive_gain', 1.0)
        weakening_radius = self._float_param('attractive_weakening_radius', 120.0)
        if goal_distance <= weakening_radius:
            attractive_mag = attractive_gain * goal_distance
        else:
            attractive_mag = attractive_gain * weakening_radius
        force = Loc(attractive_mag * goal_unit.x, attractive_mag * goal_unit.y, attractive_mag * goal_unit.z)

        for idx, (_, obstacle) in enumerate(detected):
            rep = self._repulsive_force_for_obstacle(obstacle, etas[idx], max(goal_distance, 1e-3))
            force.x += rep.x
            force.y += rep.y
            force.z += rep.z

        for virtual_obstacle in self.virtual_obstacles:
            rep = self._virtual_obstacle_force(virtual_obstacle, max(goal_distance, 1e-3))
            force.x += rep.x
            force.y += rep.y
            force.z += rep.z

        f_max = self._float_param('apf_force_clip', self.f_max)
        _, force_norm = self._unit_vector(force)
        if force_norm > f_max:
            force.x *= f_max / force_norm
            force.y *= f_max / force_norm
            force.z *= f_max / force_norm
        return force

    def _virtual_obstacle_force(self, virtual_obstacle, goal_distance):
        eta = self._float_param('virtual_eta', 20.0)
        repulsion_range = self._float_param('repulsion_range', 160.0)
        radius = getattr(virtual_obstacle, 'radius', self._float_param('virtual_obstacle_radius', 5.0))
        vec = Loc(
            self.position.x - virtual_obstacle.x,
            self.position.y - virtual_obstacle.y,
            self.position.z - virtual_obstacle.z,
        )
        unit, center_distance = self._unit_vector(vec)
        surface_distance = max(center_distance - radius, 1e-3)
        if surface_distance > repulsion_range or center_distance < 1e-8:
            return Loc(0.0, 0.0, 0.0)
        mag = eta * max(0.0, 1.0 / surface_distance - 1.0 / repulsion_range) * goal_distance / max(surface_distance ** 2, 1e-6)
        return Loc(mag * unit.x, mag * unit.y, mag * unit.z)

    def _bounded_position(self, position):
        min_h = self._float_param('Min_h', 0.0)
        max_h = self._float_param('Max_h', self.env.h if self.env is not None else 100.0)
        return Loc(
            max(0.0, min(position.x, self.env.len)),
            max(0.0, min(position.y, self.env.width)),
            max(min_h, min(position.z, max_h)),
        )

    def _random_virtual_obstacle_radius(self):
        default_radius = self._float_param('virtual_obstacle_radius', 5.0)
        min_radius = self._float_param('virtual_obstacle_min_radius', default_radius)
        max_radius = self._float_param('virtual_obstacle_max_radius', default_radius)
        if max_radius < min_radius:
            min_radius, max_radius = max_radius, min_radius
        return random.uniform(min_radius, max_radius)

    def _add_virtual_obstacle(self, anchor_position, source='predicted_collision'):
        perturb = self._float_param('virtual_obstacle_perturbation', 0.2)
        radius = self._random_virtual_obstacle_radius()
        if source == 'local_minimum':
            obstacle = Loc(anchor_position.x, anchor_position.y, anchor_position.z)
        else:
            obstacle = Loc(
                anchor_position.x + random.uniform(-perturb, perturb) * radius,
                anchor_position.y + random.uniform(-perturb, perturb) * radius,
                anchor_position.z + random.uniform(-perturb, perturb) * radius,
            )
        obstacle.radius = radius
        obstacle.source = source
        obstacle.lifetime = self._int_param('virtual_obstacle_lifetime', 20)
        self.virtual_obstacles.append(obstacle)
        self.virtual_obstacle_history.append(obstacle)
        limit = self._int_param('max_virtual_obstacle', 20)
        if len(self.virtual_obstacles) > limit:
            self.virtual_obstacles = self.virtual_obstacles[-limit:]
        if len(self.virtual_obstacle_history) > limit:
            self.virtual_obstacle_history = self.virtual_obstacle_history[-limit:]
        return obstacle

    def _decay_virtual_obstacles(self):
        active = []
        for obstacle in self.virtual_obstacles:
            obstacle.lifetime = getattr(obstacle, 'lifetime', 1) - 1
            if obstacle.lifetime > 0:
                active.append(obstacle)
        self.virtual_obstacles = active

    def _is_local_minimum(self, goal_distance):
        step_limit = self._int_param('stuck_step_limit', 10)
        distance_threshold = self._float_param('stuck_distance_threshold', 3.0)
        if step_limit <= 1:
            return False
        self.goal_distance_history.append(goal_distance)
        if len(self.goal_distance_history) < step_limit:
            return False
        recent = self.goal_distance_history[-step_limit:]
        progress = recent[0] - recent[-1]
        if progress < distance_threshold:
            self.goal_distance_history = []
            return True
        self.goal_distance_history = self.goal_distance_history[-step_limit:]
        return False

    def _min_obstacle_distance(self):
        if self.env is None or not hasattr(self.env, 'buildings') or len(self.env.buildings) == 0:
            return self._float_param('repulsion_range', 160.0)
        return min(self._obstacle_surface_distance(obstacle) for obstacle in self.env.buildings)

    def update_PathPlan_PGITD3(self, action):
        self.Step += 1
        old_position = Loc(self.position.x, self.position.y, self.position.z)
        old_goal_distance = Eu_Loc_distance(self.position, self.goal)
        detected = self._detected_obstacles()
        etas = self._eta_from_action(action)
        force = self._resultant_force(etas, detected)
        velocity = self._constrained_velocity_from_force(force)
        candidate = self._bounded_position(Loc(
            self.position.x + velocity.x,
            self.position.y + velocity.y,
            self.position.z + velocity.z,
        ))

        if self.env.Threaten_rate(candidate) == 1:
            virtual_obstacle = self._add_virtual_obstacle(candidate, source='predicted_collision')
            f_dis = self._virtual_obstacle_force(virtual_obstacle, max(old_goal_distance, 1e-3))
            force.x += f_dis.x
            force.y += f_dis.y
            force.z += f_dis.z
            disturbance = self._float_param('virtual_obstacle_perturbation', 0.2)
            force.x += random.uniform(-disturbance, disturbance) * self._float_param('apf_force_clip', self.f_max)
            force.y += random.uniform(-disturbance, disturbance) * self._float_param('apf_force_clip', self.f_max)
            force.z += random.uniform(-disturbance, disturbance) * self._float_param('apf_force_clip', self.f_max)
            velocity = self._constrained_velocity_from_force(force)
            candidate = self._bounded_position(Loc(
                self.position.x + velocity.x,
                self.position.y + velocity.y,
                self.position.z + velocity.z,
            ))

        self.V_vector = velocity
        self.V = self.Calc_V()
        self.position = candidate
        self.path.append([self.position.x, self.position.y, self.position.z])
        self.V_record.append(self.V)
        self.path_len += Eu_Loc_distance(old_position, self.position)

        step_distance = Eu_Loc_distance(old_position, self.position)
        new_goal_distance = Eu_Loc_distance(self.position, self.goal)
        if self._is_local_minimum(new_goal_distance):
            self._add_virtual_obstacle(self.position, source='local_minimum')
        self._decay_virtual_obstacles()
        min_obstacle_distance = self._min_obstacle_distance()
        safe_distance = self._float_param('safe_distance', 5.0)
        reward = -0.05 * step_distance
        if new_goal_distance < old_goal_distance:
            reward += 2.0 * (old_goal_distance - new_goal_distance)

        goal_vec = Loc(self.goal.x - old_position.x, self.goal.y - old_position.y, self.goal.z - old_position.z)
        goal_unit, _ = self._unit_vector(goal_vec)
        velocity_unit, _ = self._unit_vector(self.V_vector)
        reward += 0.5 * (
            goal_unit.x * velocity_unit.x + goal_unit.y * velocity_unit.y + goal_unit.z * velocity_unit.z
        )

        if min_obstacle_distance <= 0:
            reward += max(-3.0, -new_goal_distance / max(self.start2goal, 1e-6))
        else:
            obstacle_term = (min_obstacle_distance - safe_distance) / max(safe_distance, 1e-6)
            obstacle_term = max(-3.0, min(3.0, obstacle_term))
            reward += 0.5 * math.exp(obstacle_term) if min_obstacle_distance > safe_distance else -0.5 * math.exp(obstacle_term)

        if self.env.Threaten_rate(self.position) == 1:
            self.done = True
            reward -= 50.0
            info = 'meet_threaten'
        elif new_goal_distance < self._float_param('goal_threshold', 15.0):
            self.done = True
            reward += 100.0 + max(0, self.original_max_step - self.Step)
            self.target_step = self.Step
            self.reach_goal = 1
            info = 'success'
            with open('path.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for data in self.path:
                    csvwriter.writerow(data)
        elif self.Step >= self.Max_Step:
            self.done = True
            reward -= 20.0
            info = 'lose'
        else:
            self.done = False
            info = 'normal'

        self.score += reward
        self.total_score += reward
        return reward, self.done, info

    def state_PathPlan_PGITD3(self):
        limit = self._int_param('detected_obstacle_limit', 20)
        state_scale = max(self._float_param('state_scale', 500.0), 1e-6)
        state = np.zeros(limit * 3 + 3, dtype=float)
        detected = self._detected_obstacles()
        for idx, (_, obstacle) in enumerate(detected):
            center = self._obstacle_center(obstacle)
            base = idx * 3
            state[base] = (center.x - self.position.x) / state_scale
            state[base + 1] = (center.y - self.position.y) / state_scale
            state[base + 2] = (center.z - self.position.z) / state_scale
        state[-3] = (self.goal.x - self.position.x) / state_scale
        state[-2] = (self.goal.y - self.position.y) / state_scale
        state[-1] = (self.goal.z - self.position.z) / state_scale
        return copy.copy(state)
