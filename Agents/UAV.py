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

        # ==========================================
        # 硬體感測器頻率設定 (Sim-to-Real)
        # ==========================================
        self.freq_rl = 480       
        self.freq_rtk = 20       
        self.freq_lidar = 10     
        
        self.steps_per_rtk = max(1, self.freq_rl // self.freq_rtk)      
        self.steps_per_lidar = max(1, self.freq_rl // self.freq_lidar)  
        
        # 感測器暫存區 (Zero-Order Hold)
        self.cached_rtk_pos = None
        self.cached_rtk_vel = None
        self.cached_lidar_grid = np.zeros(121)
        
        # [新增] LiDAR 感測器與原始數據暫存
        self.lidar = LidarSimulator(fov_deg=240, num_rays=683, max_range=5.6)
        self.cached_lidar_dists = None
        self.cached_lidar_angles = None
        # ==========================================

        # ==========================================
        # [新增] 從 XML 讀取卡死偵測與虛擬障礙物參數
        # ==========================================
        self.stuck_step_limit = int(param.get("stuck_step_limit", 30))
        self.stuck_distance_threshold = float(param.get("stuck_distance_threshold", 3.0))
        self.virtual_force_initial = float(param.get("virtual_force_initial", 10.0))
        self.virtual_force_growth = float(param.get("virtual_force_growth", 2.0))
        self.max_virtual_force = float(param.get("max_virtual_force", 40.0))
        self.bonus_steps_per_v_obs = int(param.get("bonus_steps_per_v_obs", 50))
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

        # 卡死偵測與虛擬障礙物變數 (套用 XML 參數)
        self.stuck_history = collections.deque(maxlen=self.stuck_step_limit)
        self.active_virtual_obstacles = []
        self.virtual_obstacle_records = []

        # 卡死偵測與虛擬障礙物變數 (套用 XML 參數)
        self.stuck_history = collections.deque(maxlen=self.stuck_step_limit)
        self.active_virtual_obstacles = []
        self.virtual_obstacle_records = [] 

        # ==========================================
        # [新增] 硬體感測器頻率設定 (Sim-to-Real)
        # ==========================================
        self.freq_rl = 480       # 控制迴圈頻率
        self.freq_rtk = 20       # 定位更新頻率
        self.freq_lidar = 10     # 光達掃描頻率
        
        # 計算每幾步(Step)才能更新一次感測器
        self.steps_per_rtk = max(1, self.freq_rl // self.freq_rtk)      # 24 steps
        self.steps_per_lidar = max(1, self.freq_rl // self.freq_lidar)  # 48 steps
        
        # 感測器暫存區 (Zero-Order Hold)
        self.cached_rtk_pos = None
        self.cached_rtk_vel = None
        self.cached_lidar_grid = None
        # ==========================================

    def check_is_stuck(self):
        if len(self.stuck_history) < self.stuck_step_limit:
            return False
        displacement = Eu_Loc_distance(self.stuck_history[0], self.stuck_history[-1])
        # 使用 XML 設定的距離閾值
        if displacement < self.stuck_distance_threshold:
            return True
        return False

    def cal_force(self, current_pos, k_rep_obs=15.0, k_att=1.0):
        total_force = Loc(0, 0, 0)
        f_obs_sum = Loc(0, 0, 0)
        
        # 1. 來自 LiDAR 的真實障礙物斥力 (Virtual Force Field)
        # 僅當 LiDAR 有數據時才計算
        if self.cached_lidar_dists is not None:
            heading = calculate_angle(Loc(0,0,0), self.V_vector)
            d_safe = 3.0 # LiDAR 產生斥力的安全距離閾值
            
            for dist, angle in zip(self.cached_lidar_dists, self.cached_lidar_angles):
                if dist < d_safe:
                    global_angle = heading + angle
                    ray_dx = math.cos(global_angle)
                    ray_dy = math.sin(global_angle)
                    
                    # 經典 APF 斥力公式
                    force_mag = k_rep_obs * (1.0 / dist - 1.0 / d_safe) * (1.0 / (dist**2))
                    
                    # 由於有 683 道射線，單一障礙物會被數十道射線擊中，需除以衰減常數避免受力爆炸
                    force_mag = force_mag / 20.0 
                    
                    f_obs_sum.x += force_mag * (-ray_dx)
                    f_obs_sum.y += force_mag * (-ray_dy)

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
            
            if abs(vx) < 0.1 and abs(vy) < 0.1:
                vx = random.uniform(-1, 1)
                vy = random.uniform(-1, 1)
            
            norm_v = math.sqrt(vx**2 + vy**2) + 0.001
            unit_vx = vx / norm_v
            unit_vy = vy / norm_v
            
            dist_to_vobs = norm_v
            v_range = 15.0
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
        file=open(path,'a+',newline="")
        self.CsvWriter=csv.writer(file)
        self.CsvWriter.writerow(["sum_Episode","Episode"," Score"," Avg.Score","eps-greedy","success","failed","meet_threaten",'loss','ALL_UEs_D','ALL_UEs_F','energy_cost','task_collect','Energy_Efficent','UE_waiting_time','Covered_rate','task_executed','executed_rate','KL','Train_time','Testing_time'])

    def record_list(self):
        ALL_UEs_D=ALL_UEs_F=executed_rate=task_executed=Covered_rate=sum_epoch=score=average_score=eps=success=lose=meet_threaten=loss=step=energy_cost=task_collect=UE_waiting_time=Energy_Efficent=0
        list_len=len(self.infos)
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

            # [新增] 重置感測器暫存
            self.cached_rtk_pos = None
            self.cached_rtk_vel = None
            self.cached_lidar_grid = None

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

        global_r=0
        
        dist_to_goal = Eu_Loc_distance(self.position, self.goal)
        if dist_to_goal < 15: 
            self.done=True
            global_r += 100 
            # 成功抵達的剩餘步數獎勵，使用 original_max_step 計算
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

        self.Step+=1
        old_position=copy.copy(self.position)
        dis2goal_old = dist_to_goal
        self.Step+=1
        old_position=copy.copy(self.position)
        dis2goal_old = dist_to_goal
        
        # ==========================================
        # [修改] 綁定 RTK 頻率 (20Hz) 來判斷卡死
        # 避免 480Hz 下時間間隔太短導致的微小位移被誤判為卡死
        # ==========================================
        if self.Step % self.steps_per_rtk == 0:
            self.stuck_history.append(copy.copy(self.position))
            
            # 虛擬障礙物生成與成長邏輯
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
        # ==========================================        
        # 虛擬障礙物生成與成長邏輯
        if self.check_is_stuck():
            found_nearby = False
            for v_obs in self.active_virtual_obstacles:
                if Eu_Loc_distance(self.position, v_obs['pos']) < 5.0: 
                    # 套用 XML 參數：增長速度與最大值
                    v_obs['force'] = min(self.max_virtual_force, v_obs['force'] + self.virtual_force_growth)
                    found_nearby = True
                    break
            
            if not found_nearby:
                new_v_obs = {
                    'pos': copy.copy(self.position),
                    'force': self.virtual_force_initial # 套用 XML 參數：初始斥力
                }
                self.active_virtual_obstacles.append(new_v_obs)
                
                # 給予額外步數
                self.Max_Step += self.bonus_steps_per_v_obs
                
        # 同步繪圖紀錄
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
        
        dis2goal_new = Eu_Loc_distance(self.position, self.goal)
        self.path.append([self.position.x, self.position.y, self.position.z])
        self.V_record.append(self.V)

        progress = dis2goal_old - dis2goal_new
        global_r += 1.0 * progress 

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
        grid_dim = 11
        state_map = np.zeros((1, 1, 1, 126))
        
        # ==========================================
        # 1. RTK (20Hz) 更新邏輯
        # ==========================================
        if self.Step % self.steps_per_rtk == 0 or self.cached_rtk_pos is None:
            self.cached_rtk_pos = copy.copy(self.position)
            self.cached_rtk_vel = copy.copy(self.V_vector)
            
        rtk_pos = self.cached_rtk_pos
        rtk_vel = self.cached_rtk_vel
        
        state_map[0, 0, 0, 0] = (self.goal.x - rtk_pos.x) / 500.0
        state_map[0, 0, 0, 1] = (self.goal.y - rtk_pos.y) / 500.0
        state_map[0, 0, 0, 2] = (self.goal.z - rtk_pos.z) / 100.0
        state_map[0, 0, 0, 3] = rtk_vel.x / self.Max_V
        state_map[0, 0, 0, 4] = rtk_vel.y / self.Max_V
        
        # ==========================================
        # 2. LiDAR (10Hz) 更新邏輯與柵格化 (對應 4m / 5.6m 規格)
        # ==========================================
        if self.Step % self.steps_per_lidar == 0 or self.cached_lidar_dists is None:
            # 這裡我們使用相對朝向，讓網格保持「自我中心 (Egocentric)」或是「全域對齊」
            # 若要讓網格方向跟隨無人機車頭，可直接用 heading；否則用 0 代表北方對齊
            heading = calculate_angle(Loc(0,0,0), rtk_vel)
            
            # 執行掃描取得原始距離
            dists, angles = self.lidar.scan(rtk_pos, heading, self.env.buildings)
            self.cached_lidar_dists = dists
            self.cached_lidar_angles = angles
            
            # 初始化 11x11 網格，預設全部為 -1 (未知區域)
            grid = np.full((grid_dim, grid_dim), -1.0)
            
            # 【關鍵修改】：完美對應圖片規格
            # 5格 * 0.8m = 4.0m (剛好對應圖片中前方 4m)
            resolution = 0.8  
            offset = grid_dim // 2  # 中心點索引為 5
            
            for dist, angle in zip(dists, angles):
                # 如果你想讓網路輸入是以無人機為正前方的「局部相對座標」，這裡直接用 angle 即可；
                # 但若您的架構要求全域座標，則使用 heading + angle。這裡保持與物理方向一致。
                global_angle = heading + angle
                dx = math.cos(global_angle)
                dy = math.sin(global_angle)
                
                # 沿著射線步進，標記空氣區為 0 (提高步進密度確保不漏格)
                step_size = resolution / 3.0 
                current_dist = 0.0
                
                # 最大只遍歷到 5.6m (圖片規格)
                while current_dist < dist and current_dist <= 5.6:
                    x = current_dist * dx
                    y = current_dist * dy
                    idx_x = int(round(x / resolution)) + offset
                    idx_y = int(round(y / resolution)) + offset
                    
                    if 0 <= idx_x < grid_dim and 0 <= idx_y < grid_dim:
                        if grid[idx_x, idx_y] == -1.0:
                            grid[idx_x, idx_y] = 0.0
                    current_dist += step_size
                    
                # 標記障礙物邊緣點為 1
                if dist < self.lidar.max_range and dist <= 5.6:
                    hx = dist * dx
                    hy = dist * dy
                    idx_x = int(round(hx / resolution)) + offset
                    idx_y = int(round(hy / resolution)) + offset
                    if 0 <= idx_x < grid_dim and 0 <= idx_y < grid_dim:
                        grid[idx_x, idx_y] = 1.0
                        
            self.cached_lidar_grid = grid.flatten()
            
        # 將快取的 LiDAR 柵格數據放入狀態向量
        state_map[0, 0, 0, 5:126] = self.cached_lidar_grid
                
        return copy.copy(state_map[0, 0, 0, :])
    
# ==========================================
# LiDAR 模擬器 (對應附圖規格: 240度, 683道射線, 5.6m)
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