import time
from BaseClass.BaseAgent import *
from BaseClass.CalMod import *
from BaseClass.BaseCNN import *
# from PathPlan.RRT import * # 已移除 RRT
from GN import *
import random
import csv
import copy
import os
import collections # 用於 BFS
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
        self.Max_Step=int(param.get("Max_Step"))
        self.Step=0
        self.f_max=float(param.get("f_max"))+0.1*self.j

        # [動作空間] 5 維
        # 0: Steering_Bias, 1: k_rep_obs, 2: k_vortex, 3: R_vortex, 4: k_att
        
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
        
        # [New] 專家路徑 (BFS Path)
        self.expert_path = []
        
        self.total_score=0
        self.Train_start=time.time()
        self.Train_pointcut=time.time()
        self.path_len=0
        self.Train_epoch=0
        self.reach_goal=0
        self.start2goal=Eu_Loc_distance(self.position, self.goal)
        self.len_Astar=0 

    # [新增] 快速 BFS 路徑搜尋 (用於引導訓練)
    def run_bfs(self):
        # 1. 簡化網格 (Grid Resolution)
        # 地圖 500x500, 我們用 20m 一格，變成 25x25 的網格，運算極快
        grid_size = 20
        rows = int(self.env.width / grid_size) + 1
        cols = int(self.env.len / grid_size) + 1
        
        start_node = (int(self.position.x / grid_size), int(self.position.y / grid_size))
        goal_node = (int(self.goal.x / grid_size), int(self.goal.y / grid_size))
        
        queue = collections.deque([[start_node]])
        visited = set([start_node])
        
        # 4-Connectivity
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        self.expert_path = [] # 清空舊路徑
        
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            
            if (x, y) == goal_node:
                # 找到路徑，將網格座標轉換回真實座標
                self.expert_path = [Loc(nx * grid_size + grid_size/2, ny * grid_size + grid_size/2, 0) for nx, ny in path]
                return

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows and (nx, ny) not in visited:
                    # 碰撞檢查 (檢查網格中心點)
                    check_pos = Loc(nx * grid_size + grid_size/2, ny * grid_size + grid_size/2, 0)
                    if self.env.Threaten_rate(check_pos) == 0: # 無障礙
                        visited.add((nx, ny))
                        new_path = list(path)
                        new_path.append((nx, ny))
                        queue.append(new_path)
        
        # 若找不到路徑 (例如起點被包圍)，expert_path 為空

    def sense_obstacle_direction(self, current_pos, scan_radius=40.0):
        if scan_radius is None:
            scan_radius = self.Scan_Radius
        if self.V_vector.x == 0 and self.V_vector.y == 0:
            heading = 0 
        else:
            heading = math.atan2(self.V_vector.y, self.V_vector.x)
            
        density_left = 0
        density_right = 0
        scan_step_angle = 15 
        scan_step_dist = 10  
        
        for r in range(5, int(scan_radius), scan_step_dist):
            for angle_deg in range(10, 80, scan_step_angle): 
                angle_l = heading + math.radians(angle_deg)
                p_l = Loc(current_pos.x + r * math.cos(angle_l), current_pos.y + r * math.sin(angle_l), current_pos.z)
                if self.env.Threaten_rate(p_l) > 0:
                    density_left += 1
                
                angle_r = heading - math.radians(angle_deg)
                p_r = Loc(current_pos.x + r * math.cos(angle_r), current_pos.y + r * math.sin(angle_r), current_pos.z)
                if self.env.Threaten_rate(p_r) > 0:
                    density_right += 1
                    
        if density_left > density_right:
            return -1.0 
        elif density_right > density_left:
            return 1.0
        else:
            vec_to_goal_x = self.goal.x - current_pos.x
            vec_to_goal_y = self.goal.y - current_pos.y
            cross = self.V_vector.x * vec_to_goal_y - self.V_vector.y * vec_to_goal_x
            return 1.0 if cross > 0 else -1.0

    def cal_force(self, current_pos, k_rep_obs=15.0, k_vortex=2.0, R_vortex=60.0, k_att=1.0):
        total_force = Loc(0, 0, 0)
        f_obs_sum = Loc(0, 0, 0)
        repulsion_range = 25.0 
        rotation_sign = self.sense_obstacle_direction(current_pos, R_vortex)

        for threaten in self.env.buildings:
            dis = Eu_Loc_distance(current_pos, threaten.position)
            if dis < max(repulsion_range + threaten._R, R_vortex + threaten._R):
                vec_x = current_pos.x - threaten.position.x
                vec_y = current_pos.y - threaten.position.y
                norm = math.sqrt(vec_x**2 + vec_y**2) + 0.001
                unit_x = vec_x / norm
                unit_y = vec_y / norm

                if dis < threaten._R + repulsion_range:
                    d_safe = max(dis - threaten._R, 0.1)
                    rep_strength = k_rep_obs * (1.0/d_safe - 1.0/repulsion_range)**2
                    f_obs_sum.x += rep_strength * unit_x
                    f_obs_sum.y += rep_strength * unit_y

                if dis < threaten._R + R_vortex:
                    vor_strength = abs(k_vortex) * (1.0 / (max(dis - threaten._R, 1.0)))
                    f_vor_x = -unit_y * rotation_sign * vor_strength
                    f_vor_y =  unit_x * rotation_sign * vor_strength
                    f_obs_sum.x += f_vor_x
                    f_obs_sum.y += f_vor_y
        
        f_att = Loc(0, 0, 0)
        att_strength = k_att * 2.0 
        att_angle = calculate_angle(current_pos, self.goal)
        f_att.x = att_strength * math.cos(att_angle)
        f_att.y = att_strength * math.sin(att_angle)

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

        total_force.x = f_obs_sum.x + f_att.x + f_wall.x
        total_force.y = f_obs_sum.y + f_att.y + f_wall.y
        
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
            
            # [新增] 每一回合開始時，計算一次 BFS 最佳路徑
            self.run_bfs()

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
        k_rep_obs = max(20.0, (float(action[1]) + 1.0) * 10.0) if len(action) > 1 else 20.0
        k_vortex = abs(float(action[2])) * 10.0 if len(action) > 2 else 0.0
        R_vortex = max(20.0, (float(action[3]) + 1.0) * 30.0 + 20.0) if len(action) > 3 else 60.0
        k_att = max(0.5, (float(action[4]) + 1.0) * 1.75) if len(action) > 4 else 1.0

        global_r=0
        
        dist_to_goal = Eu_Loc_distance(self.position, self.goal)
        if dist_to_goal < 15: 
            self.done=True
            global_r += 100 
            global_r += (self.Max_Step - self.Step)
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
        
        total_force = self.cal_force(self.position, k_rep_obs, k_vortex, R_vortex, k_att)
        
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

        # --- 獎勵函數 ---
        progress = dis2goal_old - dis2goal_new
        global_r += 1.0 * progress 

        min_dist_obs = float('inf')
        for b in self.env.buildings:
            d = Eu_Loc_distance(self.position, b.position) - b._R
            if d < min_dist_obs: min_dist_obs = d
        
        if min_dist_obs < 30.0 and progress < 0.2:
            global_r -= 1.0 

        # [關鍵] BFS 引導獎勵 (前半段 Epoch 開啟)
        # self.Trainer.max_epoch 是總回合數
        # 如果有找到專家路徑，且處於訓練前期，計算偏離程度
        if len(self.expert_path) > 0 and self.Trainer.epoch < self.Trainer.max_epoch * 0.5:
            # 尋找無人機目前位置距離 BFS 路徑上最近的點
            min_dist_to_path = float('inf')
            # 為了效能，可以只搜尋部分路徑，這裡簡單搜全部
            for p in self.expert_path:
                d = Eu_Loc_distance(self.position, p)
                if d < min_dist_to_path:
                    min_dist_to_path = d
            
            # 偏離懲罰：每偏離 1m 扣 0.05 分
            # 這樣 Agent 會試圖調整 APF 參數(如漩渦力)來貼合 BFS 建議的路徑
            global_r -= 0.05 * min_dist_to_path

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

    # [關鍵修改] 狀態空間：126 維
    # 1. 相對目標 (3)
    # 2. 速度 (2)
    # 3. 11x11 局部地圖 (121) -> 擴大偵測範圍到 100m
    def state_PathPlan(self):
        grid_dim = 11
        # 總輸入維度 = 3 + 2 + 121 = 126
        state_map = np.zeros((1, 1, 1, 126))
        
        # 1. 目標相對位置
        state_map[0, 0, 0, 0] = (self.goal.x - self.position.x) / 500.0
        state_map[0, 0, 0, 1] = (self.goal.y - self.position.y) / 500.0
        state_map[0, 0, 0, 2] = (self.goal.z - self.position.z) / 100.0
        
        # 2. 飛行速度分量
        state_map[0, 0, 0, 3] = self.V_vector.x / self.Max_V
        state_map[0, 0, 0, 4] = self.V_vector.y / self.Max_V
        
        # 3. 局部感知小地圖 (11x11, Stride 10m => 覆蓋半徑約 50m~70m)
        stride = 10.0
        offset = (grid_dim // 2) * stride
        
        idx = 5
        for i in range(grid_dim):
            for j in range(grid_dim):
                dx = i * stride - offset
                dy = j * stride - offset
                test_pos = Loc(self.position.x + dx, self.position.y + dy, self.position.z)
                
                # 這裡直接用 Threaten_rate，地圖邊界會自動回傳 1
                threat = self.env.Threaten_rate(test_pos)
                state_map[0, 0, 0, idx] = threat
                idx += 1
                
        return copy.copy(state_map[0, 0, 0, :])