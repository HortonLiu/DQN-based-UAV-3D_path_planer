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

        self.update_function = self.update_PathPlan
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

    def reset(self,option=None):
        if option=="local reset":
            self.Step=0
            self.score=0
            self.V=self.Calc_V()
        else:
            self.Step=0; self.score=0; self.done=False
            self.V_record=[]; self.R_record=[]
            self.V_vector=Loc(5,3,0)
            seta=random.uniform(0,2*math.pi)
            self.V_dir=seta
            self.V_vector.x=self.Max_V*math.cos(seta)
            self.V_vector.y=self.Max_V*math.sin(seta)
            self.V=self.Calc_V()
            self.infos=[]; self.data_num=0
            
            x=random.uniform(10,210)
            y=random.uniform(1,10)
            
            # 【關鍵修復1】高度拉高到 5.0 防止出生在 Z=0 瞬間判定撞地死亡
            self.position = Loc(x,y,5.0)
            
            # 【關鍵修復2】強制初始化路徑紀錄，給予第一個點，確保 Line3D 不為空
            self.path = [[self.position.x, self.position.y, self.position.z]]
            
            x=random.uniform(330,490)
            y=random.uniform(420,490)
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