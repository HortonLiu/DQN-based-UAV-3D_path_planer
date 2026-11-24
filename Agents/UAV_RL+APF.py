import time
from BaseClass.BaseAgent import *
from BaseClass.CalMod import *
from BaseClass.BaseCNN import *
import sys
import random
import csv
import copy
import os
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

        # ==========================================
        # [動作空間修改] 5 維精簡為 3 維標準 APF 控制
        # 0: Steering_Bias
        # 1: k_rep_obs (直線斥力)
        # 2: k_att (目標引力)
        # ==========================================
        
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

    # ==========================================
    # [修改] 傳統 APF 力場計算 (移除漩渦力)
    # ==========================================
    def cal_force(self, current_pos, k_rep_obs=15.0, k_att=1.0):
        total_force = Loc(0, 0, 0)
        f_obs_sum = Loc(0, 0, 0)
        
        repulsion_range = 25.0 

        for threaten in self.env.buildings:
            dis = Eu_Loc_distance(current_pos, threaten.position)
            
            # 僅在直線斥力影響範圍內進行計算
            if dis < threaten._R + repulsion_range:
                vec_x = current_pos.x - threaten.position.x
                vec_y = current_pos.y - threaten.position.y
                norm = math.sqrt(vec_x**2 + vec_y**2) + 0.001
                unit_x = vec_x / norm
                unit_y = vec_y / norm

                # 直線斥力核心公式
                d_safe = max(dis - threaten._R, 0.1)
                rep_strength = k_rep_obs * (1.0/d_safe - 1.0/repulsion_range)**2
                f_obs_sum.x += rep_strength * unit_x
                f_obs_sum.y += rep_strength * unit_y
        
        # 目標引力
        f_att = Loc(0, 0, 0)
        att_strength = k_att * 2.0 
        att_angle = calculate_angle(current_pos, self.goal)
        f_att.x = att_strength * math.cos(att_angle)
        f_att.y = att_strength * math.sin(att_angle)

        # 牆壁斥力
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

    # ==========================================
    # [修改] 動作解析與路徑規劃更新優化
    # ==========================================
    def update_PathPlan(self, action):
        if hasattr(action, 'cpu'): action = action.cpu().detach().numpy()
        if hasattr(action, 'flatten'): action = action.flatten()
        elif isinstance(action, list): action = np.array(action).flatten()
            
        # 精簡為 3 維動作解析
        steering_bias = action[0] if len(action) > 0 else 0
        k_rep_obs = max(20.0, (float(action[1]) + 1.0) * 10.0) if len(action) > 1 else 20.0
        k_att = max(0.5, (float(action[2]) + 1.0) * 1.75) if len(action) > 2 else 1.0

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
        
        # 呼叫修改後的 3 參數力場函數
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

        min_dist_obs = float('inf')
        for b in self.env.buildings:
            d = Eu_Loc_distance(self.position, b.position) - b._R
            if d < min_dist_obs: min_dist_obs = d
        
        # 停滯懲罰說明修改：若卡死，提示修改引力與斥力參數以突破局部小值
        if min_dist_obs < 30.0 and progress < 0.2:
            global_r -= 5.0 

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

    def state_PathPlan(self):
        state_map=np.zeros((1,1,1,100))
        state_map[0,0,0,0]=self.Step/100
        
        state_map[0,0,0,1]=(self.goal.x-self.position.x)/500.0
        state_map[0,0,0,2]=(self.goal.y-self.position.y)/500.0
        state_map[0,0,0,3]=(self.goal.z-self.position.z)/100.0
        
        state_map[0,0,0,4]=self.V
        state_map[0,0,0,5]=self.V_vector.x
        state_map[0,0,0,6]=self.V_vector.y
        state_map[0,0,0,7]=calculate_angle(Loc(0,0,0),self.V_vector)
        
        state_map[0,0,0,8] = self.position.x / 500.0 
        state_map[0,0,0,9] = (500.0 - self.position.x) / 500.0 
        state_map[0,0,0,10] = self.position.y / 500.0 
        
        for i in range(5):
            for j in range(5):
                dx=(i-2)
                dy=(j-2)
                test_ndoe=Loc(self.position.x+dx,self.position.y+dy,self.position.z)
                threaten=self.env.Threaten_rate(test_ndoe)
                state_map[0,0,0,11+5*i+j]=threaten
        
        for i in range(5):
            for j in range(5):
                dx=(i-2)
                dy=(j-2)
                test_ndoe=Loc(self.position.x+5*dx,self.position.y+5*dy,self.position.z)
                threaten=self.env.Threaten_rate(test_ndoe)
                state_map[0,0,0,36+5*i+j]=threaten
                
        for i in range(5):
            for j in range(5):
                dx=(i-2)
                dy=(j-2)
                test_ndoe=Loc(self.position.x+10*dx,self.position.y+10*dy,self.position.z)
                threaten=self.env.Threaten_rate(test_ndoe)
                state_map[0,0,0,61+5*i+j]=threaten
        
        state_map[0,0,0,86]=(self.goal.x-self.position.x)/500.0
        state_map[0,0,0,87]=(self.goal.y-self.position.y)/500.0
        state_map[0,0,0,88]=(self.goal.z-self.position.z)/100.0
        state_map[0,0,0,89]=self.position.z/10
        
        state_map[0,0,0,90]=self.env.Threaten_rate(Loc(self.position.x,self.position.y,self.position.z-1))
        state_map[0,0,0,91]=self.env.Threaten_rate(Loc(self.position.x,self.position.y,self.position.z-2))
        state_map[0,0,0,92]=self.env.Threaten_rate(Loc(self.position.x,self.position.y,self.position.z-3))
        state_map[0,0,0,93]=self.env.Threaten_rate(Loc(self.position.x,self.position.y,self.position.z-4))
        state_map[0,0,0,94]=self.env.Threaten_rate(Loc(self.position.x,self.position.y,self.position.z-5))
        
        return copy.copy(state_map[0,0,0,:])