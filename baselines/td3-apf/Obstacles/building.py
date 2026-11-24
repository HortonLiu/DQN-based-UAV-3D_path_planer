import csv
from BaseClass.BaseThreaten import *
from BaseClass.CalMod import *
import random

class building(BaseThreaten):
    def __init__(self, param, env=None):
        super().__init__(param)
        self.env = env
        self.position = Loc(
            float(param.get("position").get('x')), 
            float(param.get("position").get('y')), 
            float(param.get("position").get('z'))
        )
        self.position_ = self.position
        self._R = None2Value(float(param.get("_R")), 10)
        self._H = None2Value(float(param.get("_H")), 20)

    def reset(self):
        pass

    # [關鍵修正] 檢查傳入的禁區列表 (safe_zones)
    def reset_random(self, safe_zones=None):
        if self.env is not None:
            max_retries = 1000 
            for _ in range(max_retries):
                # 1. 隨機生成座標
                tx = random.uniform(0, self.env.len)
                ty = random.uniform(0, self.env.width)
                
                # 2. 檢查是否與任何一個禁區衝突
                is_valid = True
                if safe_zones:
                    for zone in safe_zones:
                        z_x1, z_x2, z_y1, z_y2 = zone
                        
                        # 判定是否重疊
                        b_x1, b_x2 = tx - self._R, tx + self._R
                        b_y1, b_y2 = ty - self._R, ty + self._R
                        
                        if (b_x1 < z_x2 and b_x2 > z_x1 and
                            b_y1 < z_y2 and b_y2 > z_y1):
                            is_valid = False
                            break
                
                # 3. 如果都不衝突，就確定是這個位置
                if is_valid:
                    self.position.x = tx
                    self.position.y = ty
                    self.position_ = self.position
                    return

            print("Warning: Could not find valid position for building after retries.")
            self.position.x = tx
            self.position.y = ty

    def check_threaten(self, position: Loc):
        if position.z > self._H:
            return 0
        if Eu_Loc_distance(Loc(position.x, position.y, self.position.z), self.position) < self._R:
            return 1
        else:
            return 0

    def run(self):
        pass
