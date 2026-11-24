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
        self.clear_motion()

    def reset(self):
        if self.is_dynamic:
            self.reset_motion()

    def clear_motion(self):
        """Leave an existing cylinder static, without changing its geometry."""
        self.is_dynamic = False
        self.speed_m_s = self.speed = 0.0
        self.velocity = Loc(0.0, 0.0, 0.0)
        self.initial_position = Loc(self.position.x, self.position.y, self.position.z)
        self.motion_endpoints = []
        self.history = []
        self.elapsed_time = 0.0

    def configure_motion(self, direction, speed_m_s=4.0, bounds=None,
                         control_dt=1.0 / 480.0, record_hz=20.0, half_range=None):
        """Assign reciprocal motion to this cylinder along one fixed straight line.

        Bounds constrain the center (xmin, xmax, ymin, ymax). The cylinder radius
        and the current randomized center remain unchanged. With half_range=None,
        the endpoints are the intersections of that line with the map boundary.
        A positive half_range optionally limits travel from the initial center.
        """
        dx, dy = (float(value) for value in direction)
        norm = math.hypot(dx, dy)
        speed_m_s, control_dt, record_hz = map(float, (speed_m_s, control_dt, record_hz))
        if (not math.isfinite(norm) or norm <= 0
                or not math.isfinite(speed_m_s) or speed_m_s < 0
                or not math.isfinite(control_dt) or control_dt <= 0
                or not math.isfinite(record_hz) or record_hz <= 0):
            raise ValueError("Motion direction, speed, control_dt and record_hz must be finite and valid")
        if bounds is None:
            if self.env is None:
                raise ValueError("Motion needs an environment or explicit center bounds")
            bounds = (0.0, self.env.len, 0.0, self.env.width)
        xmin, xmax, ymin, ymax = map(float, bounds)
        if (not all(math.isfinite(value) for value in (xmin, xmax, ymin, ymax))
                or xmin >= xmax or ymin >= ymax):
            raise ValueError("Motion center bounds must have positive width and height")
        if not (xmin <= self.position.x <= xmax and ymin <= self.position.y <= ymax):
            raise ValueError("Initial obstacle center lies outside its motion bounds")
        unit = (dx / norm, dy / norm)
        lower, upper = -float('inf'), float('inf')
        for center, component, lo, hi in (
                (self.position.x, unit[0], xmin, xmax),
                (self.position.y, unit[1], ymin, ymax)):
            if abs(component) > 1e-15:
                intersections = ((lo - center) / component, (hi - center) / component)
                lower = max(lower, min(intersections))
                upper = min(upper, max(intersections))
        if half_range is not None:
            half_range = float(half_range)
            if not math.isfinite(half_range) or half_range <= 0:
                raise ValueError("Motion half_range must be positive and finite")
            lower, upper = max(lower, -half_range), min(upper, half_range)
        if upper - lower <= 1e-12:
            raise ValueError("Motion line must have a nonzero segment inside the bounds")
        self.is_dynamic = True
        self.speed_m_s = self.speed = speed_m_s
        self.control_dt = control_dt
        self.record_hz = record_hz
        self.motion_bounds = (xmin, xmax, ymin, ymax)
        self.motion_half_range = half_range
        self._motion_unit = unit
        self._motion_lower, self._motion_upper = lower, upper
        self.initial_position = Loc(self.position.x, self.position.y, self.position.z)
        self.motion_endpoints = [
            [self.position.x + unit[0] * distance,
             self.position.y + unit[1] * distance, self.position.z]
            for distance in (lower, upper)]
        self.reset_motion()

    def reset_motion(self):
        """Restart the episode at the center used by configure_motion()."""
        if not self.is_dynamic:
            return
        self.position = Loc(self.initial_position.x, self.initial_position.y, self.initial_position.z)
        self.position_ = self.position
        self.elapsed_time = 0.0
        self._next_record_time = 1.0 / self.record_hz
        self.history = [[0.0, self.position.x, self.position.y, self.position.z]]
        self._set_motion_position()
        self.initial_velocity = Loc(self.velocity.x, self.velocity.y, self.velocity.z)

    def _set_motion_position(self):
        # A triangular wave reflects all overshoot while staying on the SAME line.
        width = self._motion_upper - self._motion_lower
        phase = (-self._motion_lower + self.speed_m_s * self.elapsed_time) % (2.0 * width)
        if phase < width:
            coordinate, sign = self._motion_lower + phase, 1.0
        else:
            coordinate, sign = self._motion_upper - (phase - width), -1.0
        ux, uy = self._motion_unit
        self.position.x = self.initial_position.x + ux * coordinate
        self.position.y = self.initial_position.y + uy * coordinate
        self.velocity = Loc(sign * self.speed_m_s * ux, sign * self.speed_m_s * uy, 0.0)

    @property
    def motion_path(self):
        return [sample[1:] for sample in self.history]

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

    def run(self, dt=None):
        """Advance one physical update using m/s * seconds, exactly once."""
        if not self.is_dynamic:
            return
        dt = self.control_dt if dt is None else float(dt)
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Motion dt must be finite and nonnegative")
        if dt == 0:
            return
        self.elapsed_time += dt
        self._set_motion_position()
        if self.elapsed_time + 1e-10 >= self._next_record_time:
            self.history.append([self.elapsed_time, self.position.x, self.position.y, self.position.z])
            self._next_record_time = (math.floor((self.elapsed_time + 1e-10) * self.record_hz) + 1) / self.record_hz
