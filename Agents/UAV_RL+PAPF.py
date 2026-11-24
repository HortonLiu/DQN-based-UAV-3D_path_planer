"""SAC + PAPF using the revised manuscript, Eqs. (10)--(12), (17)--(21).

Default units: 1 grid = 1 m; one executed step travels 1 m. Each rollout
predicts five steps from the SAME physical position using one frozen 240-degree,
683-ray, 5.6 m LiDAR scan. Ten consecutive clear rollouts permit ONE physical
step. A stagnation failure adds a point source and resets the clear streak;
ten failed rollouts or 100 total rollouts in one control call cause a hover
instead of forced motion. The total budget counts successful and failed
rollouts and never resets within that control call.
Identical deterministic rollouts are intentionally repeated to meet this gate;
they do not provide independent observations or form a 50 m prediction.

Optional XML keys (defaults): papf_prediction_steps=5, papf_required_passes=10,
papf_max_replans=10, papf_max_prediction_iterations=100,
papf_force_threshold=0.1, papf_goal_tolerance=1.0,
papf_virtual_gain=10.0, papf_virtual_radius=5.6, papf_max_virtual_obstacles=200,
papf_grid_resolution=1.0, papf_collision_margin=0.05, papf_cycle_distance=0.25,
goal_reach_distance=15.0. The force threshold, goal tolerance, collision margin
and cycle distance are engineering choices, not manuscript numerical values.
A recurrence guard supplements Eq. (10) because fixed 1 m steps can oscillate
between non-equilibrium nodes. Prediction rejects unobserved
space; its only geometric obstacle data are cylinders hit in the current scan.
Visible cylinder geometry is supplied by the simulator, i.e. assumes ideal
shape estimation from LiDAR. No hidden cylinders enter the prediction field.

To select this module with the existing AgentFactory, use
<Agent_Type>UAV_RL+PAPF</Agent_Type>. The class alias below supports that name.
The current 126-input, 3-action SAC trainer is compatible. This module does not
change the active XML configuration or impose wall-clock control frequencies.
"""

import copy
import csv
import math

import numpy as np

from Agents.UAV import UAV as _BaseUAV
from BaseClass.CalMod import Loc


class UAV(_BaseUAV):
    def __init__(self, param: dict, env=None) -> None:
        settings = dict(param)
        # The requested kinematics are fixed at one grid per navigation step.
        settings["Max_V"] = 1
        super().__init__(settings, env)
        self.prediction_steps = self._positive_int(settings, "papf_prediction_steps", 5)
        self.required_prediction_passes = self._positive_int(settings, "papf_required_passes", 10)
        self.max_prediction_replans = self._positive_int(settings, "papf_max_replans", 10)
        self.max_prediction_iterations = self._positive_int(settings, "papf_max_prediction_iterations", 100)
        if self.max_prediction_iterations < self.required_prediction_passes:
            raise ValueError("papf_max_prediction_iterations must be at least papf_required_passes")
        self.max_virtual_obstacles = self._positive_int(settings, "papf_max_virtual_obstacles", 200)
        self.force_threshold = self._positive_float(settings, "papf_force_threshold", 0.1)
        self.prediction_goal_tolerance = self._positive_float(settings, "papf_goal_tolerance", 1.0)
        self.virtual_gain = self._positive_float(settings, "papf_virtual_gain", 10.0)
        self.virtual_radius = self._positive_float(settings, "papf_virtual_radius", 5.6)
        self.grid_resolution = self._positive_float(settings, "papf_grid_resolution", 1.0)
        self.collision_margin = self._positive_float(settings, "papf_collision_margin", 0.05)
        self.cycle_distance = self._positive_float(settings, "papf_cycle_distance", 0.25)
        self.goal_reach_distance = self._positive_float(settings, "goal_reach_distance", 15.0)
        if self.prediction_steps * self.Max_V > self.lidar.max_range:
            raise ValueError("PAPF prediction length must not exceed the 5.6 m LiDAR range")
        if self.virtual_radius <= 0.1:
            raise ValueError("papf_virtual_radius must exceed the 0.1 m distance floor")
        self.active_virtual_obstacles = []
        self.virtual_obstacle_records = []
        self._scan_snapshot = None
        self._scan_key = None
        self.V_dir = math.atan2(self.goal.y - self.position.y, self.goal.x - self.position.x)
        self.last_prediction_paths = []
        self.last_prediction_attempts = 0
        self.last_prediction_failures = 0
        self.last_prediction_passes = 0
        self.last_prediction_reason = None
        self.papf_status = "ready"
        self.physical_execution_count = 0
        self.hover_count = 0

    @staticmethod
    def _positive_int(settings, name, default):
        raw = float(settings.get(name, default))
        if not math.isfinite(raw) or raw < 1 or not raw.is_integer():
            raise ValueError("{} must be a positive integer".format(name))
        return int(raw)

    @staticmethod
    def _positive_float(settings, name, default):
        value = float(settings.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("{} must be finite and positive".format(name))
        return value

    @staticmethod
    def _distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _heading(self):
        if math.hypot(self.V_vector.x, self.V_vector.y) > 1e-9:
            return math.atan2(self.V_vector.y, self.V_vector.x)
        return self.V_dir

    def _get_scan_snapshot(self):
        """Ray-cast at the real pose only; keep the result fixed during replans.

        Only first-hit cylinder identities are retained. Prediction receives
        immutable copies of their geometry, never the environment's live objects.
        """
        heading = self._heading()
        key = (self.Step, self.position.x, self.position.y, self.position.z, heading)
        if key == self._scan_key and self._scan_snapshot is not None:
            return self._scan_snapshot
        angles = self.lidar.angles + heading
        dx, dy = np.cos(angles), np.sin(angles)
        distances = np.full(len(angles), self.lidar.max_range, dtype=float)
        owners = np.full(len(angles), -1, dtype=int)
        cylinders = []
        for building in self.env.buildings:
            if self.position.z > getattr(building, "_H", float("inf")):
                continue
            cx, cy, radius = float(building.position.x), float(building.position.y), float(building._R)
            if math.hypot(self.position.x - cx, self.position.y - cy) - radius > self.lidar.max_range:
                continue
            cylinders.append((cx, cy, radius))
            vx, vy = self.position.x - cx, self.position.y - cy
            projection = vx * dx + vy * dy
            discriminant = projection ** 2 - (vx ** 2 + vy ** 2 - radius ** 2)
            roots = -projection - np.sqrt(np.maximum(discriminant, 0.0))
            if vx ** 2 + vy ** 2 < radius ** 2:
                roots = np.zeros_like(roots)
            hit = (discriminant >= 0) & (roots >= 0) & (roots < distances)
            distances[hit] = roots[hit]
            owners[hit] = len(cylinders) - 1
        # Known map boundaries also terminate LiDAR rays.
        for component, origin, limit in ((dx, self.position.x, self.env.len),
                                         (dy, self.position.y, self.env.width)):
            for boundary in (0.0, float(limit)):
                roots = np.full(len(angles), np.inf)
                np.divide(boundary - origin, component, out=roots, where=np.abs(component) > 1e-9)
                hit = (roots >= 0) & (roots < distances)
                distances[hit] = roots[hit]
                owners[hit] = -2
        visible = tuple(cylinders[i] for i in sorted(set(owners[owners >= 0].tolist())))
        self._scan_snapshot = {
            "origin": copy.copy(self.position), "heading": heading,
            "distances": distances, "angles": angles, "obstacles": visible,
        }
        self._scan_key = key
        return self._scan_snapshot

    def _observed_free(self, position, snapshot):
        origin = snapshot["origin"]
        distance = self._distance(position, origin)
        if distance < 1e-8:
            return True
        bearing = math.atan2(position.y - origin.y, position.x - origin.x)
        relative = math.atan2(math.sin(bearing - snapshot["heading"]),
                              math.cos(bearing - snapshot["heading"]))
        if distance > self.lidar.max_range or abs(relative) > self.lidar.fov / 2:
            return False
        # Adjacent beams conservatively bound visibility, including shadows.
        ray = (relative + self.lidar.fov / 2) / self.lidar.fov * (self.lidar.num_rays - 1)
        lo = max(0, min(int(math.floor(ray)), self.lidar.num_rays - 1))
        hi = min(lo + 1, self.lidar.num_rays - 1)
        return distance + self.collision_margin < min(snapshot["distances"][lo], snapshot["distances"][hi])

    def _segment_clear(self, start, end, snapshot):
        if not (0 <= end.x <= self.env.len and 0 <= end.y <= self.env.width):
            return False
        vx, vy = end.x - start.x, end.y - start.y
        length_squared = vx * vx + vy * vy
        for cx, cy, radius in snapshot["obstacles"]:
            t = ((cx - start.x) * vx + (cy - start.y) * vy) / max(length_squared, 1e-12)
            t = max(0.0, min(1.0, t))
            if math.hypot(start.x + t * vx - cx, start.y + t * vy - cy) <= radius + self.collision_margin:
                return False
        # Check swept motion as well as its endpoint against the sensed region.
        samples = max(1, int(math.ceil(math.sqrt(length_squared) / 0.1)))
        for index in range(1, samples + 1):
            ratio = index / samples
            if not self._observed_free(Loc(start.x + ratio * vx, start.y + ratio * vy, start.z), snapshot):
                return False
        return True

    def cal_force(self, current_pos, k_rep_obs=15.0, k_att=1.5, snapshot=None, limit=True):
        """Eqs. (2), (4), (5), (11), (12), with an optional 20-unit limit.

        Eq. (10) must examine the uncapped resultant, even when a configured
        stagnation threshold is larger than the navigation-force limit.
        """
        if snapshot is None:
            snapshot = self._get_scan_snapshot()
        tx, ty = self.goal.x - current_pos.x, self.goal.y - current_pos.y
        target_distance = math.hypot(tx, ty)
        fx, fy = (k_att * tx / target_distance, k_att * ty / target_distance) if target_distance else (0.0, 0.0)
        influence = self.lidar.max_range
        for cx, cy, radius in snapshot["obstacles"]:
            vx, vy = current_pos.x - cx, current_pos.y - cy
            distance = math.hypot(vx, vy)
            if distance < radius + influence:
                clearance = max(distance - radius, 0.1)
                strength = k_rep_obs * (1.0 / clearance - 1.0 / influence) ** 2
                fx += strength * vx / (distance + 0.001)
                fy += strength * vy / (distance + 0.001)
        for distance, ux, uy in ((current_pos.x, 1, 0), (self.env.len - current_pos.x, -1, 0),
                                  (current_pos.y, 0, 1), (self.env.width - current_pos.y, 0, -1)):
            if distance < influence:
                strength = k_rep_obs * (1.0 / max(distance, 0.1) - 1.0 / influence) ** 2
                fx += strength * ux
                fy += strength * uy
        for source in self.active_virtual_obstacles:
            vx, vy = current_pos.x - source["pos"].x, current_pos.y - source["pos"].y
            distance = math.hypot(vx, vy)
            if distance < self.virtual_radius:
                strength = source["force"] * (1.0 / max(distance, 0.1) - 1.0 / self.virtual_radius) ** 2
                # Exactly coincident nodes contribute zero, as in manuscript Eq. (11).
                fx += strength * vx / (distance + 0.001)
                fy += strength * vy / (distance + 0.001)
        magnitude = math.hypot(fx, fy)
        if limit and magnitude > 20.0:
            fx, fy = fx * 20.0 / magnitude, fy * 20.0 / magnitude
        return Loc(fx, fy, 0)

    def _virtual_prediction(self, steering_bias, k_rep_obs, k_att, snapshot):
        """Return (local path, flagged node or None, rejection reason or None)."""
        node = copy.copy(self.position)
        path = [copy.copy(node)]
        for step in range(self.prediction_steps + 1):
            force = self.cal_force(node, k_rep_obs, k_att, snapshot, limit=False)
            target_distance = self._distance(node, self.goal)
            # Eq. (10) is evaluated at every virtual node, including the endpoint.
            if math.hypot(force.x, force.y) < self.force_threshold and target_distance > self.prediction_goal_tolerance:
                return path, copy.copy(node), "stagnation"
            if step == self.prediction_steps:
                break
            if target_distance <= self.prediction_goal_tolerance:
                # A goal node is terminal, never a stagnation point.
                path.extend(copy.copy(node) for _ in range(self.prediction_steps - step))
                break
            angle = math.atan2(force.y, force.x) + steering_bias * self.Steering_angle
            candidate = Loc(node.x + self.Max_V * math.cos(angle),
                            node.y + self.Max_V * math.sin(angle), node.z)
            if not self._segment_clear(node, candidate, snapshot):
                return path, None, "blocked_or_unobserved"
            # A fixed-length update can jump over an equilibrium indefinitely.
            # Reject recurrence to an earlier non-adjacent node as stagnation,
            # even when the sampled endpoint forces do not satisfy Eq. (10).
            if (self._distance(candidate, self.goal) > self.prediction_goal_tolerance
                    and any(self._distance(candidate, previous) <= self.cycle_distance
                            for previous in path[:-1])):
                path.append(copy.copy(candidate))
                return path, copy.copy(candidate), "prediction_cycle"
            node = candidate
            path.append(copy.copy(node))
        return path, None, None

    def _add_virtual_obstacle(self, position):
        # Repeated flags create sources at the exact flagged node, never offsets
        # or randomly chosen escape directions. Bound retained sources per episode.
        if len(self.active_virtual_obstacles) >= self.max_virtual_obstacles:
            self.active_virtual_obstacles.pop(0)
        self.active_virtual_obstacles.append({"pos": copy.copy(position), "force": self.virtual_gain})
        self.virtual_obstacle_records = [
            {"pos": copy.copy(source["pos"]), "max_force": source["force"]}
            for source in self.active_virtual_obstacles
        ]

    def _plan(self, steering_bias, k_rep_obs, k_att, snapshot):
        """Bound all rollouts independently of failure and clear-streak counts."""
        self.last_prediction_paths = []
        self.last_prediction_attempts = 0
        self.last_prediction_failures = 0
        self.last_prediction_passes = 0
        self.last_prediction_reason = None
        # Every iteration consumes the total budget, including successful ones.
        # A fixed for-loop makes termination independent of streak resets.
        for _ in range(self.max_prediction_iterations):
            path, stagnation, reason = self._virtual_prediction(steering_bias, k_rep_obs, k_att, snapshot)
            self.last_prediction_attempts += 1
            self.last_prediction_paths.append([[p.x, p.y, p.z] for p in path])
            self.last_prediction_reason = reason
            if reason is not None:
                self.last_prediction_failures += 1
                self.last_prediction_passes = 0
                if stagnation is not None:
                    self._add_virtual_obstacle(stagnation)
                if self.last_prediction_failures >= self.max_prediction_replans:
                    self.papf_status = "hover_replan_limit"
                    return None
                continue
            self.last_prediction_passes += 1
            if self.last_prediction_passes == self.required_prediction_passes:
                self.papf_status = "execute"
                return path
        self.papf_status = "hover_prediction_limit"
        return None

    def state_PathPlan(self):
        """126 features: relative XY, velocity XY, altitude, 11x11 sensed cells."""
        snapshot = self._get_scan_snapshot()
        state = np.zeros(126, dtype=float)
        state[:5] = [(self.goal.x - self.position.x) / self.env.len,
                     (self.goal.y - self.position.y) / self.env.width,
                     self.V_vector.x / self.Max_V, self.V_vector.y / self.Max_V,
                     self.position.z / self.env.h]
        grid = np.full((11, 11), -1.0)
        for i in range(11):
            for j in range(11):
                point = Loc(self.position.x + (i - 5) * self.grid_resolution,
                            self.position.y + (j - 5) * self.grid_resolution, self.position.z)
                if self._observed_free(point, snapshot):
                    grid[i, j] = 0.0
        for distance, angle in zip(snapshot["distances"], snapshot["angles"]):
            if distance < self.lidar.max_range:
                i = int(round(distance * math.cos(angle) / self.grid_resolution)) + 5
                j = int(round(distance * math.sin(angle) / self.grid_resolution)) + 5
                if 0 <= i < 11 and 0 <= j < 11:
                    grid[i, j] = 1.0
        grid[5, 5] = 0.0
        state[5:] = grid.flatten()
        return state

    def _physical_collision(self, start, end):
        """Simulator ground-truth feedback after planning, including a hover.

        This is environment feedback, not perception supplied to the planner.
        Swept circle checks prevent a 1 m step tunneling through a small obstacle.
        """
        if self.env.Threaten_rate(end) == 1:
            return True
        vx, vy = end.x - start.x, end.y - start.y
        length_squared = vx * vx + vy * vy
        for building in self.env.buildings:
            if end.z > getattr(building, "_H", float("inf")):
                continue
            t = ((building.position.x - start.x) * vx + (building.position.y - start.y) * vy) / max(length_squared, 1e-12)
            t = max(0.0, min(1.0, t))
            if math.hypot(start.x + t * vx - building.position.x,
                          start.y + t * vy - building.position.y) <= building._R:
                return True
        return False

    def _finish_step(self, reward, done, info):
        self.done = done
        self.score += reward
        self.total_score += reward
        self.R_record.append(reward)
        return reward, done, info

    def update_PathPlan(self, action):
        if self.done:
            return 0.0, True, "success" if self.reach_goal else "lose"
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size != 3 or not np.all(np.isfinite(action)):
            raise ValueError("SAC+PAPF requires three finite actions: steering, repulsion, attraction")
        steering_bias, repulsive_action, attractive_action = np.clip(action, -1.0, 1.0)
        k_rep_obs = max(5.0, (repulsive_action + 1.0) * 15.0)
        k_att = max(0.5, (attractive_action + 1.0) * 1.5)
        old_position = copy.copy(self.position)
        old_distance = self._distance(self.position, self.goal)
        # Collision has priority over goal attainment, including an obstacle
        # that moved onto a hovering drone since the previous control call.
        if self.env.Threaten_rate(old_position) == 1:
            self.Step += 1
            self.Train_epoch += 1
            self.V_vector = Loc(0, 0, 0)
            self.V = 0.0
            self.path.append([self.position.x, self.position.y, self.position.z])
            self.V_record.append(0.0)
            self.papf_status = "collision"
            self._scan_snapshot = None
            self._scan_key = None
            return self._finish_step(-100.0, True, "meet_threaten")
        if old_distance < self.goal_reach_distance:
            self.reach_goal = 1
            self.target_step = self.Step
            return self._finish_step(100.0 + 50.0 * max(0, self.original_max_step - self.Step), True, "success")
        snapshot = self._get_scan_snapshot()
        path = self._plan(float(steering_bias), float(k_rep_obs), float(k_att), snapshot)
        # Rollouts above never change physical position, velocity, Step or rewards.
        self.Step += 1
        self.Train_epoch += 1
        boundary_violation = False
        collision = False
        if path is None:
            self.V_vector = Loc(0, 0, 0)
            self.V = 0.0
            self.hover_count += 1
            collision = self._physical_collision(old_position, old_position)
            if collision:
                self.papf_status = "collision"
        else:
            candidate = copy.copy(path[1])
            boundary_violation = not (0 <= candidate.x <= self.env.len and 0 <= candidate.y <= self.env.width)
            collision = self._physical_collision(old_position, candidate)
            if collision:
                self.V_vector = Loc(0, 0, 0)
                self.V = 0.0
                self.papf_status = "collision"
            else:
                self.position = candidate
                self.V_vector = Loc(candidate.x - old_position.x, candidate.y - old_position.y, 0)
                self.V = self.Calc_V()
                self.V_dir = self._heading()
                self.physical_execution_count += 1
        self.path.append([self.position.x, self.position.y, self.position.z])
        self.V_record.append(self.V)
        self.path_len += self._distance(old_position, self.position)
        self._scan_snapshot = None
        self._scan_key = None
        if collision:
            # Eq. (21) specifies -100; Algorithm 1's +100 is a manuscript typo.
            return self._finish_step(-100.0, True, "meet_threaten")
        new_distance = self._distance(self.position, self.goal)
        reward = 2.0 * (old_distance - new_distance) - 0.02 * abs(steering_bias) - 0.05 - 10.0 * boundary_violation
        if new_distance < self.goal_reach_distance:
            self.reach_goal = 1
            self.target_step = self.Step
            reward += 100.0 + 50.0 * max(0, self.original_max_step - self.Step)
            with open("path.csv", "w", newline="") as stream:
                csv.writer(stream).writerows(self.path)
            return self._finish_step(reward, True, "success")
        if self.Step >= self.Max_Step:
            return self._finish_step(reward - 50.0, True, "lose")
        return self._finish_step(reward, False, "normal")

    def reset(self, option=None):
        super().reset(option)
        self.last_prediction_paths = []
        self.last_prediction_attempts = 0
        self.last_prediction_failures = 0
        self.last_prediction_passes = 0
        self.last_prediction_reason = None
        self._scan_snapshot = None
        self._scan_key = None
        self.papf_status = "ready"
        if option != "local reset":
            self.active_virtual_obstacles = []
            self.virtual_obstacle_records = []
            self.physical_execution_count = 0
            self.hover_count = 0
            # Initial yaw points toward the goal so the first scan faces the task.
            self.V_dir = math.atan2(self.goal.y - self.position.y, self.goal.x - self.position.x)
            self.V_vector = Loc(self.Max_V * math.cos(self.V_dir), self.Max_V * math.sin(self.V_dir), 0)
            self.V = self.Calc_V()


# importlib accepts '+' in module names; getattr needs the matching exported name.
globals()["UAV_RL+PAPF"] = UAV
