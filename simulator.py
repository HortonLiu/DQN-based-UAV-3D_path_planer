"""Choose hardware timing, dynamic obstacles and UAV speed, then start S2R."""
import argparse
import csv
import datetime
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
root = str(ROOT)
sys.path.insert(0, root)
try:
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from BaseClass.CalMod import XML2Dict
    from FactoryClass.EnvFactory import EnvFactory
except ModuleNotFoundError as error:
    raise SystemExit('Missing dependency: {}. Install with:\n  "{}" -m pip install -r "{}"'.format(
        error.name, sys.executable, ROOT / 'requirements.txt')) from error


class simulator:
    CSV_FIELDS = [
        'sum_Episode', 'Episode', ' Score', ' Avg.Score', 'eps-greedy',
        'success', 'failed', 'meet_threaten', 'loss', 'step',
        'avg_trainning_time', 'avg_testing_time', 'total_time',
        'timestamp', 'control_updates', 'simulation_time_s', 'path_length_m',
        'virtual_obstacles', 'episode_budget_s', 'remaining_goal_distance_m', 'html_path',
        'hardware_hz_enabled', 'control_hz', 'control_dt_s',
        'dynamic_obstacles_enabled', 'scene', 'uav_speed_m_s',
    ]
    MANUSCRIPT_CSV_FIELDS = ['controller_profile', 'map_level', 'scene_seed', 'termination_reason']

    def __init__(self, episodes=None, max_seconds=None, no_train=False, config_path=None,
                 scene='static', hardware_hz=True, uav_speed=None,
                 controller_profile='legacy', map_level=None, fixed_map_xml=None):
        if scene not in ('static', 'mixed'):
            raise ValueError('scene must be static or mixed')
        if controller_profile not in ('legacy', 'manuscript'):
            raise ValueError('controller_profile must be legacy or manuscript')
        if map_level is not None and (controller_profile != 'manuscript' or scene != 'static'):
            raise ValueError('map_level requires the manuscript controller and a static scene')
        if map_level not in (None, 1, 2, 3):
            raise ValueError('map_level must be 1, 2 or 3')
        if (map_level == 1) != (fixed_map_xml is not None):
            raise ValueError('Level 1 requires a verified fixed map XML, which is only used for Level 1')
        self.scene = scene
        self.controller_profile = controller_profile
        self.map_level = map_level
        self.fixed_map_xml = Path(fixed_map_xml).resolve() if fixed_map_xml is not None else None
        if hardware_hz not in (0, 1):
            raise ValueError('hardware_hz must be 0 or 1')
        self.hardware_hz_enabled = bool(hardware_hz)
        if uav_speed is not None and uav_speed not in (1, 4):
            raise ValueError('uav_speed must be 1 or 4 m/s')
        self.uav_speed = None if uav_speed is None else float(uav_speed)
        os.chdir(ROOT)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.EnvFactory = EnvFactory()
        self.epoch = 0
        self.env = None
        self.stat = 1
        self.Max_score = float('-inf')
        self.infos = []
        self.executed_time = 0.0
        self.writer = None
        self.CsvWriter = None
        self._result_file = None
        self.no_train = bool(no_train)
        speed_label = 'Vxml' if self.uav_speed is None else 'V{:g}'.format(self.uav_speed)
        stamp = '{}_{}_{}Hz_{}_{}'.format(
            datetime.datetime.now().strftime('%m_%d_%Y(%H_%M_%S_%f)'),
            self.scene, 480 if self.hardware_hz_enabled else 1, speed_label, os.getpid())
        if self.controller_profile == 'manuscript':
            stamp += '_manuscript' if self.map_level is None else '_manuscript_level%d' % self.map_level
        logs = ROOT / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        self.result_path = str(logs / ('score_' + stamp + '.csv'))
        try:
            self.Init_From_XML(config_path)
            if episodes is not None:
                self.num_episodes = int(episodes)
            if self.num_episodes < 1:
                raise ValueError('num_episodes must be positive')
            if self.map_level is not None and self.num_episodes > 1000:
                raise ValueError('the manuscript map schedule covers 1,000 episodes')
            for agent in self.env.Agents:
                if max_seconds is not None:
                    seconds = float(max_seconds)
                    if not math.isfinite(seconds) or seconds <= 0:
                        raise ValueError('max_seconds must be positive and finite')
                    agent.set_episode_duration(seconds)
                if self.no_train:
                    agent.Trainer.Is_Train = 0
            self.Init_Record_Mod()
            self.writer = SummaryWriter(log_dir=str(logs / ('tensorboard_' + stamp)))
        except Exception:
            self.close()
            raise

    def Init_From_XML(self, XML_path=None):
        default_config = 'PathPlan_City_Static.xml' if self.scene == 'static' else 'PathPlan_City.xml'
        path = Path(XML_path) if XML_path else ROOT / 'config' / default_config
        data = XML2Dict(str(path))
        if not data or 'simulator' not in data:
            raise ValueError('Missing simulator configuration in {}'.format(path))
        config = data['simulator']
        self.record_epo = int(config.get('record_epo', 10))
        self.num_episodes = int(config.get('num_episodes', 1000))
        self.max_eps_episode = int(config.get('max_eps_episode', 1000))
        self.min_eps = float(config.get('min_eps', 0.1))
        self.TARGET_UPDATE = int(config.get('TARGET_UPDATE', 10))
        if not config.get('env'):
            raise ValueError('Missing env configuration in {}'.format(path))
        # Set timing before creating the UAV, sensors, budgets and trainer.
        config['env']['Hardware_Hz_Enabled'] = int(getattr(self, 'hardware_hz_enabled', True))
        if getattr(self, 'uav_speed', None) is not None:
            config['env']['UAV_Speed_M_S'] = self.uav_speed
        config['env']['controller_profile'] = getattr(self, 'controller_profile', 'legacy')
        config['env']['map_level'] = getattr(self, 'map_level', None)
        config['env']['fixed_map_xml'] = getattr(self, 'fixed_map_xml', None)
        self.env = self.EnvFactory.Create_Env(config['env'])
        if self.env is None or not self.env.Agents:
            raise RuntimeError('Cannot initialize the configured environment and UAV')
        for agent in self.env.Agents:
            if getattr(agent, 'Trainer', None) is None:
                raise RuntimeError('Trainer initialization failed for {!r}; check the trainer configuration and checkpoint files'.format(
                    getattr(agent, 'name', 'UAV')))

    def Init_Record_Mod(self):
        self._result_file = open(self.result_path, 'w', newline='', encoding='utf-8')
        self.CsvWriter = csv.writer(self._result_file)
        fields = self.CSV_FIELDS
        if getattr(self, 'controller_profile', 'legacy') == 'manuscript':
            fields = fields + self.MANUSCRIPT_CSV_FIELDS
        self.CsvWriter.writerow(fields)
        self._result_file.flush()

    @staticmethod
    def _number(value):
        return value.item() if hasattr(value, 'item') else value

    def record_list(self):
        if not self.infos:
            return
        agents = self.env.Agents
        agent = agents[0]
        train_time = sum(item.Train_time for item in agents) / len(agents)
        testing_time = sum(item.Testing_time for item in agents) / len(agents)
        for info in self.infos:
            distance = math.dist(
                (agent.position.x, agent.position.y, agent.position.z),
                (agent.goal.x, agent.goal.y, agent.goal.z))
            row = [
                info.get('sum_epoch', agent.Trainer.epoch), self.epoch,
                info.get('score', agent.score), info.get('average_score', agent.score),
                info.get('eps', 0), info.get('success', 0), info.get('lose', 0),
                info.get('meet_threaten', 0), info.get('loss', 0), info.get('step', agent.Step),
                train_time, testing_time, self.executed_time,
                datetime.datetime.now().isoformat(timespec='seconds'),
                info.get('control_updates', agent.Control_step),
                info.get('simulation_time_s', agent.sim_time),
                info.get('path_len', agent.path_len),
                info.get('virtual_obstacle_count', len(agent.active_virtual_obstacles)),
                info.get('active_step_budget', agent.Max_Step) / (2.0 * agent.control_hz),
                info.get('remaining_goal_distance_m', distance),
                info.get('html_path', getattr(self.env, 'last_render_path', '')),
                int(getattr(agent, 'hardware_hz_enabled', agent.control_hz == 480)),
                agent.control_hz, getattr(agent, 'control_dt', 1.0 / agent.control_hz),
                int(bool(getattr(self.env, 'dynamic_enabled', False))),
                'mixed' if getattr(self.env, 'dynamic_enabled', False) else 'static',
                getattr(agent, 'Max_V', ''),
            ]
            if getattr(self, 'controller_profile', 'legacy') == 'manuscript':
                scene = self.env.manuscript_scene
                row.extend((self.controller_profile, self.map_level,
                            scene.seed if scene is not None else None,
                            getattr(agent, 'terminal_info', '')))
            self.CsvWriter.writerow([self._number(value) for value in row])
            self._result_file.flush()
        self.infos.clear()

    def record(self, info=None):
        if info is not None:
            self.infos.append(info)
            self.record_list()

    def epsilon_annealing(self):
        slope = (self.min_eps - 1.0) / (self.max_eps_episode + 0.1)
        return max(slope * self.epoch + 1.0, self.min_eps)

    def print_configuration(self):
        agent = self.env.Agents[0]
        # Motion is configured when the first episode resets its scene.
        moving_count = 6 if self.env.dynamic_enabled else 0
        obstacle_speed = self.env.dynamic_speed if self.env.dynamic_enabled else 0.0
        print('Timing mode: {}'.format(
            'hardware 480 Hz' if agent.hardware_hz_enabled else 'one second per update'), flush=True)
        print('Control: {:g} Hz | each update: Step +2 | counter rate: {:g}/s'.format(
            agent.control_hz, 2 * agent.control_hz), flush=True)
        print('UAV speed: {:g} m/s'.format(agent.Max_V), flush=True)
        print('Displacement/update: UAV {:.9f} m | moving obstacle {:.9f} m'.format(
            agent.Max_V * agent.control_dt, obstacle_speed * agent.control_dt), flush=True)
        print('Episode budget: {:g} s = {:g} Step units | scene: {} static + {} moving'.format(
            agent.base_duration_s, agent.original_max_step,
            len(self.env.buildings) - moving_count, moving_count), flush=True)
        print('Sensors: RTK {:g} Hz | LiDAR {:g} Hz | stuck window {:g} s'.format(
            agent.freq_rtk, agent.freq_lidar, agent.stuck_window_seconds), flush=True)
        print('SAC decision/replay interval: {:g} s | training: {}'.format(
            agent.reference_dt, 'off' if self.no_train else 'on'), flush=True)
        if self.controller_profile == 'manuscript':
            print('Controller profile: manuscript | map level: {}'.format(
                self.map_level if self.map_level is not None else 'existing S2R reset'), flush=True)
        print('Checkpoints: {}'.format(agent.Trainer.checkpoint_dir), flush=True)
        print('Scores: {}'.format(self.result_path), flush=True)
        print('HTML: {}'.format(ROOT / 'DataBase' / 'experience'), flush=True)

    def StartAndTrain(self, flag=0):
        with tqdm(total=self.num_episodes, desc='Simulation', unit='episode', ascii=True) as progress:
            for _ in range(self.num_episodes):
                while self.stat == 2:
                    time.sleep(0.1)
                if self.stat == 3:
                    break
                self.epoch += 1
                if flag:
                    self.env.run_XML_scene()
                    progress.update(1)
                    continue
                started = time.perf_counter()
                info = self.env.run_eposide(self.epsilon_annealing())
                self.executed_time += time.perf_counter() - started
                self.infos.append(info)
                self.Max_score = max(self.Max_score, float(info['average_score']))
                if self.writer:
                    for name, key in [('Average_Score', 'average_score'), ('Loss', 'loss'),
                                      ('Success_Rate', 'success'), ('Collision', 'meet_threaten')]:
                        self.writer.add_scalar('Train/' + name, self._number(info[key]), self.epoch)
                    self.writer.add_scalar('Simulation/Time_s', self.env.Agents[0].sim_time, self.epoch)
                    self.writer.flush()
                self.record_list()
                progress.set_postfix(episode=self.epoch, score='{:.3f}'.format(info['average_score']),
                                     success=info['success'], sim_s='{:.2f}'.format(self.env.Agents[0].sim_time))
                progress.update(1)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self._result_file is not None:
            self._result_file.close()
            self._result_file = None
        for agent in getattr(self.env, 'Agents', []):
            record = getattr(agent, '_record_file', None)
            if record is not None and not record.closed:
                record.close()

    def Update_target(self): self.env.Trainer.hard_update()
    def Set_Batch_Size(self, value): self.BATCH_SIZE = value
    def Set_Gamma(self, value): self.gamma = value
    def Set_Learning_Rate(self, value): self.LEARNING_RATE = value
    def Set_Target_Update(self, value): self.TARGET_UPDATE = value
    def Set_Num_Episodes(self, value): self.num_episodes = value
    def Set_Print_Every(self, value): self.print_every = value
    def Set_Min_Eps(self, value): self.min_eps = value
    def Set_Max_Eps_Episode(self, value): self.max_eps_episode = value
    def Set_Space_Dim(self, value): self.space_dim = value
    def Set_Action_Dim(self, value): self.action_dim = value


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return number


def prompt_choice(question, choices):
    """Validate a startup choice before creating the environment or trainer."""
    accepted = {str(value): value for value in choices}
    while True:
        try:
            answer = input(question).strip()
        except EOFError as error:
            raise ValueError(
                'No startup selection received. Enter one of {}, or specify '
                '--hardware-hz 0|1 --scene static|mixed --uav-speed 1|4 '
                'for a noninteractive run.'.format(', '.join(accepted))) from error
        if answer in accepted:
            return accepted[answer]
        print('請輸入 {}。'.format(' 或 '.join(accepted)), flush=True)


def prompt_binary(question):
    return prompt_choice(question, (0, 1))


def prompt_uav_speed(question):
    return prompt_choice(question, (1, 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware-hz', type=int, choices=(0, 1), default=None,
                        help='1: 480 Hz; 0: one second per update; prompt when omitted')
    parser.add_argument('--scene', choices=('static', 'mixed'), default=None,
                        help='static: 26 fixed; mixed: 20 fixed + 6 moving; prompt when omitted')
    parser.add_argument('--uav-speed', type=int, choices=(1, 4), default=None,
                        help='UAV speed in m/s: 1 or 4; prompt when omitted')
    parser.add_argument('--episodes', type=positive_integer, help='override configured episode count')
    parser.add_argument('--max-seconds', type=positive_seconds, help='override episode simulation duration')
    parser.add_argument('--no-train', action='store_true', help='simulate without optimizing or saving weights')
    parser.add_argument('--controller-profile', choices=('legacy', 'manuscript'), default='legacy',
                        help='select the archived simulator or the manuscript VOAPF 2.0 implementation')
    parser.add_argument('--map-level', type=int, choices=(1, 2, 3),
                        help='use the manuscript seed schedule for a static scene')
    parser.add_argument('--fixed-map-xml', type=Path,
                        help='verified fixed non-overlapping map required for Level 1')
    args = parser.parse_args()
    sim = None
    try:
        hardware_hz = args.hardware_hz
        if hardware_hz is None:
            hardware_hz = prompt_binary('是否加入硬體 Hz 模擬？(1: 480 Hz；0: 每次更新 1 秒)：')
        scene = args.scene
        if scene is None:
            dynamic = prompt_binary('是否加入動態障礙物？(1: 20 靜態 + 6 動態；0: 26 個全靜態)：')
            scene = 'mixed' if dynamic else 'static'
        uav_speed = args.uav_speed
        if uav_speed is None:
            uav_speed = prompt_uav_speed('請選擇 UAV 速度 V (1: 1 m/s；4: 4 m/s)：')
        sim = simulator(args.episodes, args.max_seconds, args.no_train,
                        scene=scene, hardware_hz=bool(hardware_hz), uav_speed=uav_speed,
                        controller_profile=args.controller_profile, map_level=args.map_level,
                        fixed_map_xml=args.fixed_map_xml)
        sim.print_configuration()
        sim.StartAndTrain()
        return 0
    except KeyboardInterrupt:
        print('\nSimulation stopped; completed episode results are saved.', flush=True)
        return 130
    except Exception as error:
        print('Simulation failed: {}: {}'.format(type(error).__name__, error), file=sys.stderr)
        return 1
    finally:
        if sim is not None:
            sim.close()


if __name__ == '__main__':
    raise SystemExit(main())
