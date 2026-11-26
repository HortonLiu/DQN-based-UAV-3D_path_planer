from BaseClass.BaseTrainer import *
from BaseClass.CalMod import *
from BaseClass.BaseCNN import *
import os
import random
import numpy as np
import torch
import torch.nn.functional as F

root = os.path.dirname(os.path.abspath(__file__)) + '/../'


class RiskPrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, use_priority=True):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_increment = float(beta_increment)
        self.use_priority = bool(use_priority)
        self.buffer = []
        self.priorities = []
        self.memory = []

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
        max_priority = max(self.priorities, default=1.0)
        self.priorities.append(max_priority)

    def push(self, batch, error=0):
        self.memory.append(batch)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample2(self, batch_size):
        count = len(self.buffer)
        if count < batch_size:
            raise ValueError("not enough replay samples")

        if self.use_priority and sum(self.priorities) > 0:
            priorities = np.array(self.priorities, dtype=np.float64)
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(count, batch_size, replace=False, p=probabilities)
            self.beta = min(1.0, self.beta + self.beta_increment)
            weights = (count * probabilities[indices]) ** (-self.beta)
            weights = weights / (weights.max() + 1e-8)
        else:
            indices = np.array(random.sample(range(count), batch_size))
            weights = np.ones(batch_size, dtype=np.float32)

        transitions = [self.buffer[int(i)] for i in indices]
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done, indices.tolist(), weights.astype(np.float32)

    def update_priorities(self, indices, abs_errors, risks=None):
        if indices is None:
            return
        risks = np.zeros_like(abs_errors, dtype=np.float64) if risks is None else risks
        for idx, err, risk in zip(indices, abs_errors, risks):
            priority = (abs(float(err)) + 1e-5) * (1.0 + float(risk))
            self.priorities[int(idx)] = max(priority, 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)


class PG_ITD3_Trainer(BaseTrainer):
    def __init__(self, param: dict) -> None:
        super().__init__(param)
        self.name = param.get('name')
        self.manuscript_map_level = param.get('manuscript_map_level')
        self.checkpoint_directory = os.path.join(root, 'Mod')
        if self.manuscript_map_level is not None:
            if self.manuscript_map_level not in (2, 3):
                raise ValueError('Shared map level must be 2 or 3')
            self.checkpoint_directory = os.path.join(
                self.checkpoint_directory, 'manuscript', f'level{int(self.manuscript_map_level)}'
            )
        actor_param = param.get('actor')
        critic_param = param.get('critic')
        td3_param = param.get('TD3_param') or {}

        actor_lr = float(actor_param.get('lr'))
        critic_lr = float(critic_param.get('lr'))
        self.actor = self.NetworkFactory.Create_Network(actor_param).to(device)
        self.actor_target = self.NetworkFactory.Create_Network(actor_param).to(device)
        self.critic_1 = self.NetworkFactory.Create_Network(critic_param).to(device)
        self.critic_2 = self.NetworkFactory.Create_Network(critic_param).to(device)
        self.critic_target_1 = self.NetworkFactory.Create_Network(critic_param).to(device)
        self.critic_target_2 = self.NetworkFactory.Create_Network(critic_param).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.gamma = float(None2Value(td3_param.get('gamma'), self.gamma))
        self.tau = float(None2Value(td3_param.get('tau'), 0.005))
        self.policy_noise = float(None2Value(td3_param.get('policy_noise'), 0.2))
        self.noise_clip = float(None2Value(td3_param.get('noise_clip'), 0.5))
        self.policy_delay = int(None2Value(td3_param.get('policy_delay'), 2))
        self.exploration_noise = float(None2Value(td3_param.get('exploration_noise'), 0.2))
        self.exploration_noise_min = float(None2Value(td3_param.get('exploration_noise_min'), 0.02))
        self.action_bound = float(None2Value(actor_param.get('action_bound'), 1.0))

        self.risk_weight = float(None2Value(td3_param.get('risk_weight'), 1.0))
        self.aftershock_k = float(None2Value(td3_param.get('aftershock_k'), 0.2))
        self.aftershock_c = float(None2Value(td3_param.get('aftershock_c'), 0.1))
        self.aftershock_p = float(None2Value(td3_param.get('aftershock_p'), 1.1))
        self.aftershock_delta_t = float(None2Value(td3_param.get('aftershock_delta_t'), 1.0))
        self.repulsion_range = float(None2Value(td3_param.get('repulsion_range'), 160.0))
        self.state_scale = float(None2Value(td3_param.get('state_scale'), 500.0))

        self.IsPriority_Replay = int(None2Value(param.get('IsPriority_Replay'), 1))
        self.replay_memory = RiskPrioritizedReplayMemory(
            self.replay_size,
            alpha=float(None2Value(td3_param.get('priority_alpha'), 0.6)),
            beta=float(None2Value(td3_param.get('priority_beta'), 0.4)),
            beta_increment=float(None2Value(td3_param.get('priority_beta_increment'), 0.001)),
            use_priority=self.IsPriority_Replay == 1,
        )
        self.device = device
        self.epoch = 0
        self.loss = 0
        self.update_count = 0
        self.Load_Mod()

    def Load_Mod(self, Mod=None):
        if Mod is not None:
            self.actor.load_state_dict(Mod['actor_model'])
            self.critic_1.load_state_dict(Mod['critic_1_model'])
            self.critic_2.load_state_dict(Mod['critic_2_model'])
            return

        actor_path = os.path.join(self.checkpoint_directory, 'PG_ITD3_actor_%s.pth' % self.name)
        critic_1_path = os.path.join(self.checkpoint_directory, 'PG_ITD3_critic_1_%s.pth' % self.name)
        critic_2_path = os.path.join(self.checkpoint_directory, 'PG_ITD3_critic_2_%s.pth' % self.name)
        if not (os.path.exists(actor_path) and os.path.exists(critic_1_path) and os.path.exists(critic_2_path)):
            return
        try:
            actor_mod = torch.load(actor_path, map_location=self.device)
            critic_1_mod = torch.load(critic_1_path, map_location=self.device)
            critic_2_mod = torch.load(critic_2_path, map_location=self.device)
            self.actor.load_state_dict(actor_mod['model'])
            self.critic_1.load_state_dict(critic_1_mod['model'])
            self.critic_2.load_state_dict(critic_2_mod['model'])
            self.actor_optimizer.load_state_dict(actor_mod['optimizer'])
            self.critic_1_optimizer.load_state_dict(critic_1_mod['optimizer'])
            self.critic_2_optimizer.load_state_dict(critic_2_mod['optimizer'])
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target_1.load_state_dict(self.critic_1.state_dict())
            self.critic_target_2.load_state_dict(self.critic_2.state_dict())
            self.epoch = actor_mod.get('epoch', 0)
        except Exception as e:
            print(e.args)

    def save(self, directory=None):
        if directory is None:
            directory = self.checkpoint_directory
        elif self.manuscript_map_level is not None and os.path.realpath(directory) != os.path.realpath(self.checkpoint_directory):
            raise ValueError(f'Shared map checkpoints use {self.checkpoint_directory}')
        os.makedirs(directory, exist_ok=True)
        torch.save(
            {'model': self.actor.state_dict(), 'optimizer': self.actor_optimizer.state_dict(), 'epoch': self.epoch},
            os.path.join(directory, 'PG_ITD3_actor_%s.pth' % self.name),
        )
        torch.save(
            {'model': self.critic_1.state_dict(), 'optimizer': self.critic_1_optimizer.state_dict(), 'epoch': self.epoch},
            os.path.join(directory, 'PG_ITD3_critic_1_%s.pth' % self.name),
        )
        torch.save(
            {'model': self.critic_2.state_dict(), 'optimizer': self.critic_2_optimizer.state_dict(), 'epoch': self.epoch},
            os.path.join(directory, 'PG_ITD3_critic_2_%s.pth' % self.name),
        )

    def _soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def _state_aftershock_risk(self, states):
        states_np = states.detach().cpu().numpy()
        obstacle_vecs = states_np[:, :-3].reshape(states_np.shape[0], -1, 3) * self.state_scale
        dists = np.linalg.norm(obstacle_vecs, axis=2)
        dists[dists <= 1e-8] = np.inf
        min_dist = np.min(dists, axis=1)
        min_dist[np.isinf(min_dist)] = self.repulsion_range
        spatial_risk = np.exp(-min_dist / max(self.repulsion_range, 1e-6))

        if abs(self.aftershock_p - 1.0) < 1e-8:
            integral = self.aftershock_k * math.log((self.aftershock_c + self.aftershock_delta_t) / self.aftershock_c)
        else:
            integral = self.aftershock_k * (
                (self.aftershock_c + self.aftershock_delta_t) ** (1.0 - self.aftershock_p)
                - self.aftershock_c ** (1.0 - self.aftershock_p)
            ) / (1.0 - self.aftershock_p)
        temporal_risk = 1.0 - math.exp(-max(integral, 0.0))
        return self.risk_weight * temporal_risk * spatial_risk

    def update(self, transition_dict):
        self.epoch += 1
        train_result = {'sum_epoch': self.epoch, 'loss': self.loss}
        if not transition_dict or len(transition_dict.get('states', [])) == 0:
            return train_result

        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float, device=self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float, device=self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float, device=self.device).view(-1, 1)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float, device=self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float, device=self.device).view(-1, 1)

        weights = transition_dict.get('weights')
        if weights is None:
            weights = torch.ones_like(rewards)
        else:
            weights = torch.tensor(np.array(weights), dtype=torch.float, device=self.device).view(-1, 1)

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.action_bound, self.action_bound)
            target_q_1 = self.critic_target_1(next_states, next_actions)
            target_q_2 = self.critic_target_2(next_states, next_actions)
            target_q = rewards + self.gamma * (1.0 - dones) * torch.min(target_q_1, target_q_2)

        current_q_1 = self.critic_1(states, actions)
        current_q_2 = self.critic_2(states, actions)
        critic_1_loss = (weights * F.mse_loss(current_q_1, target_q, reduction='none')).mean()
        critic_2_loss = (weights * F.mse_loss(current_q_2, target_q, reduction='none')).mean()

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.Is_Train:
            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optimizer.step()

            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()

            self.update_count += 1
            if self.update_count % self.policy_delay == 0:
                actor_loss = -self.critic_1(states, self.actor(states)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self._soft_update(self.actor, self.actor_target)
                self._soft_update(self.critic_1, self.critic_target_1)
                self._soft_update(self.critic_2, self.critic_target_2)

        abs_errors = torch.abs(torch.min(current_q_1, current_q_2) - target_q).detach().cpu().numpy().reshape(-1)
        risks = self._state_aftershock_risk(states)
        self.replay_memory.update_priorities(transition_dict.get('idx'), abs_errors, risks)

        if self.epoch % self.save_loop == 0:
            self.save()

        train_result['loss'] = (critic_1_loss + critic_2_loss + actor_loss.detach()).detach()
        return train_result

    def get_action(self, state, eps):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=self.device)
            action = self.actor(state).cpu().numpy()[0]
        if self.Is_Train:
            noise_scale = max(self.exploration_noise_min, self.exploration_noise * float(eps))
            action = action + np.random.normal(0.0, noise_scale, size=action.shape)
        action = np.clip(action, -self.action_bound, self.action_bound)
        return action.tolist()

    def Push_Replay(self, Experience, error=0):
        return None
