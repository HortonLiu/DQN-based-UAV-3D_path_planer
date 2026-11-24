# Portions adapted from Hands-on Reinforcement Learning (DDPG example in chapter 13).
# https://github.com/boyu-ai/Hands-on-RL/tree/4a151a4bfeafd3061c0e0ef59bece25ca8fb2bf8
# RLGF added configurable replay and checkpointing.
# Apache-2.0 terms are in the repository root under licenses/Hands-on-RL-LICENSE.

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from BaseClass.BaseTrainer import BaseTrainer
from BaseClass.CalMod import None2Value, device

Path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(Path)

root = os.path.dirname(os.path.abspath(__file__)) + "/../"


class DDPG_Trainer(BaseTrainer):
    def __init__(self, param: dict) -> None:
        super().__init__(param)
        self.name = param.get("name")
        self.device = device
        self.tau = float(None2Value(param.get("tau"), 0.005))
        self.Update_loop = int(None2Value(param.get("Update_loop"), 1))
        self.action_bound = float(None2Value(param.get("action_bound"), 1.0))
        self.exploration_noise = float(None2Value(param.get("exploration_noise"), 0.2))
        self.exploration_noise_min = float(None2Value(param.get("exploration_noise_min"), 0.02))
        self.load_model = int(float(None2Value(param.get("Load_Model"), 1)))
        self.epoch = 0
        self.loss = 0.0

        actor_param = param.get("actor")
        critic_param = param.get("critic")
        actor_lr = float(actor_param.get("lr"))
        critic_lr = float(critic_param.get("lr"))

        self.actor = self.NetworkFactory.Create_Network(actor_param).to(self.device)
        self.critic = self.NetworkFactory.Create_Network(critic_param).to(self.device)
        self.target_actor = self.NetworkFactory.Create_Network(actor_param).to(self.device)
        self.target_critic = self.NetworkFactory.Create_Network(critic_param).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        if self.load_model == 1:
            self.Load_Mod()
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def Load_Mod(self, Mod=None):
        if Mod is not None:
            self.actor.load_state_dict(Mod["actor_model"])
            self.critic.load_state_dict(Mod["critic_model"])
            self.epoch = Mod["epoch"]
            return

        if self.name is None:
            path_actor = root + "/Mod/DDPG_actor.pth"
            path_critic = root + "/Mod/DDPG_critic.pth"
        else:
            path_actor = root + "/Mod/DDPG_actor_" + self.name + ".pth"
            path_critic = root + "/Mod/DDPG_critic_" + self.name + ".pth"

        if os.path.exists(path_actor) and os.path.exists(path_critic):
            try:
                mod_actor = torch.load(path_actor, map_location=self.device)
                mod_critic = torch.load(path_critic, map_location=self.device)
                self.actor.load_state_dict(mod_actor["model"])
                self.critic.load_state_dict(mod_critic["model"])
                self.actor_optimizer.load_state_dict(mod_actor["optimizer"])
                self.critic_optimizer.load_state_dict(mod_critic["optimizer"])
                self.epoch = mod_actor["epoch"]
            except Exception as e:
                print(e.args)

    def save(self, directory=root + "/Mod/"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        actor_state = {
            "model": self.actor.state_dict(),
            "optimizer": self.actor_optimizer.state_dict(),
            "epoch": self.epoch,
        }
        torch.save(actor_state, "%s/DDPG_actor_%s.pth" % (directory, self.name))
        critic_state = {
            "model": self.critic.state_dict(),
            "optimizer": self.critic_optimizer.state_dict(),
            "epoch": self.epoch,
        }
        torch.save(critic_state, "%s/DDPG_critic_%s.pth" % (directory, self.name))

    def update(self, transition_dict):
        self.epoch += 1
        train_result = {"sum_epoch": self.epoch, "loss": 0}
        if not transition_dict or len(transition_dict.get("states", [])) == 0:
            return train_result

        states = torch.tensor(np.array(transition_dict["states"]), dtype=torch.float, device=self.device)
        actions = torch.tensor(np.array(transition_dict["actions"]), dtype=torch.float, device=self.device)
        rewards = torch.tensor(np.array(transition_dict["rewards"]), dtype=torch.float, device=self.device).view(-1, 1)
        next_states = torch.tensor(np.array(transition_dict["next_states"]), dtype=torch.float, device=self.device)
        dones = torch.tensor(np.array(transition_dict["dones"]), dtype=torch.float, device=self.device).view(-1, 1)

        if actions.dim() == 1:
            actions = actions.view(states.size(0), -1)

        if self.Is_Train:
            with torch.no_grad():
                next_actions = self.target_actor(next_states).clamp(-self.action_bound, self.action_bound)
                next_q_values = self.target_critic(next_states, next_actions)
                q_targets = rewards + self.gamma * next_q_values * (1.0 - dones)

            current_q = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q, q_targets)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic, self.target_critic)
            loss = (critic_loss + actor_loss.detach()).detach()
        else:
            loss = torch.tensor(0.0, device=self.device)

        if self.epoch % self.save_loop == 0:
            self.save()

        train_result["loss"] = loss
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

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def Push_Replay(self, Experience, error=0):
        return None
