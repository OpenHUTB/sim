# rl_algorithms/ppo/smooth_agent.py
# Smooth-PPO 智能体：平滑裁剪 + LayerNorm 网络（毕设创新算法）

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.base_agent import BaseAgent
from rl_algorithms.ppo.smooth_network import SmoothActor, SmoothCritic
from rl_algorithms.ppo.storage import RolloutStorage
from rl_algorithms.ppo.clip_utils import smooth_clip
from config.ppo_config import (
    STATE_SIZE, ACTION_SIZE,
    LR_ACTOR, LR_CRITIC,
    GAMMA, LAMBDA, EPS_CLIP,
    UPDATE_EVERY, UPDATE_POLICY_TIMES, BATCH_SIZE,
    VALUE_LOSS_COEF, ENTROPY_COEF, MAX_GRAD_NORM,
    SMOOTH_ENABLED, SMOOTH_EPS_LOW, SMOOTH_EPS_HIGH, SMOOTH_BETA,
)


class SmoothPPOAgent(BaseAgent):
    """
    Smooth-PPO 智能体（毕设创新算法）。

    双重创新：
    1. 网络层：LayerNorm 平滑特征分布，减少梯度震荡
    2. 裁剪层：平滑裁剪函数替代硬截断，在边界处连续过渡

    适用场景: #4 前车急刹, #5 旁车加塞（行为自然度要求高）
    """

    def __init__(self, state_size=None, action_size=None):
        state_size = state_size or STATE_SIZE
        action_size = action_size or ACTION_SIZE
        super().__init__(state_size, action_size, name="SmoothPPO")

        # 平滑网络（含 LayerNorm）
        self.actor = SmoothActor(state_size, action_size).to(self.device)
        self.critic = SmoothCritic(state_size).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.gamma = GAMMA
        self.lambd = LAMBDA
        self.eps_clip = EPS_CLIP
        self.update_every = UPDATE_EVERY
        self.k_epochs = UPDATE_POLICY_TIMES
        self.batch_size = BATCH_SIZE
        self.value_coef = VALUE_LOSS_COEF
        self.entropy_coef = ENTROPY_COEF
        self.max_grad_norm = MAX_GRAD_NORM

        # 平滑裁剪参数
        self.smooth_enabled = SMOOTH_ENABLED
        self.smooth_alpha = (SMOOTH_EPS_HIGH - SMOOTH_EPS_LOW) / 2.0

        self.storage = RolloutStorage()
        self.last_loss_info = {}

    # ================================================================
    # 核心接口
    # ================================================================

    def act(self, state, evaluate=False):
        with torch.no_grad():
            state_t = self.to_tensor(state).unsqueeze(0)
            probs = self.actor(state_t)
            dist = torch.distributions.Categorical(probs)
            if evaluate:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def store(self, state, action, log_prob, reward, next_state, done):
        self.storage.push(state, action, log_prob, reward, next_state, done)

    def train(self):
        """收集足够步数后执行 Smooth-PPO 更新"""
        if len(self.storage) < self.update_every:
            return None

        self.train_steps += 1

        states, actions, old_log_probs, rewards, next_states, dones = \
            self.storage.get_all()

        s = self.to_tensor(states)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_lp = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            values = self.critic(s).squeeze(-1)
            next_val = self.critic(
                self.to_tensor(next_states[-1:])).squeeze(-1).item()

        advantages = self._compute_gae(
            rewards, values.detach().cpu().numpy(), next_val, dones)
        advantages = self.to_tensor(advantages)
        returns = advantages + values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        n = len(states)
        for _ in range(self.k_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                s_b = s[idx]; a_b = a[idx]; old_lp_b = old_lp[idx]
                adv_b = advantages[idx]; ret_b = returns[idx]

                probs = self.actor(s_b)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(a_b)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_b)
                # 使用平滑裁剪（核心创新）
                clipped = smooth_clip(ratio, self.eps_clip, self.smooth_alpha)
                actor_loss = -torch.min(
                    ratio * adv_b, clipped * adv_b).mean()
                actor_loss = actor_loss - self.entropy_coef * entropy

                self.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                values_pred = self.critic(s_b).squeeze(-1)
                critic_loss = self.value_coef * nn.MSELoss()(values_pred, ret_b)

                self.critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        self.storage.clear()
        self.last_loss_info = {
            "actor_loss": total_actor_loss / max(self.k_epochs, 1),
            "critic_loss": total_critic_loss / max(self.k_epochs, 1),
        }
        return self.last_loss_info

    # ================================================================
    # GAE
    # ================================================================

    def _compute_gae(self, rewards, values, next_value, dones):
        T = len(rewards)
        vals = np.append(values, next_value)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * vals[t + 1] * (1 - dones[t]) - vals[t]
            gae = delta + self.gamma * self.lambd * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    # ================================================================
    # 持久化
    # ================================================================

    def _save_checkpoint(self, checkpoint, path):
        checkpoint.update({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        })
        torch.save(checkpoint, path)

    def _load_checkpoint(self, checkpoint):
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt"])
