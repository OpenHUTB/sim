# rl_algorithms/ppo/agent.py
# 标准 PPO 智能体：Actor-Critic + GAE + 标准裁剪

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.base_agent import BaseAgent
from rl_algorithms.ppo.network import Actor, Critic
from rl_algorithms.ppo.storage import RolloutStorage
from rl_algorithms.ppo.clip_utils import standard_clip
from config.ppo_config import (
    STATE_SIZE, ACTION_SIZE,
    LR_ACTOR, LR_CRITIC,
    GAMMA, LAMBDA, EPS_CLIP,
    UPDATE_EVERY, UPDATE_POLICY_TIMES, BATCH_SIZE,
    VALUE_LOSS_COEF, ENTROPY_COEF, MAX_GRAD_NORM,
)


class PPOAgent(BaseAgent):
    """
    标准 PPO 智能体（Proximal Policy Optimization）。

    适用场景: #1 大雨跟车, #6 行人横穿, #8 行人闯红灯
    """

    def __init__(self, state_size=None, action_size=None):
        state_size = state_size or STATE_SIZE
        action_size = action_size or ACTION_SIZE
        super().__init__(state_size, action_size, name="PPO")

        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size).to(self.device)

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

        self.storage = RolloutStorage()

        # 上次训练的 loss 信息
        self.last_loss_info = {}

    # ================================================================
    # 核心接口
    # ================================================================

    def act(self, state, evaluate=False):
        """
        采样动作。返回 (action, log_prob)。
        evaluate=True 时返回概率最高的动作。
        """
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
        """收集足够步数后执行 PPO 更新"""
        if len(self.storage) < self.update_every:
            return None

        self.train_steps += 1

        states, actions, old_log_probs, rewards, next_states, dones = \
            self.storage.get_all()

        s = self.to_tensor(states)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_lp = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)

        # 计算 GAE 和 returns
        with torch.no_grad():
            values = self.critic(s).squeeze(-1)
            next_val = self.critic(
                self.to_tensor(next_states[-1:])).squeeze(-1).item()

        advantages = self._compute_gae(
            rewards, values.detach().cpu().numpy(), next_val, dones)
        advantages = self.to_tensor(advantages)
        returns = advantages + values.detach()

        # 标准化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        n = len(states)
        for _ in range(self.k_epochs):
            # 小批量训练
            indices = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]

                s_batch = s[idx]
                a_batch = a[idx]
                old_lp_batch = old_lp[idx]
                adv_batch = advantages[idx]
                ret_batch = returns[idx]

                # Actor 损失
                probs = self.actor(s_batch)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(a_batch)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_batch)
                clipped = standard_clip(ratio, self.eps_clip)
                actor_loss = -torch.min(
                    ratio * adv_batch, clipped * adv_batch).mean()
                actor_loss = actor_loss - self.entropy_coef * entropy

                self.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                # Critic 损失
                values_pred = self.critic(s_batch).squeeze(-1)
                critic_loss = self.value_coef * nn.MSELoss()(values_pred, ret_batch)

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
        """
        计算 Generalized Advantage Estimation。

        rewards: list of float
        values: np.ndarray (T,)  V(s_t)
        next_value: float       V(s_{T+1})
        dones: np.ndarray (T,)
        """
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
