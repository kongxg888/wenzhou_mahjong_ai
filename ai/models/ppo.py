"""
PPO (Proximal Policy Optimization) Agent

比DQN更稳定的策略梯度算法
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Optional
import random


class ActorNetwork(nn.Module):
    """Actor网络：输出动作概率"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class CriticNetwork(nn.Module):
    """Critic网络：输出状态价值"""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class PPOMemory:
    """PPO经验回放缓冲区"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.values = []
        self.dones = []

    def add(self, state, action, reward, prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.probs.append(prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.values = []
        self.dones = []

    def get_batch(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.probs),
            np.array(self.values),
            np.array(self.dones)
        )


class PPOAgent:
    """
    PPO智能体

    使用clip机制稳定训练
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 k_epochs: int = 4,
                 hidden_dim: int = 128,
                 device: str = 'cpu'):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            epsilon: clip参数
            k_epochs: 每次更新轮数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.k_epochs = k_epochs
        self.device = device

        # 网络
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(device)

        self.actor_old = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 经验回放
        self.memory = PPOMemory()

        # 训练状态
        self.training = False
        self.update_count = 0

    def select_action(self, state: np.ndarray, legal_actions: List[int]) -> Tuple[int, float]:
        """
        选择动作

        Returns:
            (action, log_prob)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.actor_old(state_tensor)[0]

        # 只考虑合法动作
        mask = torch.zeros_like(probs)
        for a in legal_actions:
            mask[a] = 1.0

        probs = probs * mask
        probs = probs / probs.sum() + 1e-10

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估状态-动作对"""
        probs = self.actor(state)
        dist = Categorical(probs)
        log_probs = dist.log_prob(action)
        values = self.critic(state).squeeze()
        return log_probs, values

    def update(self):
        """更新网络"""
        if len(self.memory.states) < 8:
            return

        states, actions, rewards, old_probs, old_values, dones = self.memory.get_batch()

        # 转为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)
        old_values = torch.FloatTensor(old_values).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算GAE
        advantages = self._compute_gae(rewards, old_values, dones)

        # 多次更新
        for _ in range(self.k_epochs):
            log_probs, values = self.evaluate(states, actions)

            # PPO loss
            ratios = torch.exp(log_probs - old_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, rewards + advantages)

            # 更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            (actor_loss + 0.5 * critic_loss).backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # 更新旧网络
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.memory.clear()
        self.update_count += 1

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """计算GAE (Generalized Advantage Estimation)"""
        advantages = torch.zeros_like(rewards)
        last_adv = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * 0.95 * (1 - dones[t]) * last_adv
            last_adv = advantages[t]

        return advantages

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_old': self.actor_old.state_dict(),
            'update_count': self.update_count,
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_old.load_state_dict(checkpoint['actor_old'])
        self.update_count = checkpoint.get('update_count', 0)

    def train(self):
        """设置为训练模式"""
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        """设置为评估模式"""
        self.training = False
        self.actor.eval()
        self.critic.eval()
