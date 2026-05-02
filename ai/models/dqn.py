"""
DQN (Deep Q-Network) Agent
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import random


class QNetwork(nn.Module):
    """Q网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN智能体

    支持：
    - 经验回放
    - 目标网络
    - epsilon-greedy探索
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 64,
                 memory_size: int = 100000,
                 target_update_freq: int = 100,
                 hidden_dim: int = 128,
                 device: str = 'cpu'):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            epsilon: 探索率
            epsilon_min: 最小探索率
            epsilon_decay: 探索率衰减
            batch_size: 批量大小
            memory_size: 经验回放大小
            target_update_freq: 目标网络更新频率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        # Q网络和目标网络
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # 经验回放
        self.memory = ReplayBuffer(memory_size)

        # 训练计数器
        self.train_count = 0

    def select_action(self, state: np.ndarray, legal_actions: List[int]) -> int:
        """
        选择动作（epsilon-greedy）

        Returns:
            action: 选中的动作
        """
        if random.random() < self.epsilon:
            # 探索：随机选择合法动作
            return random.choice(legal_actions)

        # 利用：选择Q值最大的合法动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)[0]

        # 屏蔽非法动作
        q_values_masked = torch.full((self.action_dim,), float('-inf'))
        for a in legal_actions:
            q_values_masked[a] = q_values[a]

        return q_values_masked.argmax().item()

    def update(self) -> float:
        """训练一次"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # 采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 转为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算当前Q值
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 计算目标Q值
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # 计算损失
        loss = self.loss_fn(current_q, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # 更新目标网络
        self.train_count += 1
        if self.train_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # 衰减epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon,
            'train_count': self.train_count,
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.train_count = checkpoint.get('train_count', 0)

    def train(self):
        """设置为训练模式"""
        self.q_net.train()

    def eval(self):
        """设置为评估模式"""
        self.q_net.eval()
