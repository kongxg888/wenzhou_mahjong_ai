"""
AlphaZero风格网络 - Mac M4优化版
PPO + MPS加速 + 轻量MCTS
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict
from game_state import GameState
from rules import ACTION_CHI, ACTION_PENG, ACTION_MING_GANG, ACTION_JIA_GANG, ACTION_AN_GANG, ACTION_HU
import math
import random
from collections import defaultdict


class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class AlphaZeroNet(nn.Module):
    """
    AlphaZero风格网络

    支持两种输入:
    - 136维 (34×4): 简化状态 (手牌+弃牌+牌墙+财神)
    - 316维: 环境完整obs
    输出: 策略(34维) + 价值(1维)
    """

    def __init__(self, channels: int = 64, num_blocks: int = 8, input_dim: int = 136):
        super().__init__()
        self.input_dim = input_dim

        if input_dim == 316:
            # 316维输入: 先投影到128维，再reshape成 (batch, 4, 32) -> conv2d
            self.input_proj = nn.Sequential(
                nn.Linear(316, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
            self.input_conv = nn.Sequential(
                nn.Conv2d(4, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            )
            self.final_flatten_size = 128  # 32*4*1
        else:
            # 136维输入 -> 34×4 图像格式
            self.input_proj = None
            self.input_conv = nn.Sequential(
                nn.Conv2d(34, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            )
            self.final_flatten_size = 128

        # 残差塔
        self.res_tower = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # 策略头 (动态输入大小)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 1)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 1, 128),
            nn.ReLU(),
            nn.Linear(128, 41)  # 0-33出牌 + 34过 + 35吃 + 36碰 + 37明杠 + 38加杠 + 39暗杠 + 40胡
        )

        # 价值头 (动态输入大小)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 1)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 136) 或 (batch, 316) 或 (batch, 34, 4, 1)
        """
        if x.dim() == 2:
            batch = x.shape[0]
            if self.input_dim == 316:
                # 316维输入: 线性投影 -> reshape成 (batch, 4, 32, 1)
                x = self.input_proj(x)
                # 128 -> (batch, 4, 32, 1) for conv2d (channels=4, height=32, width=1)
                x = x.view(batch, 4, 32, 1)
            else:
                # 136维输入 -> 34×4×1
                x = x.view(batch, 34, 4, 1)

        x = x / 255.0 if x.max() > 1.0 else x

        # 特征提取
        x = self.input_conv(x)
        for block in self.res_tower:
            x = block(x)

        # 输出 - 只对34维出牌计算softmax，特殊动作直接用logits
        policy_logits = self.policy_head(x)  # (batch, 41)
        policy_34 = policy_logits[:, :34]  # 出牌部分
        policy_softmax = torch.softmax(policy_34, dim=-1)  # 只对34维归一化

        # 组合: 前34维是softmax，后7维直接用logits
        policy = torch.cat([policy_softmax, policy_logits[:, 34:]], dim=-1)

        value = self.value_head(x)

        return policy, value.squeeze(-1)


class MCTSNode:
    """MCTS节点"""
    def __init__(self, prior: float, player: int, caishen_id: int, action: int = None,
                 parent: 'MCTSNode' = None, state: 'GameState' = None):
        self.player = player
        self.caishen_id = caishen_id
        self.action = action  # 导致这个节点的动作
        self.parent = parent   # 父节点
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.state = state  # 关联的游戏状态

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class AlphaZeroMCTS:
    """
    AlphaZero风格的MCTS - 优化版

    使用神经网络引导搜索，支持批量推理加速
    """

    def __init__(self, net: AlphaZeroNet, caishen_id: int,
                 c_puct: float = 1.5, num_simulations: int = 50,
                 batch_size: int = 32):
        self.net = net
        self.caishen_id = caishen_id
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.device = next(net.parameters()).device
        # 预分配观察buffer，避免每步分配
        self._obs_buffer = np.zeros(136, dtype=np.float32)

    def search(self, state: 'GameState', last_discarded: int = None) -> Dict[int, float]:
        """
        执行MCTS搜索

        Args:
            state: 游戏状态
            last_discarded: 对手上一步打出的牌（用于判断能否吃碰杠）

        Returns:
            动作 -> visit_count 分布
        """
        root = MCTSNode(prior=1.0, player=state.current_player, caishen_id=self.caishen_id, state=state)

        # 批量评估：收集需要评估的叶节点
        batch_states = []
        batch_nodes = []

        for sim_idx in range(self.num_simulations):
            # 深拷贝状态
            state_copy = self._copy_state(state)
            node = self._select(root, state_copy, last_discarded)

            if not node.is_expanded() and not state_copy.is_terminal():
                # 收集到批量
                batch_states.append(state_copy.get_observation())
                node.state = state_copy  # 关联状态
                batch_nodes.append(node)

                # 达到批量大小或最后一个，执行批量推理
                if len(batch_states) >= self.batch_size or sim_idx == self.num_simulations - 1:
                    self._batch_evaluate(batch_nodes, batch_states, last_discarded)
                    batch_states = []
                    batch_nodes = []
            else:
                # 终态直接评估
                value = 0.0
                self._backpropagate(node, value)

        # 处理剩余节点
        if batch_states:
            self._batch_evaluate(batch_nodes, batch_states, last_discarded)

        # 返回策略分布
        visits = {a: c.visit_count for a, c in root.children.items()}
        total = sum(visits.values()) or 1
        return {a: v / total for a, v in visits.items()}

    def _batch_evaluate(self, nodes: List[MCTSNode], states: List[np.ndarray], last_discarded: int):
        """批量评估多个节点"""
        if not states:
            return

        # 批量构建输入
        obs_batch = np.stack(states)  # (batch, 136)
        obs_tensor = torch.from_numpy(obs_batch).float().to(self.device)

        with torch.no_grad():
            policies, values = self.net(obs_tensor)

        policies = policies.cpu().numpy()
        values = values.cpu().numpy()

        for i, node in enumerate(nodes):
            node.expanded = True
            # 使用关联的state获取legal_actions
            if node.state is not None:
                legal_actions = node.state.get_legal_actions_with_special(last_discarded)
            else:
                legal_actions = list(range(34))  # fallback

            for action in legal_actions:
                if action < 34:
                    prior = max(policies[i, action], 0.01)
                elif action == ACTION_HU:
                    prior = 0.5  # 胡牌高优先级
                elif action == ACTION_PENG:
                    prior = 0.3  # 碰中等优先级
                elif action == ACTION_CHI:
                    prior = 0.15  # 吃较低优先级
                elif action == ACTION_MING_GANG:
                    prior = 0.1   # 明杠低优先级
                elif action == ACTION_JIA_GANG:
                    prior = 0.08  # 加杠较低
                elif action == ACTION_AN_GANG:
                    prior = 0.05  # 暗杠最低
                else:
                    prior = 0.1
                node.children[action] = MCTSNode(prior, node.player, self.caishen_id,
                                               action=action, parent=node, state=None)

            self._backpropagate(node, values[i])

    def _copy_state(self, state: 'GameState') -> 'GameState':
        """深拷贝游戏状态"""
        new_state = GameState(
            hand=state.hand.copy(),
            caishen_id=state.caishen_id,
            wall_remaining=state.wall_remaining,
            current_player=state.current_player,
            last_discarded=state.last_discarded,
            fulus=state.fulus.copy() if getattr(state, 'fulus', None) else [],
            opponent_discards=getattr(state, 'opponent_discards', []).copy() if getattr(state, 'opponent_discards', None) else []
        )
        new_state.discarded = state.discarded.copy() if getattr(state, 'discarded', None) else []
        return new_state

    def _select(self, node: MCTSNode, state: 'GameState', last_discarded: int = None) -> MCTSNode:
        """选择阶段 - 选择最佳子节点直到到达未展开节点"""
        while node.is_expanded():
            # UCB选择
            best_action = None
            best_ucb = float('-inf')

            for action, child in node.children.items():
                ucb = self._calc_ucb(node, child)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_action = action

            # 执行动作更新状态
            state.do_action(best_action)
            node = node.children[best_action]

        return node

    def _calc_ucb(self, parent: MCTSNode, child: MCTSNode) -> float:
        """PUCT公式"""
        q = child.q_value
        u = self.c_puct * child.prior * math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
        return q + u

    def _evaluate(self, node: MCTSNode, state: 'GameState', last_discarded: int = None) -> float:
        """评估节点"""
        # 获取完整合法动作（包括吃碰杠胡 35-38）
        legal_actions = state.get_legal_actions_with_special(last_discarded)

        if not legal_actions or state.is_terminal():
            return 0.0

        # 用网络评估 - 确保tensor在正确的设备上
        obs = state.get_observation()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(next(self.net.parameters()).device)

        with torch.no_grad():
            policy, value = self.net(obs_tensor)

        policy = policy[0].cpu().numpy()
        node.expanded = True

        # 展开节点 - 包括所有合法动作
        for action in legal_actions:
            if action < 34:
                prior = max(policy[action], 0.01)  # 出牌用网络输出，最小0.01
            elif action == ACTION_HU:
                prior = 0.5  # 胡牌高优先级
            elif action == ACTION_PENG:
                prior = 0.3  # 碰中等优先级
            elif action == ACTION_CHI:
                prior = 0.15  # 吃较低优先级
            elif action == ACTION_MING_GANG:
                prior = 0.1   # 明杠低优先级
            elif action == ACTION_JIA_GANG:
                prior = 0.08  # 加杠较低
            elif action == ACTION_AN_GANG:
                prior = 0.05  # 暗杠最低
            else:
                prior = 0.1
            node.children[action] = MCTSNode(prior, node.player, self.caishen_id, action=action, parent=node)

        return value[0].cpu().item()

    def _backpropagate(self, node: MCTSNode, value: float):
        """反向传播"""
        while node:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent


class AlphaZeroAgent:
    """
    AlphaZero智能体

    Mac M4优化版
    """

    def __init__(self, state_dim: int = 136, action_dim: int = 39,
                 device: str = 'mps', lr: float = 1e-3):
        """
        Args:
            state_dim: 状态维度 (136或316)
            action_dim: 动作维度 (默认39: 34出牌 + 5特殊动作)
            device: 'mps'(Mac GPU), 'cuda', 'cpu'
        """
        self.device = device
        self.action_dim = action_dim
        self.state_dim = state_dim

        # 网络
        self.net = AlphaZeroNet(channels=64, num_blocks=8, input_dim=state_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # 经验回放
        self.memory: List[Dict] = []

        # 训练状态
        self.train_count = 0

    def select_action(self, obs: np.ndarray, legal_actions: List[int],
                     use_mcts: bool = True) -> int:
        """
        选择动作

        Args:
            obs: 状态编码
            legal_actions: 合法动作列表 (0-33出牌, 34过, 35吃, 36碰, 37杠, 38胡)
            use_mcts: 是否使用MCTS

        Returns:
            选中的动作
        """
        if not use_mcts:
            return self._policy_select(obs, legal_actions)

        # MCTS搜索
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy, _ = self.net(obs_tensor)

        policy = policy[0].cpu().numpy()

        # 只考虑合法动作 (0-40)
        mask = np.zeros(41)
        for a in legal_actions:
            if 0 <= a < 41:
                mask[a] = policy[a]

        if mask.sum() == 0:
            return random.choice(legal_actions)

        # 采样
        probs = mask / mask.sum()
        action = np.random.choice(41, p=probs)

        return action

    def _policy_select(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        """直接策略选择"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy, _ = self.net(obs_tensor)

        policy = policy[0].cpu().numpy()

        # 屏蔽非法动作
        best_action = legal_actions[0] if legal_actions else 0
        best_prob = -1
        for a in legal_actions:
            if 0 <= a < 39 and policy[a] > best_prob:
                best_prob = policy[a]
                best_action = a

        return best_action

    def train_step(self, batch: List[Dict], batch_size: int = 32):
        """
        训练一步

        Args:
            batch: [(state, policy, value), ...]
            policy 是39维向量 (0-33出牌概率, 34过, 35吃, 36碰, 37杠, 38胡)
        """
        if len(batch) < batch_size:
            return 0.0

        # 采样
        batch_data = random.sample(batch, batch_size)

        # 修复: 先stack成numpy数组，避免从list创建tensor
        states_np = np.stack([d['state'] for d in batch_data])
        states = torch.from_numpy(states_np).float().to(self.device)

        # 目标策略: 41维 - 预分配并填充
        target_policies = torch.zeros(batch_size, 41)
        for i, d in enumerate(batch_data):
            p = d['policy']
            if len(p) == 34:
                p = np.pad(p, (0, 7), 'constant')
            target_policies[i] = torch.from_numpy(np.array(p[:41], dtype=np.float32))

        target_policies = target_policies.to(self.device)
        target_values = torch.from_numpy(np.array([d['value'] for d in batch_data])).float().to(self.device)

        # 前向传播
        policies, values = self.net(states)  # policies: (batch, 39)

        # KL散度 Loss - 安全计算
        # 避免log(0)和数值爆炸
        eps = 1e-10
        policies_safe = torch.clamp(policies, min=eps, max=1.0)
        target_safe = torch.clamp(target_policies, min=eps, max=1.0)

        # 归一化确保和为1
        policies_safe = policies_safe / policies_safe.sum(dim=-1, keepdim=True)
        target_safe = target_safe / target_safe.sum(dim=-1, keepdim=True)

        policy_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(policies_safe),
            target_safe
        )
        value_loss = nn.MSELoss()(values.squeeze(), target_values)
        loss = policy_loss + 0.5 * value_loss

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        self.train_count += 1
        return loss.item()

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_count': self.train_count,
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_count = checkpoint.get('train_count', 0)

    def eval_mode(self):
        """评估模式"""
        self.net.eval()

    def train_mode(self):
        """训练模式"""
        self.net.train()
