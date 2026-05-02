"""
MCTS (Monte Carlo Tree Search) 决策器

用于增强AI决策质量
"""

import math
import random
from typing import List, Dict, Tuple, Callable, Optional
import numpy as np


class MCTSNode:
    """MCTS节点"""

    def __init__(self, state: 'MCTSState', parent: Optional['MCTSNode'] = None,
                 action: int = None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MCTSState:
    """MCTS使用的游戏状态"""

    def __init__(self, hand: List[int], caishen_id: int,
                 wall_remaining: int = 50):
        self.hand = hand.copy()
        self.caishen_id = caishen_id
        self.wall_remaining = wall_remaining

    def get_legal_actions(self) -> List[int]:
        """获取合法动作（排除财神和白板）"""
        legal = []
        for tile in set(self.hand):
            if tile != self.caishen_id and tile != 33:
                legal.append(tile)
        return legal if legal else list(set(self.hand))

    def do_action(self, action: int) -> 'MCTSState':
        """执行动作，返回新状态"""
        new_state = MCTSState(self.hand, self.caishen_id, self.wall_remaining - 1)
        if action in new_state.hand:
            new_state.hand.remove(action)
        return new_state

    def is_terminal(self) -> bool:
        """是否终局"""
        return self.wall_remaining <= 0


class MCTS:
    """
    MCTS搜索器

    使用UCB1公式选择节点
    """

    def __init__(self, caishen_id: int, c_puct: float = 1.5,
                 num_simulations: int = 100):
        self.caishen_id = caishen_id
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def search(self, state: MCTSState) -> int:
        """
        执行MCTS搜索

        Returns:
            best_action: 最佳动作
        """
        root = MCTSNode(state)

        for _ in range(self.num_simulations):
            node = self._select(root)
            value = self._simulate(node.state)
            self._backpropagate(node, value)

        return self._get_best_action(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择阶段 - 一直选到叶节点"""
        while node.is_expanded():
            node = self._ucb_select(node)
        return node

    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        """UCB1选择"""
        best_action = None
        best_ucb = float('-inf')

        for action, child in node.children.items():
            ucb = self._ucb(node, child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        return node.children[best_action]

    def _ucb(self, parent: MCTSNode, child: MCTSNode) -> float:
        """UCB1公式"""
        q = child.q_value
        u = self.c_puct * child.prior * math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
        return q + u

    def _simulate(self, state: MCTSState) -> float:
        """模拟阶段 - 随机 rollout 到终局"""
        depth = 0
        max_depth = 50

        while not state.is_terminal() and depth < max_depth:
            legal = state.get_legal_actions()
            if not legal:
                break
            action = random.choice(legal)
            state = state.do_action(action)
            depth += 1

        # 简化奖励：基于剩余牌数
        return (50 - state.wall_remaining) / 50.0

    def _backpropagate(self, node: MCTSNode, value: float):
        """反向传播"""
        while node:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def _get_best_action(self, root: MCTSNode) -> int:
        """选择访问次数最多的动作"""
        best_action = None
        best_count = -1

        for action, child in root.children.items():
            if child.visit_count > best_count:
                best_count = child.visit_count
                best_action = action

        return best_action if best_action is not None else 0


class MCTSPlayer:
    """
    基于MCTS的AI玩家

    结合规则和MCTS搜索
    """

    def __init__(self, caishen_id: int, use_mcts: bool = True,
                 num_simulations: int = 50):
        self.caishen_id = caishen_id
        self.use_mcts = use_mcts
        self.num_simulations = num_simulations

        if use_mcts:
            self.mcts = MCTS(caishen_id, num_simulations=num_simulations)

    def select_action(self, hand: List[int], legal_actions: List[int],
                     wall_remaining: int = 50) -> int:
        """
        选择动作

        Args:
            hand: 手牌
            legal_actions: 合法动作列表

        Returns:
            action: 选中的动作
        """
        if not legal_actions:
            return hand[0] if hand else 0

        if not self.use_mcts:
            return self._rule_based_select(hand, legal_actions)

        # MCTS搜索
        state = MCTSState(hand, self.caishen_id, wall_remaining)
        return self.mcts.search(state)

    def _rule_based_select(self, hand: List[int], legal_actions: List[int]) -> int:
        """
        基于规则的选牌

        策略：
        1. 优先打孤立牌
        2. 优先打边张
        3. 保留对子和刻子
        """
        from collections import Counter

        counts = Counter(hand)

        # 计算每张牌的优先级
        priorities = {}
        for tile in legal_actions:
            cnt = counts[tile]

            # 孤立牌（只有1张）
            if cnt == 1:
                priorities[tile] = 10.0
            # 有2张的牌（对子）
            elif cnt == 2:
                priorities[tile] = 0.0  # 保留
            # 有3张以上的牌（刻子）
            else:
                priorities[tile] = -10.0  # 绝对保留

            # 字牌优先打
            if tile >= 27:
                priorities[tile] += 5.0

            # 边张（1、9、2、8）优先打
            tile_in_suit = tile % 9
            if tile_in_suit in (0, 1, 7, 8):
                priorities[tile] += 3.0

        # 选择优先级最高的
        return max(legal_actions, key=lambda a: priorities.get(a, 0))
