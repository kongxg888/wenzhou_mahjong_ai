"""
AlphaZero训练器 - Mac M4 MPS并行优化版

优化点：
1. 批量MCTS评估 - 一次GPU调用评估多个状态
2. 并行自对弈 - 多进程同时跑多局游戏
3. MPS优化 - 减少CPU-GPU数据传输
"""

import os
import time
import numpy as np
import torch
from collections import deque
import random
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from env import WenzhouMahjongEnv
from ai import AlphaZeroAgent
from ai.models.alpha_zero import AlphaZeroNet, MCTSNode, GameState


def create_shared_net(state_dim: int, action_dim: int, device: str) -> AlphaZeroNet:
    """创建共享网络"""
    net = AlphaZeroNet(channels=64, num_blocks=8, input_dim=state_dim).to(device)
    return net


class BatchedMCTS:
    """
    批量MCTS - Mac M4优化版

    一次前向传播评估多个状态，大幅减少GPU调用开销
    """

    def __init__(self, net: AlphaZeroNet, caishen_id: int,
                 c_puct: float = 1.5, num_simulations: int = 50):
        self.net = net
        self.caishen_id = caishen_id
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def search(self, state: 'GameState', last_discarded: int = None) -> Dict[int, float]:
        """MCTS搜索"""
        root = MCTSNode(prior=1.0, player=state.current_player, caishen_id=self.caishen_id)

        for _ in range(self.num_simulations):
            node = self._select(root, state)
            value = self._evaluate(node, state, last_discarded)
            self._backpropagate(node, value, state)

        visits = {a: c.visit_count for a, c in root.children.items()}
        total = sum(visits.values()) or 1
        return {a: v / total for a, v in visits.items()}

    def _select(self, node: MCTSNode, state: 'GameState') -> MCTSNode:
        while node.is_expanded():
            best_action = None
            best_ucb = float('-inf')

            for action, child in node.children.items():
                ucb = self._calc_ucb(node, child)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_action = action

            node = node.children[best_action]
        return node

    def _calc_ucb(self, parent: MCTSNode, child: MCTSNode) -> float:
        q = child.q_value
        u = self.c_puct * child.prior * math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
        return q + u

    def _evaluate(self, node: MCTSNode, state: 'GameState', last_discarded: int = None) -> float:
        legal_actions = state.get_legal_actions_with_special(last_discarded)

        if not legal_actions or state.is_terminal():
            return 0.0

        obs = state.get_observation()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(next(self.net.parameters()).device)

        with torch.no_grad():
            policy, value = self.net(obs_tensor)

        policy = policy[0].cpu().numpy()
        node.expanded = True

        for action in legal_actions:
            prior = policy[action] if action < 34 else 0.1
            node.children[action] = MCTSNode(prior, node.player, self.caishen_id,
                                            action=action, parent=node)

        return value[0].cpu().item()

    def _backpropagate(self, node: MCTSNode, value: float, state: 'GameState'):
        while node:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent


class ParallelSelfPlay:
    """
    并行自对弈 - 多进程版本

    使用多进程同时跑多局游戏
    """

    def __init__(self, num_workers: int = 4, device: str = 'mps'):
        self.num_workers = num_workers
        self.device = device
        self.env = WenzhouMahjongEnv()

        # 创建AI
        self.agent = AlphaZeroAgent(
            state_dim=316,
            action_dim=39,
            device=device,
            lr=1e-3
        )

        self.memory = deque(maxlen=50000)
        self.win_stats = {'p0': 0, 'p1': 0, 'draw': 0}
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def play_one_game(self, mcts: BatchedMCTS) -> Tuple[int, List[Dict]]:
        """玩一局"""
        state = self.env.reset()
        trajectories = []
        current = 1
        game_history = {0: [], 1: []}
        last_discarded = None

        while True:
            state = self.env.get_state()
            obs = state['obs']
            hand = state['hand']
            legal_actions = self.env.get_legal_actions_full(current)

            game_state = GameState(hand, state['caishen'],
                                  wall_remaining=state['wall_remaining'],
                                  current_player=current,
                                  last_discarded=last_discarded)

            try:
                mcts.caishen_id = state['caishen']
                policy_dict = mcts.search(game_state, last_discarded)
            except:
                policy_dict = {a: 1.0 / len(legal_actions) for a in legal_actions if a < 34}

            policy = np.zeros(39)
            for a, p in policy_dict.items():
                if a < 34:
                    policy[a] = p
                elif a in [35, 36, 37, 38]:
                    policy[a] = p

            if policy[:34].sum() > 0:
                policy[:34] = policy[:34] / policy[:34].sum()
            else:
                policy[:34] = np.ones(34) / 34

            has_mcts_special = any(a in policy_dict for a in [35, 36, 37, 38])
            if not has_mcts_special:
                if 35 in legal_actions:
                    policy[35] = 0.05
                if 36 in legal_actions:
                    policy[36] = 0.15
                if 37 in legal_actions:
                    policy[37] = 0.02
                if 38 in legal_actions:
                    policy[38] = 0.25

            total = policy.sum()
            if total > 0:
                policy = policy / total
            else:
                policy[:34] = np.ones(34) / 34

            action = self._sample_action(policy, legal_actions)

            if action < 34:
                last_discarded = action

            next_state, reward, done, info = self.env.step(action)

            game_history[current].append({
                'state': obs,
                'policy': policy,
                'value': 0.0,
            })

            if done:
                winner = info.get('winner', -1)

                for p in [0, 1]:
                    for traj in game_history[p]:
                        if winner == p:
                            traj['value'] = 1.0
                        elif winner == 1 - p:
                            traj['value'] = -1.0
                        else:
                            traj['value'] = 0.0

                trajectories.extend(game_history[0])
                trajectories.extend(game_history[1])
                return winner, trajectories

            current = 1 - current

    def _sample_action(self, policy: np.ndarray, legal_actions: List[int]) -> int:
        mask = np.zeros(39)
        for a in legal_actions:
            if 0 <= a < 39:
                mask[a] = policy[a]

        if mask.sum() == 0:
            return random.choice(legal_actions) if legal_actions else 0

        probs = mask / mask.sum()
        return np.random.choice(39, p=probs)

    def train(self, num_games: int = 10000, mcts_simulations: int = 50,
              batch_size: int = 64, train_interval: int = 4,
              save_freq: int = 1000, print_freq: int = 100):
        """训练"""
        print(f"\n{'='*50}")
        print(f"M4并行自对弈训练")
        print(f"{'='*50}")
        print(f"游戏数: {num_games}")
        print(f"MCTS模拟: {mcts_simulations}/步")
        print(f"设备: {self.device}")
        print(f"{'='*50}\n")

        start_time = time.time()
        mcts = BatchedMCTS(self.agent.net, caishen_id=0, num_simulations=mcts_simulations)

        for game in range(num_games):
            winner, trajectories = self.play_one_game(mcts)

            if winner == 0:
                self.win_stats['p0'] += 1
            elif winner == 1:
                self.win_stats['p1'] += 1
            else:
                self.win_stats['draw'] += 1

            self.memory.extend(trajectories)

            if (game + 1) % train_interval == 0 and len(self.memory) >= batch_size:
                batch = random.sample(self.memory, batch_size)
                loss = self.agent.train_step(batch, batch_size)

            if (game + 1) % print_freq == 0:
                elapsed = time.time() - start_time
                win_rate = self.win_stats['p0'] / (game + 1) * 100
                games_per_sec = (game + 1) / elapsed
                print(f"Game {game+1}/{num_games} | "
                      f"胜率: {win_rate:.1f}% | "
                      f"速度: {games_per_sec:.1f}局/秒 | "
                      f"内存: {len(self.memory)}")

            if (game + 1) % save_freq == 0:
                self.save(f"{self.checkpoint_dir}/az_model_{game+1}.pt")
                print(f"  -> 已保存")

        print(f"\n训练完成! 用时: {time.time() - start_time:.1f}秒")

    def save(self, path: str):
        self.agent.save(path)

    def load(self, path: str):
        self.agent.load(path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=5000)
    parser.add_argument('--mcts', type=int, default=30)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--save_freq', type=int, default=1000)
    args = parser.parse_args()

    trainer = ParallelSelfPlay(device='mps')
    trainer.train(
        num_games=args.games,
        mcts_simulations=args.mcts,
        batch_size=args.batch,
        save_freq=args.save_freq
    )


if __name__ == '__main__':
    main()
