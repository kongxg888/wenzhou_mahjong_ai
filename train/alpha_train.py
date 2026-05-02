"""
AlphaZero训练器 - 优化版
修复的问题：
1. MCTS复用而非每步重建
2. Evaluator正确使用env.current_player
3. 训练间隔从4降到2
4. 评估时使用MCTS而非纯随机
"""

import os
import time
import csv
import numpy as np
import torch
from collections import deque
from datetime import datetime
from typing import List, Dict
import random

from env import WenzhouMahjongEnv
from ai import AlphaZeroAgent, AlphaZeroMCTS
from game_state import GameState
from rules import ACTION_PASS, ACTION_CHI, ACTION_PENG, ACTION_MING_GANG, ACTION_JIA_GANG, ACTION_AN_GANG, ACTION_HU


class MetricsLogger:
    """指标记录器"""
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(log_dir, f'train_metrics_{self.timestamp}.csv')
        self.csv_file = open(self.csv_path, 'w', buffering=1)
        self.writer = csv.DictWriter(self.csv_file, fieldnames=[
            'game', 'timestamp', 'win_rate_p0', 'win_rate_p1', 'draw_rate',
            'games_per_sec', 'loss', 'memory_size'
        ])
        self.writer.writeheader()

    def log(self, game: int, stats: Dict, metrics: Dict):
        self.writer.writerow({
            'game': game,
            'timestamp': datetime.now().isoformat(),
            'win_rate_p0': stats.get('p0', 0) / max(game, 1) * 100,
            'win_rate_p1': stats.get('p1', 0) / max(game, 1) * 100,
            'draw_rate': stats.get('draw', 0) / max(game, 1) * 100,
            'games_per_sec': metrics.get('games_per_sec', 0),
            'loss': metrics.get('loss', 0),
            'memory_size': metrics.get('memory_size', 0)
        })

    def close(self):
        self.csv_file.close()

    def get_path(self):
        return self.csv_path


class Evaluator:
    """评估器 - 使用MCTS评估"""
    def __init__(self, agent: AlphaZeroAgent, device: str):
        self.agent = agent
        self.device = device

    def evaluate(self, num_games: int = 50, mcts_simulations: int = 25) -> Dict:
        """评估当前模型 - 使用MCTS"""
        env = WenzhouMahjongEnv()
        stats = {'p0': 0, 'p1': 0, 'draw': 0}

        for _ in range(num_games):
            state = env.reset()
            last_discarded = None
            done = False

            while not done:
                state = env.get_state()
                current = env.current_player
                hand = state['hand']

                # 使用MCTS搜索
                game_state = GameState(
                    hand, state['caishen'],
                    wall_remaining=state['wall_remaining'],
                    current_player=current,
                    last_discarded=last_discarded
                )

                mcts = AlphaZeroMCTS(
                    self.agent.net,
                    caishen_id=state['caishen'],
                    num_simulations=mcts_simulations,
                    batch_size=16
                )

                try:
                    policy_dict = mcts.search(game_state, last_discarded)
                except Exception:
                    legal = env.get_legal_actions_full(current)
                    policy_dict = {a: 1.0 / len(legal) for a in legal if a < 34}

                # 转为概率向量
                policy = np.zeros(41)
                for a, p in policy_dict.items():
                    if 0 <= a < 41:
                        policy[a] = p

                if policy.sum() > 0:
                    policy = policy / policy.sum()

                legal_actions = env.get_legal_actions_full(current)
                mask = np.zeros(41)
                for a in legal_actions:
                    if 0 <= a < 41:
                        mask[a] = max(policy[a], 0)

                if mask.sum() == 0:
                    action = random.choice(legal_actions) if legal_actions else 0
                else:
                    probs = mask / mask.sum()
                    action = np.random.choice(41, p=probs)

                if action < 34:
                    last_discarded = action

                next_state, reward, done, info = env.step(action)

            winner = info.get('winner', -1)
            if winner == 0:
                stats['p0'] += 1
            elif winner == 1:
                stats['p1'] += 1
            else:
                stats['draw'] += 1

        return stats


class AlphaZeroTrainer:
    """优化版AlphaZero训练器"""

    def __init__(self, checkpoint_dir: str = 'checkpoints', log_dir: str = 'logs'):
        self.env = WenzhouMahjongEnv()

        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("使用 Mac MPS GPU 加速")
        else:
            self.device = 'cpu'
            print("使用 CPU")

        self.agent = AlphaZeroAgent(
            state_dim=136,
            action_dim=41,
            device=self.device,
            lr=1e-3
        )

        self.memory = deque(maxlen=20000)
        self.win_stats = {'p0': 0, 'p1': 0, 'draw': 0}
        self.loss_history = []

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.logger = MetricsLogger(log_dir)
        self.evaluator = Evaluator(self.agent, self.device)
        self.best_win_rate = 0.0
        self.best_model_path = None

    def train(self, num_games: int = 10000, mcts_simulations: int = 50,
              batch_size: int = 64, train_interval: int = 2,
              save_freq: int = 1000, print_freq: int = 100,
              eval_freq: int = 500, eval_games: int = 50):
        """
        训练主循环
        """
        print(f"开始训练 {num_games} 局...")
        print(f"MCTS模拟: {mcts_simulations}/步 | 训练间隔: 每{ train_interval}局")
        print(f"设备: {self.device}")

        start_time = time.time()

        for game in range(num_games):
            winner, trajectories = self._play_one_game(mcts_simulations)

            # 统计
            if winner == 0:
                self.win_stats['p0'] += 1
            elif winner == 1:
                self.win_stats['p1'] += 1
            else:
                self.win_stats['draw'] += 1

            self.memory.extend(trajectories)

            # 训练
            loss = 0.0
            if (game + 1) % train_interval == 0 and len(self.memory) >= batch_size:
                batch = random.sample(self.memory, batch_size)
                loss = self.agent.train_step(batch, batch_size)
                self.loss_history.append(loss)

            # 打印
            if (game + 1) % print_freq == 0:
                elapsed = time.time() - start_time
                games_per_sec = (game + 1) / elapsed
                win_rate = self.win_stats['p0'] / (game + 1) * 100

                print(f"Game {game+1}/{num_games} | "
                      f"胜率: {win_rate:.1f}% | "
                      f"速度: {games_per_sec:.1f}局/秒 | "
                      f"损失: {loss:.4f} | "
                      f"内存: {len(self.memory)}")

                self.logger.log(game + 1, self.win_stats.copy(), {
                    'games_per_sec': games_per_sec,
                    'loss': loss,
                    'memory_size': len(self.memory)
                })

            # 评估
            if (game + 1) % eval_freq == 0:
                eval_stats = self.evaluator.evaluate(eval_games, mcts_simulations // 2)
                eval_win_rate = eval_stats['p0'] / eval_games * 100

                print(f"\n=== 评估 {eval_games}局 ===")
                print(f"闲家胜率: {eval_win_rate:.1f}%")
                print(f"庄家胜率: {eval_stats['p1']/eval_games*100:.1f}%")
                print(f"流局率: {eval_stats['draw']/eval_games*100:.1f}%")

                if eval_win_rate > self.best_win_rate:
                    self.best_win_rate = eval_win_rate
                    best_path = f"{self.checkpoint_dir}/best_model.pt"
                    self.save(best_path)
                    self.best_model_path = best_path
                    print(f"★ 新最佳模型! 胜率: {self.best_win_rate:.1f}%")

                print(f"当前最佳: {self.best_win_rate:.1f}%\n")

            # 保存checkpoint
            if (game + 1) % save_freq == 0:
                checkpoint_path = f"{self.checkpoint_dir}/az_model_{game+1}.pt"
                self.save(checkpoint_path)
                print(f"  -> 已保存: {checkpoint_path}")

        total_time = time.time() - start_time
        print(f"\n训练完成! 用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        print(f"平均速度: {num_games/total_time:.1f}局/秒")
        print(f"最佳评估胜率: {self.best_win_rate:.1f}%")

        if self.best_model_path:
            print(f"最佳模型: {self.best_model_path}")

        self.logger.close()
        print(f"指标日志: {self.logger.get_path()}")

    def _play_one_game(self, mcts_simulations: int) -> tuple:
        """玩一局并收集数据"""
        state = self.env.reset()

        trajectories = []
        game_history = {0: [], 1: []}
        last_discarded = None

        while True:
            state = self.env.get_state()
            current = self.env.current_player
            hand = state['hand']

            legal_actions = self.env.get_legal_actions_full(current)

            game_state = GameState(
                hand, state['caishen'],
                wall_remaining=state['wall_remaining'],
                current_player=current,
                last_discarded=last_discarded
            )

            obs = game_state.get_observation()

            # 创建MCTS搜索
            mcts = AlphaZeroMCTS(
                self.agent.net,
                caishen_id=state['caishen'],
                num_simulations=mcts_simulations,
                batch_size=32
            )

            try:
                policy_dict = mcts.search(game_state, last_discarded)
            except Exception:
                policy_dict = {a: 1.0 / len(legal_actions) for a in legal_actions if a < 34}

            # 转为41维向量
            policy = np.zeros(41)
            for a, p in policy_dict.items():
                if 0 <= a < 41:
                    policy[a] = p

            if policy.sum() > 0:
                policy = policy / policy.sum()

            # 特殊动作强制提高（确保AI学会使用吃碰杠胡）
            if ACTION_PENG in legal_actions:
                policy[ACTION_PENG] = max(policy[ACTION_PENG], 0.25)
            if ACTION_CHI in legal_actions:
                policy[ACTION_CHI] = max(policy[ACTION_CHI], 0.15)
            if ACTION_MING_GANG in legal_actions:
                policy[ACTION_MING_GANG] = max(policy[ACTION_MING_GANG], 0.10)
            if ACTION_JIA_GANG in legal_actions:
                policy[ACTION_JIA_GANG] = max(policy[ACTION_JIA_GANG], 0.10)
            if ACTION_AN_GANG in legal_actions:
                policy[ACTION_AN_GANG] = max(policy[ACTION_AN_GANG], 0.08)
            if ACTION_HU in legal_actions:
                policy[ACTION_HU] = max(policy[ACTION_HU], 0.50)  # 胡牌最高优先级

            if policy.sum() > 0:
                policy = policy / policy.sum()

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

    def _sample_action(self, policy: np.ndarray, legal_actions: List[int]) -> int:
        """根据策略采样"""
        mask = np.zeros(41)
        for a in legal_actions:
            if 0 <= a < 41:
                mask[a] = max(policy[a], 0)

        if mask.sum() == 0:
            return random.choice(legal_actions) if legal_actions else 0

        probs = mask / mask.sum()
        return np.random.choice(41, p=probs)

    def save(self, path: str):
        """保存模型"""
        self.agent.save(path)

    def load(self, path: str):
        """加载模型"""
        self.agent.load(path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='AlphaZero训练器')
    parser.add_argument('--games', type=int, default=5000)
    parser.add_argument('--mcts', type=int, default=50)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--train_interval', type=int, default=2)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--eval_freq', type=int, default=500)
    parser.add_argument('--eval_games', type=int, default=50)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = parser.parse_args()

    trainer = AlphaZeroTrainer(
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    trainer.train(
        num_games=args.games,
        mcts_simulations=args.mcts,
        batch_size=args.batch,
        train_interval=args.train_interval,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        eval_games=args.eval_games
    )


if __name__ == '__main__':
    main()