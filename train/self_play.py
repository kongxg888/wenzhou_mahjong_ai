"""
简化版自对弈训练 - 快速版
"""

import os
import time
import numpy as np
from collections import defaultdict

from env import WenzhouMahjongEnv


class FastTrainer:
    """快速训练器"""

    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.env = WenzhouMahjongEnv()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Q表 (简化)
        self.q_table = defaultdict(lambda: np.zeros(34))
        self.alpha = 0.1  # 学习率
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def train(self, num_episodes: int = 1000, print_freq: int = 100):
        """训练"""
        print(f"开始训练 {num_episodes} 局...")

        start = time.time()
        wins = {'p0': 0, 'p1': 0, 'draw': 0}

        for ep in range(num_episodes):
            winner = self._play_one_game()

            if winner == 0:
                wins['p0'] += 1
            elif winner == 1:
                wins['p1'] += 1
            else:
                wins['draw'] += 1

            # 衰减探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if (ep + 1) % print_freq == 0:
                elapsed = time.time() - start
                rate = wins['p0'] / (ep + 1) * 100
                print(f"Episode {ep+1}/{num_episodes} | "
                      f"胜率: {rate:.1f}% | "
                      f"用时: {elapsed:.1f}s | "
                      f"Eps: {self.epsilon:.3f}")

        print(f"训练完成! 用时: {time.time() - start:.1f}s")

    def _play_one_game(self) -> int:
        """玩一局并学习"""
        state = self.env.reset()

        # 状态历史
        state_history = {0: [], 1: []}
        action_history = {0: [], 1: []}
        reward_history = {0: [], 1: []}

        current = 1  # 庄家先出

        while True:
            # 获取状态
            state = self.env.get_state()
            hand = state['hand']
            legal = state['legal_actions']
            hand_key = tuple(sorted(hand))

            # epsilon-greedy选择动作
            if np.random.random() < self.epsilon:
                action = np.random.choice(legal)
            else:
                action = self._get_best_action(hand_key, legal)

            # 执行
            next_state, reward, done, info = self.env.step(action)

            # 记录
            state_history[current].append(hand_key)
            action_history[current].append(action)

            if done:
                winner = info.get('winner', -1)

                # 反向更新
                self._update_q(winner, state_history, action_history)

                return winner

            current = 1 - current

    def _get_best_action(self, hand_key, legal_actions):
        """选择Q值最大的合法动作"""
        q_values = [self.q_table[hand_key][a] for a in legal_actions]
        return legal_actions[np.argmax(q_values)]

    def _update_q(self, winner, state_history, action_history):
        """更新Q表"""
        for p in [0, 1]:
            # 简化的奖励分配
            if winner == p:
                reward = 1.0
            elif winner == 1 - p:
                reward = -0.5
            else:
                reward = 0.0

            # 反向更新
            for i in range(len(state_history[p])):
                state = state_history[p][-(i+1)]
                action = action_history[p][-(i+1)]
                self.q_table[state][action] += self.alpha * (reward - self.q_table[state][action])
                reward *= self.gamma  # 折扣

    def save(self, path: str):
        """保存"""
        np.save(path, dict(self.q_table))

    def load(self, path: str):
        """加载"""
        data = np.load(path, allow_pickle=True).item()
        self.q_table = defaultdict(lambda: np.zeros(34), data)


def main():
    trainer = FastTrainer()
    trainer.train(num_episodes=1000)


if __name__ == '__main__':
    main()
